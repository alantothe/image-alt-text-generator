import asyncio
import io
import logging
import os
import re
import threading
from typing import Any

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from PIL import Image
import mlx.core as mx
from mlx_vlm import generate, load

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mlx_vlm")

app = FastAPI()

MODEL_NAME = os.getenv(
    "VLM_MODEL",
    "microsoft/Florence-2-base-ft",
)
FLORENCE_MLX_FALLBACKS = {
    "microsoft/Florence-2-base-ft": "mlx-community/Florence-2-base-ft-8bit",
    "microsoft/Florence-2-large-ft": "mlx-community/Florence-2-large-ft-8bit",
}
PROMPT = os.getenv(
    "VLM_PROMPT",
    "<CAPTION>",
).strip()
INFERENCE_TIMEOUT_S = float(os.getenv("INFERENCE_TIMEOUT_S", "20"))
MAX_TOKENS = int(os.getenv("VLM_MAX_TOKENS", "32"))
TEMPERATURE = float(os.getenv("VLM_TEMPERATURE", "0.2"))

MODEL = None
PROCESSOR = None
ACTIVE_MODEL_NAME = None
MODEL_LOCK = threading.Lock()


def patch_florence2_sanitize() -> None:
    try:
        from mlx_vlm.models import florence2
    except Exception:
        return
    language_model = getattr(florence2, "LanguageModel", None)
    if language_model is None or getattr(
        language_model, "_alt_sanitize_patched", False
    ):
        return

    def sanitize(self, weights):
        if "language_model.lm_head.weight" not in weights:
            for candidate in (
                "language_model.model.shared.weight",
                "language_model.shared.weight",
            ):
                if candidate in weights:
                    weights["language_model.lm_head.weight"] = weights[candidate]
                    break
        return weights

    language_model.sanitize = sanitize
    language_model._alt_sanitize_patched = True


def configure_metal_backend() -> str:
    if hasattr(mx, "set_default_device") and hasattr(mx, "gpu"):
        mx.set_default_device(mx.gpu)
        return "metal"
    return "metal-default"


def load_model_with_name(model_name: str, device_label: str):
    patch_florence2_sanitize()
    logger.info("Loading VLM model %s on %s", model_name, device_label)
    loaded = load(model_name, trust_remote_code=True)
    if not isinstance(loaded, tuple) or len(loaded) < 2:
        raise RuntimeError("load() did not return model and processor")
    return loaded


def load_model() -> None:
    global MODEL, PROCESSOR, ACTIVE_MODEL_NAME
    if MODEL is not None and PROCESSOR is not None:
        return
    with MODEL_LOCK:
        if MODEL is not None and PROCESSOR is not None:
            return
        device_label = configure_metal_backend()
        requested_model = MODEL_NAME
        logger.info("Requested VLM model %s", requested_model)
        try:
            loaded = load_model_with_name(requested_model, device_label)
            ACTIVE_MODEL_NAME = requested_model
        except Exception as exc:
            fallback = FLORENCE_MLX_FALLBACKS.get(requested_model)
            if fallback:
                logger.warning(
                    "Model load failed for %s: %s", requested_model, exc
                )
                logger.info("Trying MLX Florence-2 fallback %s", fallback)
                try:
                    loaded = load_model_with_name(fallback, device_label)
                    ACTIVE_MODEL_NAME = fallback
                except Exception as fallback_exc:
                    logger.exception("Failed to load fallback model")
                    raise RuntimeError(
                        f"model load failed: {fallback_exc}"
                    ) from fallback_exc
            else:
                logger.exception("Failed to load model")
                raise RuntimeError(f"model load failed: {exc}") from exc
        MODEL, PROCESSOR = loaded[:2]
        logger.info("Model loaded: %s", ACTIVE_MODEL_NAME)


@app.on_event("startup")
def startup() -> None:
    load_model()


def extract_text(output: Any) -> str:
    if isinstance(output, str):
        return output
    for attr in ("text", "caption", "response", "output"):
        value = getattr(output, attr, None)
        if isinstance(value, str):
            return value
    if isinstance(output, dict):
        for key in ("text", "caption", "response", "output"):
            value = output.get(key)
            if isinstance(value, str):
                return value
    if isinstance(output, (list, tuple)):
        for item in output:
            if isinstance(item, str):
                return item
            if isinstance(item, dict):
                for key in ("text", "caption", "response", "output"):
                    value = item.get(key)
                    if isinstance(value, str):
                        return value
    if hasattr(output, "__dict__"):
        for key in ("text", "caption", "response", "output"):
            value = output.__dict__.get(key)
            if isinstance(value, str):
                return value
    rendered = str(output)
    match = re.search(
        r"(?:text|caption|response|output)\s*[:=]\s*['\"]?([^'\"\n]+)",
        rendered,
        re.IGNORECASE,
    )
    if match:
        return match.group(1).strip()
    return rendered


def clean_model_output(text: str) -> str:
    """Clean common special tokens from model output."""
    if not text:
        return text

    # Remove common special tokens
    cleaned = re.sub(r'<[^>]*>', '', text)  # Remove <token> patterns
    cleaned = re.sub(r'^\s*s\s+', '', cleaned)  # Remove leading "s " that might be a token remnant
    cleaned = cleaned.strip()

    return cleaned or "unidentified image"




def run_inference(img: Image.Image) -> str:
    if MODEL is None or PROCESSOR is None:
        raise RuntimeError("model not loaded")
    output = generate(
        MODEL,
        PROCESSOR,
        PROMPT,
        img,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
    )
    raw_text = extract_text(output)
    return clean_model_output(raw_text)


async def infer_caption(image: UploadFile) -> str:
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="file must be an image")

    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty file")

    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="invalid image") from exc

    load_model()

    try:
        caption_text = await asyncio.wait_for(
            asyncio.to_thread(run_inference, img),
            timeout=INFERENCE_TIMEOUT_S,
        )
    except asyncio.TimeoutError as exc:
        raise HTTPException(status_code=504, detail="captioning timed out") from exc
    except Exception as exc:
        logger.exception("Captioning failed")
        raise HTTPException(status_code=500, detail=f"captioning failed: {exc}") from exc

    return caption_text


async def generate_alt(image: UploadFile) -> str:
    return await infer_caption(image)


@app.post("/caption")
async def caption(image: UploadFile = File(...)):
    alt = await generate_alt(image)
    return {"alt": alt, "words": len(alt.split())}


@app.post("/alt")
async def alt_only(
    image: UploadFile = File(...),
    raw: bool = Query(False),
    debug: bool = Query(False),
):
    caption_text = await infer_caption(image)
    if raw or debug:
        return {"alt": caption_text, "raw": caption_text}
    return {"alt": caption_text}
