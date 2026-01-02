# Image Alt Text Generator

A FastAPI service that generates alt text/captions for images using vision-language models (VLMs) powered by MLX on Apple Silicon.

## Features

- **Automatic Captioning**: Generate descriptive alt text for images using Florence-2 models
- **FastAPI Integration**: RESTful API with endpoints for captioning and alt text generation
- **MLX Optimized**: Leverages Apple's MLX framework for efficient inference on Apple Silicon
- **Fallback Support**: Automatic fallback to quantized models if full-precision models fail to load
- **Configurable**: Environment variables for model selection, prompts, and inference parameters

## Prerequisites

- macOS with Apple Silicon (M1/M2/M3/M4 chips)
- Python 3.8+

## Installation

1. Clone or download this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Starting the Server

```bash
python app.py
```

The server will start on `http://localhost:8000` by default.

### API Endpoints

#### POST `/caption`
Returns both the alt text and word count.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: `image` (file upload)

**Response:**
```json
{
  "alt": "A descriptive caption of the image",
  "words": 5
}
```

#### POST `/alt`
Returns just the alt text with optional raw/debug output.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: `image` (file upload)
- Query Parameters:
  - `raw` (boolean): Include raw model output
  - `debug` (boolean): Include debug information

**Response:**
```json
{
  "alt": "A descriptive caption of the image"
}
```

### Configuration

Configure the service using environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `VLM_MODEL` | `microsoft/Florence-2-base-ft` | Vision-language model to use |
| `VLM_PROMPT` | `<CAPTION>` | Prompt template for the model |
| `INFERENCE_TIMEOUT_S` | `20` | Maximum inference time in seconds |
| `VLM_MAX_TOKENS` | `32` | Maximum tokens to generate |
| `VLM_TEMPERATURE` | `0.2` | Sampling temperature |

### Supported Models

- `microsoft/Florence-2-base-ft` (with fallback to `mlx-community/Florence-2-base-ft-8bit`)
- `microsoft/Florence-2-large-ft` (with fallback to `mlx-community/Florence-2-large-ft-8bit`)

## Development

The application uses:
- **FastAPI**: Web framework for building APIs
- **MLX-VLM**: Vision-language model inference on Apple Silicon
- **PIL/Pillow**: Image processing
- **Uvicorn**: ASGI server for FastAPI

