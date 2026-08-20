"""
main.py
-------
FastAPI backend for the Fuzzy-Based Image Enhancement project.

Endpoints
---------
POST /api/enhance
    Accepts an uploaded image + algorithm parameters, runs the fuzzy
    enhancement pipeline (ImageEnh.py for grayscale, ColoredImageEnh.py
    for RGB), and returns the enhanced image (base64 PNG) along with
    processing time and before/after statistics.

Run locally:
    pip install -r requirements.txt
    uvicorn main:app --reload

The frontend (fuzzy-image-enhancement.html, or a React app calling this
API) can POST to /api/enhance instead of / in addition to the in-browser
JS implementation used for the instant-feedback demo.
"""

import base64
import io

import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError

from ImageEnh import enhance_grayscale, image_statistics
from ColoredImageEnh import enhance_color, is_grayscale, color_image_statistics

app = FastAPI(title="Fuzzy-Based Image Enhancement API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this for production deployments
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/jpg", "image/png"}
MAX_FILE_BYTES = 12 * 1024 * 1024  # 12MB
MAX_DIMENSION = 2000  # guard against extremely large uploads


def _pil_to_base64_png(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


@app.post("/api/enhance")
async def enhance_image(
    file: UploadFile = File(...),
    window_size: int = Form(9),
    gamma: float = Form(0.45),
    strength: float = Form(1.20),
    num_scales: int = Form(3),
):
    # ---- validation / error handling ----
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail="Unsupported format. Please upload JPG, JPEG or PNG.")

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="No image uploaded.")
    if len(raw) > MAX_FILE_BYTES:
        raise HTTPException(status_code=400, detail="File too large. Please upload an image under 12MB.")

    try:
        pil_img = Image.open(io.BytesIO(raw))
        pil_img.load()
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Corrupted or invalid image file.")

    if max(pil_img.size) > MAX_DIMENSION:
        scale = MAX_DIMENSION / max(pil_img.size)
        new_size = (int(pil_img.width * scale), int(pil_img.height * scale))
        pil_img = pil_img.resize(new_size)

    if window_size < 3 or window_size % 2 == 0:
        raise HTTPException(status_code=400, detail="window_size must be an odd integer >= 3.")
    if gamma <= 0 or strength <= 0:
        raise HTTPException(status_code=400, detail="gamma and strength must be positive numbers.")
    if not (1 <= num_scales <= 3):
        raise HTTPException(status_code=400, detail="num_scales must be between 1 and 3.")

    try:
        rgb_img = pil_img.convert("RGB")
        arr = np.array(rgb_img)  # (H, W, 3) uint8

        grayscale = is_grayscale(arr)
        if grayscale:
            enhanced_channel, meta = enhance_grayscale(
                arr[:, :, 0], window_size=window_size, gamma=gamma,
                strength=strength, num_scales=num_scales,
            )
            enhanced_arr = np.stack([enhanced_channel] * 3, axis=-1)
            orig_stats = image_statistics(arr[:, :, 0])
            enh_stats = image_statistics(enhanced_channel)
        else:
            enhanced_arr, meta = enhance_color(
                arr, window_size=window_size, gamma=gamma,
                strength=strength, num_scales=num_scales,
            )
            orig_stats = color_image_statistics(arr)
            enh_stats = color_image_statistics(enhanced_arr)

        enhanced_img = Image.fromarray(enhanced_arr.astype(np.uint8), mode="RGB")

    except Exception as exc:  # pragma: no cover - defensive catch-all
        raise HTTPException(status_code=500, detail=f"Processing failed: {exc}")

    return JSONResponse({
        "enhanced_image_base64": _pil_to_base64_png(enhanced_img),
        "is_grayscale": grayscale,
        "width": arr.shape[1],
        "height": arr.shape[0],
        "meta": meta,
        "original_stats": orig_stats,
        "enhanced_stats": enh_stats,
        "contrast_improvement": enh_stats["contrast"] - orig_stats["contrast"],
        "mean_intensity_change": enh_stats["mean"] - orig_stats["mean"],
    })


@app.get("/api/health")
async def health():
    return {"status": "ok"}
