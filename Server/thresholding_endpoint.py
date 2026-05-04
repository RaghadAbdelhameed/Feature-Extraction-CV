"""
Thresholding endpoint — add this to your existing Flask server file (app.py).

Requires:
    pip install flask flask-cors opencv-python scikit-image numpy pillow

Usage: POST http://127.0.0.1:5000/api/thresholding
  Form fields:
    image      — the uploaded image file (grayscale preferred; color auto-converted)
    method     — one of: "optimal" | "otsu" | "spectral" | "local"
    n_classes  — int, number of classes for spectral (default 3, min 3)
    block_size — int (odd), local neighbourhood size (default 35)
    offset     — int, local threshold offset (default 10)

Response JSON:
    result_image_base64       — base64-encoded PNG of the thresholded image
    computation_time_seconds  — float
    threshold_value           — float (global methods) or list[float] (spectral) or null (local)
"""

import time
import base64
import io
import numpy as np
import cv2
from flask import request, jsonify
from skimage.filters import threshold_multiotsu, threshold_local


# ── helpers ──────────────────────────────────────────────────────────────────

def _read_grayscale(file_storage):
    """Read an uploaded file and convert to uint8 grayscale numpy array."""
    buf = np.frombuffer(file_storage.read(), np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Could not decode image.")
    return img


def _encode_png(img: np.ndarray) -> str:
    """Encode a numpy image (grayscale or BGR) to a base64 PNG string."""
    success, buf = cv2.imencode(".png", img)
    if not success:
        raise RuntimeError("Failed to encode output image.")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


# ── thresholding algorithms ───────────────────────────────────────────────────

def optimal_threshold(gray: np.ndarray):
    """
    Iterative optimal thresholding:
      1. Start with mean as initial threshold T.
      2. Split pixels into two groups: ≤T and >T.
      3. New T = mean of both group means.
      4. Repeat until convergence.
    Returns (binary_image, threshold_value).
    """
    T = float(gray.mean())
    for _ in range(1000):
        lower = gray[gray <= T]
        upper = gray[gray >  T]
        if lower.size == 0 or upper.size == 0:
            break
        T_new = (lower.mean() + upper.mean()) / 2.0
        if abs(T_new - T) < 0.5:
            T = T_new
            break
        T = T_new

    _, binary = cv2.threshold(gray, T, 255, cv2.THRESH_BINARY)
    return binary, round(T, 2)


def otsu_threshold(gray: np.ndarray):
    """
    Otsu's method via OpenCV — maximises between-class variance.
    Returns (binary_image, threshold_value).
    """
    T, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return binary, round(float(T), 2)


def spectral_threshold(gray: np.ndarray, n_classes: int = 3):
    n_classes = max(n_classes, 3)

    # Resize large images for speed, then apply result to full size
    h, w = gray.shape
    max_dim = 512
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        small = cv2.resize(gray, (int(w * scale), int(h * scale)))
    else:
        small = gray

    thresholds = threshold_multiotsu(small, classes=n_classes)

    # Apply thresholds to the FULL resolution image
    regions = np.digitize(gray, bins=thresholds)
    levels  = np.linspace(0, 255, n_classes, dtype=np.uint8)
    output  = levels[regions]

    return output.astype(np.uint8), [round(float(t), 2) for t in thresholds]

def local_threshold(gray: np.ndarray, block_size: int = 35, offset: int = 10):
    """
    Local (adaptive) thresholding using scikit-image's gaussian method.
    Each pixel is compared to the weighted mean of its neighbourhood.
    Returns (binary_image, None)  — no single global threshold value.
    """
    # block_size must be odd
    if block_size % 2 == 0:
        block_size += 1

    thresh_map = threshold_local(gray, block_size=block_size, offset=offset)
    binary = (gray > thresh_map).astype(np.uint8) * 255
    return binary, None


# ── Flask route ───────────────────────────────────────────────────────────────

# Paste the function below into your existing Flask app.py, e.g.:
#
#   from thresholding_endpoint import register_thresholding
#   register_thresholding(app)
#
# OR simply copy the route function directly into app.py.

def register_thresholding(app):
    """Call this from app.py: register_thresholding(app)"""

    @app.route("/api/thresholding", methods=["POST"])
    def thresholding():
        if "image" not in request.files:
            return jsonify({"error": "No image provided"}), 400

        file       = request.files["image"]
        method     = request.form.get("method",     "otsu")
        n_classes  = int(request.form.get("n_classes",  3))
        block_size = int(request.form.get("block_size", 35))
        offset     = int(request.form.get("offset",     10))

        try:
            gray = _read_grayscale(file)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

        t0 = time.perf_counter()

        try:
            if method == "optimal":
                result, thresh_val = optimal_threshold(gray)
            elif method == "otsu":
                result, thresh_val = otsu_threshold(gray)
            elif method == "spectral":
                result, thresh_val = spectral_threshold(gray, n_classes)
            elif method == "local":
                result, thresh_val = local_threshold(gray, block_size, offset)
            else:
                return jsonify({"error": f"Unknown method: {method}"}), 400
        except Exception as e:
            return jsonify({"error": f"Processing failed: {str(e)}"}), 500

        elapsed = time.perf_counter() - t0

        return jsonify({
            "result_image_base64":      _encode_png(result),
            "computation_time_seconds": round(elapsed, 4),
            "threshold_value":          thresh_val,   # float | list | None
        })