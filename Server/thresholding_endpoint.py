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

NOTE: OpenCV is used ONLY for image I/O (reading/encoding files).
      ALL thresholding logic is implemented purely with NumPy / scikit-image.
"""

import time
import base64
import numpy as np
import cv2                                      # I/O only
from skimage.filters import threshold_multiotsu, threshold_local
from flask import request, jsonify


# ── I/O helpers (OpenCV for reading/writing — NOT for thresholding logic) ────

def _read_grayscale(file_storage) -> np.ndarray:
    """Read an uploaded file and return a uint8 grayscale numpy array."""
    buf = np.frombuffer(file_storage.read(), np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)   # cv2 used only here
    if img is None:
        raise ValueError("Could not decode image.")
    return img


def _encode_png(img: np.ndarray) -> str:
    """Encode a numpy array to a base64 PNG string."""
    success, buf = cv2.imencode(".png", img)          # cv2 used only here
    if not success:
        raise RuntimeError("Failed to encode output image.")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


# ── Thresholding algorithms — pure NumPy / scikit-image, zero OpenCV ─────────

def optimal_threshold(gray: np.ndarray):
    """
    Iterative optimal thresholding (pure NumPy):
      1. Start with the image mean as the initial threshold T.
      2. Split pixels into two groups: ≤ T  and  > T.
      3. New T = average of the two group means.
      4. Repeat until convergence (< 0.5 grey-level change).
    Returns (binary_image uint8, threshold_value float).
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

    # Apply threshold manually with NumPy — no cv2.threshold
    binary = np.where(gray > T, np.uint8(255), np.uint8(0))
    return binary, round(T, 2)


def otsu_threshold(gray: np.ndarray):
    """
    Otsu's method — pure NumPy implementation.

    Maximises between-class variance by scanning all candidate thresholds
    derived from the normalised histogram.

    Returns (binary_image uint8, threshold_value float).
    """
    # Normalised histogram
    hist, bin_edges = np.histogram(gray.ravel(), bins=256, range=(0, 256))
    hist = hist.astype(np.float64)
    prob = hist / hist.sum()                    # probability of each level

    # Cumulative sums and cumulative means
    omega = np.cumsum(prob)                     # cumulative weight (class 1)
    mu    = np.cumsum(prob * np.arange(256))    # cumulative mean (class 1)
    mu_T  = mu[-1]                              # total mean

    # Between-class variance for every candidate threshold
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_b_sq = np.where(
            (omega > 0) & (omega < 1),
            (mu_T * omega - mu) ** 2 / (omega * (1.0 - omega)),
            0.0
        )

    T = int(np.argmax(sigma_b_sq))

    # Apply threshold with NumPy — no cv2.threshold
    binary = np.where(gray > T, np.uint8(255), np.uint8(0))
    return binary, round(float(T), 2)


def spectral_threshold(gray: np.ndarray, n_classes: int = 3):
    """
    Multi-level (spectral) thresholding via scikit-image's threshold_multiotsu,
    which generalises Otsu to N classes (N > 2).

    n_classes is clamped to a minimum of 3 (more than 2 modes as required).
    Large images are downscaled for threshold computation only; the full-
    resolution image is segmented with the derived thresholds.

    Returns (segmented_image uint8, list_of_thresholds).
    """
    n_classes = max(n_classes, 3)   # enforce "more than 2 modes"

    # Downscale for speed — threshold computation only
    h, w   = gray.shape
    max_dim = 512
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        # Use NumPy slicing to resize (nearest-neighbour) — no cv2.resize
        new_h, new_w = int(h * scale), int(w * scale)
        row_idx = (np.arange(new_h) * h / new_h).astype(int)
        col_idx = (np.arange(new_w) * w / new_w).astype(int)
        small   = gray[np.ix_(row_idx, col_idx)]
    else:
        small = gray

    # threshold_multiotsu is a scikit-image function — not OpenCV
    thresholds = threshold_multiotsu(small, classes=n_classes)

    # Map each pixel to its class index, then to an evenly-spaced grey level
    regions = np.digitize(gray, bins=thresholds)            # pure NumPy
    levels  = np.linspace(0, 255, n_classes, dtype=np.uint8)
    output  = levels[regions]                               # pure NumPy indexing

    return output.astype(np.uint8), [round(float(t), 2) for t in thresholds]


def local_threshold(gray: np.ndarray, block_size: int = 35, offset: int = 10):
    """
    Local (adaptive) thresholding via scikit-image's threshold_local.

    Each pixel's threshold is the Gaussian-weighted mean of its neighbourhood
    minus `offset`. No single global threshold exists.

    Returns (binary_image uint8, None).
    """
    if block_size % 2 == 0:     # block_size must be odd
        block_size += 1

    # threshold_local is a scikit-image function — not OpenCV
    thresh_map = threshold_local(gray, block_size=block_size, offset=offset)

    # Apply the per-pixel threshold map with NumPy — no cv2.threshold
    binary = np.where(gray > thresh_map, np.uint8(255), np.uint8(0))
    return binary, None


# ── Flask route ───────────────────────────────────────────────────────────────

def register_thresholding(app):
    """Call this from app.py:  register_thresholding(app)"""

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
            "threshold_value":          thresh_val,   # float | list[float] | None
        })