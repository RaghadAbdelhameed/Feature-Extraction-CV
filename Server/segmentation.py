"""
Segmentation — implemented from scratch using NumPy only.
OpenCV used only for image color conversion and resizing.

Methods:
  - K-Means          (from scratch, vectorized, fast)
  - Region Growing   (from scratch, stack-based flood fill)
  - Agglomerative    (from scratch, bottom-up hierarchical)
  - Mean Shift       (from scratch, vectorized, subsampled)
"""

import numpy as np
import cv2
import time


# =========================================================================== #
#  HELPERS                                                                     #
# =========================================================================== #

def _resize_for_processing(image: np.ndarray, max_side: int) -> np.ndarray:
    h, w = image.shape[:2]
    scale = min(max_side / max(h, w), 1.0)
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return image.copy()


def _labels_to_image(labels: np.ndarray, pixels: np.ndarray,
                     h: int, w: int, n_ch: int) -> np.ndarray:
    """Replace each pixel with the mean color of its cluster."""
    n_clusters = int(labels.max()) + 1
    colors = np.zeros((n_clusters, n_ch), dtype=np.float32)
    for k in range(n_clusters):
        mask = labels == k
        if mask.any():
            colors[k] = pixels[mask].mean(axis=0)
    return colors[labels].reshape(h, w, n_ch).astype(np.uint8)


# =========================================================================== #
#  1. K-MEANS  (from scratch, vectorized)                                      #
# =========================================================================== #

def run_kmeans(image: np.ndarray, n_clusters: int = 4,
               max_iter: int = 15, tol: float = 1.0) -> tuple:
    """
    K-Means from scratch.
    - K-Means++ initialization for better convergence
    - Fully vectorized assignment step (no Python loops over pixels)
    - Runs on downscaled image then upscales result
    """
    small = _resize_for_processing(image, max_side=200)
    h, w  = small.shape[:2]
    n_ch  = 1 if len(small.shape) == 2 else 3
    pixels = small.reshape(-1, n_ch).astype(np.float32)
    N = len(pixels)

    # K-Means++ initialization
    rng = np.random.default_rng(42)
    centers = [pixels[int(rng.integers(0, N))].copy()]
    for _ in range(1, n_clusters):
        c_arr = np.array(centers)
        diff  = pixels[:, None, :] - c_arr[None, :, :]
        dists = np.min(np.sum(diff ** 2, axis=2), axis=1)
        probs = dists / (dists.sum() + 1e-10)
        chosen = int(rng.choice(N, p=probs))
        centers.append(pixels[chosen].copy())
    centers = np.array(centers, dtype=np.float32)

    labels = np.zeros(N, dtype=np.int32)

    for _ in range(max_iter):
        # Vectorized assignment: (N, K, D) -> distances -> argmin
        diff      = pixels[:, None, :] - centers[None, :, :]
        dists     = np.sum(diff ** 2, axis=2)
        new_labels = np.argmin(dists, axis=1)

        # Update centers
        new_centers = np.zeros_like(centers)
        for k in range(n_clusters):
            mask = new_labels == k
            new_centers[k] = pixels[mask].mean(axis=0) if mask.any() else centers[k]

        shift  = float(np.max(np.linalg.norm(new_centers - centers, axis=1)))
        centers = new_centers
        labels  = new_labels
        if shift < tol:
            break

    result_small = _labels_to_image(labels, pixels, h, w, n_ch)
    result = cv2.resize(result_small, (image.shape[1], image.shape[0]),
                        interpolation=cv2.INTER_NEAREST)
    return result, n_clusters


# =========================================================================== #
#  2. REGION GROWING  (from scratch, stack-based flood fill)                   #
# =========================================================================== #

def run_region_growing(image: np.ndarray,
                       seed_row: int = None, seed_col: int = None,
                       threshold: float = 15.0) -> tuple:
    """
    Region Growing from scratch.
    Starts at seed (default = center), expands to 4-connected neighbors
    whose intensity is within `threshold` of the seed value.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    else:
        gray = image.astype(np.float32)

    h, w = gray.shape
    seed_row = int(np.clip(seed_row if seed_row is not None else h // 2, 0, h - 1))
    seed_col = int(np.clip(seed_col if seed_col is not None else w // 2, 0, w - 1))

    seed_val = float(gray[seed_row, seed_col])
    visited  = np.zeros((h, w), dtype=bool)
    region   = np.zeros((h, w), dtype=bool)

    stack = [(seed_row, seed_col)]
    visited[seed_row, seed_col] = True

    while stack:
        r, c = stack.pop()
        if abs(float(gray[r, c]) - seed_val) <= threshold:
            region[r, c] = True
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc]:
                    visited[nr, nc] = True
                    stack.append((nr, nc))

    if len(image.shape) == 3:
        output = image.copy()
    else:
        output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Darken background so region stands out
    output[~region] = (output[~region] * 0.25).astype(np.uint8)
    return output, 2


# =========================================================================== #
#  3. AGGLOMERATIVE  (from scratch, bottom-up hierarchical)                    #
# =========================================================================== #

def run_agglomerative(image: np.ndarray, n_clusters: int = 4) -> tuple:
    """
    Agglomerative clustering from scratch.
    1. Sample representative pixels
    2. Greedily merge closest pair of clusters until n_clusters remain
    3. Assign all image pixels to nearest final cluster center
    """
    small = _resize_for_processing(image, max_side=100)
    h_s, w_s = small.shape[:2]
    n_ch = 1 if len(small.shape) == 2 else 3
    pixels_small = small.reshape(-1, n_ch).astype(np.float32)

    # Sample ~300 pixels for the clustering
    N = len(pixels_small)
    n_samples = min(300, N)
    rng = np.random.default_rng(42)
    idx = rng.choice(N, size=n_samples, replace=False)
    samples = pixels_small[idx].copy()

    # Each sample is its own cluster (store as list of pixel arrays)
    clusters = {i: samples[i:i+1].copy() for i in range(n_samples)}

    while len(clusters) > n_clusters:
        keys    = list(clusters.keys())
        centers = np.array([c.mean(axis=0) for c in clusters.values()])

        # Pairwise distances between centers
        diff  = centers[:, None, :] - centers[None, :, :]
        dists = np.sum(diff ** 2, axis=2)
        np.fill_diagonal(dists, np.inf)

        flat_idx = int(np.argmin(dists))
        i, j = divmod(flat_idx, len(keys))
        ki, kj = keys[i], keys[j]

        # Merge j into i
        clusters[ki] = np.vstack([clusters[ki], clusters[kj]])
        del clusters[kj]

    # Final centers
    final_centers = np.array([c.mean(axis=0) for c in clusters.values()])

    # Assign every pixel in the small image to nearest center
    diff   = pixels_small[:, None, :] - final_centers[None, :, :]
    dists  = np.sum(diff ** 2, axis=2)
    labels = np.argmin(dists, axis=1)

    result_small = _labels_to_image(labels, pixels_small, h_s, w_s, n_ch)
    result = cv2.resize(result_small, (image.shape[1], image.shape[0]),
                        interpolation=cv2.INTER_NEAREST)
    return result, n_clusters


# =========================================================================== #
#  4. MEAN SHIFT  (from scratch, vectorized)                                   #
# =========================================================================== #

def run_mean_shift(image: np.ndarray,
                   bandwidth: float = 60.0,
                   max_iter: int = 8,
                   tol: float = 2.0) -> tuple:
    """
    Mean Shift from scratch.
    - Downscales to 50px max for speed
    - Fully vectorized shift step (no Python loop over pixels)
    - Higher bandwidth = fewer, larger segments
    """
    small = _resize_for_processing(image, max_side=50)
    h, w  = small.shape[:2]
    n_ch  = 1 if len(small.shape) == 2 else 3
    pixels = small.reshape(-1, n_ch).astype(np.float32)
    N = len(pixels)

    # Each pixel is a seed that shifts toward local mean
    points = pixels.copy()
    bw_sq  = bandwidth ** 2

    for _ in range(max_iter):
        # Fully vectorized: (N, N, D)
        diff     = pixels[None, :, :] - points[:, None, :]   # (N, N, D)
        dists_sq = np.sum(diff ** 2, axis=2)                  # (N, N)
        in_win   = dists_sq <= bw_sq                          # (N, N) bool

        new_points = np.zeros_like(points)
        for i in range(N):
            if in_win[i].any():
                new_points[i] = pixels[in_win[i]].mean(axis=0)
            else:
                new_points[i] = points[i]

        shift  = float(np.max(np.linalg.norm(new_points - points, axis=1)))
        points = new_points
        if shift < tol:
            break

    # Merge converged seeds within bandwidth/2 of each other
    labels     = -np.ones(N, dtype=np.int32)
    cluster_id = 0
    for i in range(N):
        if labels[i] != -1:
            continue
        dists = np.linalg.norm(points - points[i], axis=1)
        same  = dists <= bandwidth * 0.8
        labels[same] = cluster_id
        cluster_id += 1

    n_segments = int(labels.max()) + 1

    result_small = _labels_to_image(labels, pixels, h, w, n_ch)
    result = cv2.resize(result_small, (image.shape[1], image.shape[0]),
                        interpolation=cv2.INTER_NEAREST)
    return result, n_segments


# =========================================================================== #
#  DISPATCHER                                                                  #
# =========================================================================== #

def segment_image(image_bgr: np.ndarray, method: str, params: dict):
    """
    Main entry point called from app.py.
    Returns (result_bgr, n_segments, elapsed_seconds)
    """
    t0 = time.time()

    n_clusters = int(params.get("n_clusters", 4))
    threshold  = float(params.get("threshold", 15.0))
    seed_row   = params.get("seed_row", None)
    seed_col   = params.get("seed_col", None)

    if method == "kmeans":
        result, n_seg = run_kmeans(image_bgr, n_clusters=n_clusters)

    elif method == "region_growing":
        result, n_seg = run_region_growing(
            image_bgr,
            seed_row=int(seed_row) if seed_row is not None else None,
            seed_col=int(seed_col) if seed_col is not None else None,
            threshold=threshold,
        )

    elif method == "agglomerative":
        result, n_seg = run_agglomerative(image_bgr, n_clusters=n_clusters)

    elif method == "mean_shift":
        result, n_seg = run_mean_shift(image_bgr)

    else:
        raise ValueError(f"Unknown method: {method}")

    elapsed = time.time() - t0

    # Ensure BGR output for encoding
    if len(result.shape) == 2:
        result_bgr = cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    else:
        result_bgr = result.astype(np.uint8)

    return result_bgr, n_seg, elapsed