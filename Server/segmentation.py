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
    # Shrink the image so clustering runs fast 
    # (max 200px on longest side)
    small = _resize_for_processing(image, max_side=200)
    h, w  = small.shape[:2]

    # Determine number of channels: 1 for grayscale, 3 for color
    n_ch  = 1 if len(small.shape) == 2 else 3

    # Flatten the image to a 2-D array of shape
    #  (N_pixels, n_channels) row-> pixel ..column-> channel
    pixels = small.reshape(-1, n_ch).astype(np.float32)
    N = len(pixels)

    # ---- K-Means++ initialization ----------------------------------------
    # Seed the RNG for reproducibility
    rng = np.random.default_rng(42)

    # Pick the very first center uniformly at random
    centers = [pixels[int(rng.integers(0, N))].copy()]

    for _ in range(1, n_clusters):
        # Build array of current centers: shape (k, n_ch)
        c_arr = np.array(centers)

        # Compute squared distance from every pixel to every current center:
    
        diff  = pixels[:, None, :] - c_arr[None, :, :]
        # For each pixel keep only the distance to its nearest center
        dists = np.min(np.sum(diff ** 2, axis=2), axis=1)

        # Convert distances to a probability distribution (farther = more likely)
        probs = dists / (dists.sum() + 1e-10)

        # Sample the next center proportional to squared distance
        chosen = int(rng.choice(N, p=probs))
        centers.append(pixels[chosen].copy())

    # Stack the list of centers into a single array: shape (n_clusters, n_ch)
    centers = np.array(centers, dtype=np.float32)

    # Initialize label array (each pixel's cluster index)
    labels = np.zeros(N, dtype=np.int32)

    # ---- Iterative assignment + update loop --------------------------------
    for _ in range(max_iter):
        # Vectorized distance computation: broadcast pixels against all centers
        # diff shape → (N, n_clusters, n_ch)
        diff      = pixels[:, None, :] - centers[None, :, :]
        # Sum squared differences over the channel axis → (N, n_clusters)
        dists     = np.sum(diff ** 2, axis=2)
        # Each pixel gets the index of the closest center
        new_labels = np.argmin(dists, axis=1)

        # Recompute each center as the mean of all pixels assigned to it
        new_centers = np.zeros_like(centers)
        for k in range(n_clusters):
            mask = new_labels == k
            # If no pixels fell into cluster k, keep the old center to avoid NaN
            new_centers[k] = pixels[mask].mean(axis=0) if mask.any() else centers[k]

        # Measure how far centers moved; stop early if movement is tiny
        shift  = float(np.max(np.linalg.norm(new_centers - centers, axis=1)))
        centers = new_centers
        labels  = new_labels
        if shift < tol:
            break

    # Colorize the small segmented image using per-cluster mean colors
    result_small = _labels_to_image(labels, pixels, h, w, n_ch)

    # Scale the result back up to the original image dimensions
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
    # Convert color image to grayscale so intensity comparisons are 1-D
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    else:
        gray = image.astype(np.float32)

    h, w = gray.shape

    # Default seed is the image center; clamp to valid pixel coordinates
    seed_row = int(np.clip(seed_row if seed_row is not None else h // 2, 0, h - 1))
    seed_col = int(np.clip(seed_col if seed_col is not None else w // 2, 0, w - 1))

    # Record the intensity at the seed pixel — the reference value for growing
    seed_val = float(gray[seed_row, seed_col])

    # visited: tracks pixels already pushed onto the stack (avoids duplicates)
    visited  = np.zeros((h, w), dtype=bool)
    # region: marks pixels that passed the threshold test and belong to the region
    region   = np.zeros((h, w), dtype=bool)

    # Initialise the stack with just the seed pixel
    stack = [(seed_row, seed_col)]
    visited[seed_row, seed_col] = True

    # ---- Flood-fill loop ---------------------------------------------------
    while stack:
        r, c = stack.pop()

        # Accept this pixel into the region if its intensity is close to the seed
        if abs(float(gray[r, c]) - seed_val) <= threshold:
            region[r, c] = True

            # Explore all 4-connected neighbours (up, down, left, right)
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                # Only push neighbours that are inside the image and not yet visited
                if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc]:
                    visited[nr, nc] = True
                    stack.append((nr, nc))

    # Build the output image: start from the original color image
    if len(image.shape) == 3:
        output = image.copy()
    else:
        # Grayscale source — convert to BGR so we can display it uniformly
        output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Darken pixels outside the grown region to make the result stand out visually
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
    # Downscale heavily so the O(n²) merge loop stays tractable
    small = _resize_for_processing(image, max_side=100)
    h_s, w_s = small.shape[:2]
    n_ch = 1 if len(small.shape) == 2 else 3

    # Flatten to (N_pixels, n_channels)
    pixels_small = small.reshape(-1, n_ch).astype(np.float32)

    # ---- Pixel sampling ----------------------------------------------------
    N = len(pixels_small)
    # Work with at most 300 representative pixels to keep pairwise ops cheap
    n_samples = min(300, N)
    rng = np.random.default_rng(42)
    # Draw indices without replacement for an unbiased sample
    idx = rng.choice(N, size=n_samples, replace=False)
    samples = pixels_small[idx].copy()

    # Start with every sample pixel as its own singleton cluster
    clusters = {i: samples[i:i+1].copy() for i in range(n_samples)}

    # ---- Bottom-up merging loop --------------------------------------------
    while len(clusters) > n_clusters:
        keys    = list(clusters.keys())
        # Compute the centroid (mean color) of each current cluster
        centers = np.array([c.mean(axis=0) for c in clusters.values()])

        # Pairwise squared distances between all cluster centroids:
        # diff shape → (K, K, n_ch); sum over channels → (K, K)
        diff  = centers[:, None, :] - centers[None, :, :]
        dists = np.sum(diff ** 2, axis=2)

        # Exclude self-distances so a cluster is never merged with itself
        np.fill_diagonal(dists, np.inf)

        # Find the pair of clusters with the smallest centroid distance
        flat_idx = int(np.argmin(dists))
        # Convert the flat index back to (row, col) in the distance matrix
        i, j = divmod(flat_idx, len(keys))
        ki, kj = keys[i], keys[j]

        # Merge cluster j into cluster i by concatenating their pixel arrays
        clusters[ki] = np.vstack([clusters[ki], clusters[kj]])
        # Remove the now-absorbed cluster
        del clusters[kj]

    # ---- Pixel assignment --------------------------------------------------
    # Compute the final centroid for each surviving cluster
    final_centers = np.array([c.mean(axis=0) for c in clusters.values()])

    # Assign every pixel in the (small) image to its nearest final centroid
    diff   = pixels_small[:, None, :] - final_centers[None, :, :]
    dists  = np.sum(diff ** 2, axis=2)
    labels = np.argmin(dists, axis=1)

    # Colorize and upscale back to the original resolution
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
    # Aggressively downscale — mean shift is O(N²) per iteration
    small = _resize_for_processing(image, max_side=50)
    h, w  = small.shape[:2]
    n_ch  = 1 if len(small.shape) == 2 else 3

    # Flatten to (N_pixels, n_channels)
    pixels = small.reshape(-1, n_ch).astype(np.float32)
    N = len(pixels)

    # Each pixel starts as its own "seed" that will drift toward a local mode
    points = pixels.copy()

    # Pre-compute squared bandwidth to avoid repeated squaring in the loop
    bw_sq  = bandwidth ** 2

    # ---- Iterative shift loop ----------------------------------------------
    for _ in range(max_iter):
        # Compute pairwise differences between all seeds and all data pixels:
        # pixels[None] → (1, N, D), points[:, None] → (N, 1, D)
        # diff shape → (N_seeds, N_pixels, n_channels)
        diff     = pixels[None, :, :] - points[:, None, :]

        # Squared Euclidean distance from each seed to each pixel → (N, N)
        dists_sq = np.sum(diff ** 2, axis=2)

        # Boolean mask: True where a pixel falls within the bandwidth window
        in_win   = dists_sq <= bw_sq

        new_points = np.zeros_like(points)
        for i in range(N):
            if in_win[i].any():
                # Shift seed i to the mean of all pixels inside its window
                new_points[i] = pixels[in_win[i]].mean(axis=0)
            else:
                # No neighbours found — seed stays put
                new_points[i] = points[i]

        # Measure the largest movement of any seed this iteration
        shift  = float(np.max(np.linalg.norm(new_points - points, axis=1)))
        points = new_points

        # Early stopping: if all seeds moved less than tol, they have converged
        if shift < tol:
            break

    # ---- Cluster labelling -------------------------------------------------
    # Start with all seeds unlabelled (-1)
    labels     = -np.ones(N, dtype=np.int32)
    cluster_id = 0

    for i in range(N):
        # Skip seeds that have already been assigned to a cluster
        if labels[i] != -1:
            continue

        # Find all seeds whose converged position is within 80% of bandwidth
        # from seed i — these are considered part of the same mode
        dists = np.linalg.norm(points - points[i], axis=1)
        same  = dists <= bandwidth * 0.8

        # Assign the same cluster id to all seeds near this mode
        labels[same] = cluster_id
        cluster_id += 1

    # Total number of discovered segments
    n_segments = int(labels.max()) + 1

    # Colorize small result and upscale to original resolution
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