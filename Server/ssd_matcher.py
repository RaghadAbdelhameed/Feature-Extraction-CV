"""
SSD Feature Matcher — from scratch.

Detection  : SIFT keypoints via sift_detector.detect_sift_features_fast()
Descriptors: Raw pixel patches extracted around each SIFT keypoint — pure NumPy
Matching   : SSD with ratio test — pure NumPy (vectorized)
Drawing    : OpenCV used only for line/circle rendering in visualisation
"""

import numpy as np
import cv2
import time
from typing import List, Tuple, Dict, Any

from utils import base64_to_image_array, image_array_to_base64


# =========================================================================== #
#  1. PATCH DESCRIPTOR EXTRACTION  — pure NumPy                               #
# =========================================================================== #

def extract_patch_descriptors(
    image: np.ndarray,
    keypoints: List[Dict],
    patch_size: int = 21
) -> Tuple[List[np.ndarray], List[Dict]]:
   # Ensure patch size is always odd
   #  so the keypoint sits exactly at the center
    if patch_size % 2 == 0:
        patch_size += 1
    # Half-width used to compute the patch boundaries around each keypoint    
    half = patch_size // 2
    h, w = image.shape[:2]
   #define descriptors and a valid kps list to append to them next
    descriptors: List[np.ndarray] = []
    valid_kps:   List[Dict]       = []

    for kp in keypoints:
        row = int(round(kp['x']))   # SIFT: x = row
        col = int(round(kp['y']))   # SIFT: y = col

        # Skip keypoints too close to the image border 
        # — their patch would go out of bounds
        if (row - half < 0 or row + half >= h or
                col - half < 0 or col + half >= w):
            continue
        # Crop the square patch centered on the keypoint
        #btkhly el kepoint fel nos w 7waleha el patch
        patch = image[row - half: row + half + 1,
                      col - half: col + half + 1].astype(np.float64)
        
        # Flatten the 2D patch into a 1D descriptor vector
        flat = patch.ravel()
        # Mean-normalize so SSD is invariant to additive brightness shifts
        #ex: one image is slightly darker 
        #without mean normalization the ssd will assume they're different
        #  even if it's the same photo with brightness difference
        flat = flat - flat.mean()
        descriptors.append(flat)
        valid_kps.append(kp)

    return descriptors, valid_kps


# =========================================================================== #
#  2. SSD MATCHING  — pure NumPy (vectorized)                                 #
# =========================================================================== #

def match_ssd(
    descriptors_A: List[np.ndarray],
    descriptors_B: List[np.ndarray],
    ratio_thresh: float = 0.75
) -> List[Tuple[int, int]]:
    
    matches: List[Tuple[int, int]] = []
    start = time.time()

    if not descriptors_A or not descriptors_B:
        return matches

    # Stack descriptor lists into 2D matrices for vectorized operations
    A = np.array(descriptors_A, dtype=np.float64)   # (N_A, D)
    B = np.array(descriptors_B, dtype=np.float64)   # (N_B, D)

    # Re-normalize each descriptor row to remove any remaining brightness bias
    # This makes SSD invariant to additive intensity shifts.
    A = A - A.mean(axis=1, keepdims=True)
    B = B - B.mean(axis=1, keepdims=True)
   
    # Compute squared L2 norms for each descriptor
    #Instead of computing (f - g)^2 directly, 
    # I use ||f||² - 2f·gᵀ + ||g||², which is mathematically equivalent 
    # but much faster using matrix operations (no python loop needed)
    A_sq  = np.sum(A ** 2, axis=1, keepdims=True)   # (N_A, 1)
    B_sq  = np.sum(B ** 2, axis=1, keepdims=True)   # (N_B, 1)
    ssd_matrix = A_sq - 2.0 * (A @ B.T) + B_sq.T   # (N_A, N_B)

    # Numerical noise can produce tiny negatives — clamp to 0
    ssd_matrix = np.maximum(ssd_matrix, 0.0)

#"For each descriptor in image 1, we find the best and second-best match in image 2. 
# If best/second < 0.75 the match is accepted. 
# The idea is: a good match should be clearly better than the runner-up
#   if two candidates are similarly close, the match is ambiguous and we reject it."

    for i in range(len(descriptors_A)):
        row = ssd_matrix[i]

        if len(row) < 2:
            continue
        # Sort SSD scores ascending — index 0 is the best (lowest) match
        idx        = np.argsort(row)        # ascending — lower SSD = better
        best_idx   = idx[0]
        second_idx = idx[1]
        best       = row[best_idx]
        second     = row[second_idx]

        # FIX: if second ≈ 0 the match is degenerate (two identical patches),
        # skip it.  Otherwise apply Lowe's ratio test normally.
        if second < 1e-8:
            continue
      #without this test we will get a lot of false matches 
        if (best / second) < ratio_thresh: #0.75
            matches.append((i, int(best_idx)))

    print(f"[SSD] match time: {time.time()-start:.3f}s  found: {len(matches)}")
    return matches


# =========================================================================== #
#  3. VISUALISATION  — OpenCV used only for drawing                           #
# =========================================================================== #

def visualize_ssd_matches(
    img1: np.ndarray,
    img2: np.ndarray,
    match_pairs:  List[Tuple[int, int]],
    valid_kps1:   List[Dict],
    valid_kps2:   List[Dict],
    descriptors1: List[np.ndarray],
    descriptors2: List[np.ndarray],
    max_lines: int = 60
) -> np.ndarray:
    """
    Side-by-side visualisation with coloured match lines.

    Colour by relative SSD score (inverted — lower SSD = better):
      Green  → top third    (lowest SSD = best matches)
      Yellow → middle third
      Red    → bottom third (highest SSD = worst matches)
    """
    def to_bgr(img: np.ndarray) -> np.ndarray:
        arr = np.clip(img, 0, 255).astype(np.uint8) if img.dtype != np.uint8 else img.copy()
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR) if arr.ndim == 2 else arr

    bgr1 = to_bgr(img1)
    bgr2 = to_bgr(img2)
    h1, w1 = bgr1.shape[:2]
    h2, w2 = bgr2.shape[:2]

    canvas = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    canvas[:h1, :w1]      = bgr1
    canvas[:h2, w1:w1+w2] = bgr2

    # Compute SSD scores for colour mapping
    shown    = match_pairs[:max_lines]
    ssd_vals = []
    for idx_a, idx_b in shown:
        f = descriptors1[idx_a]   # already mean-normalized
        g = descriptors2[idx_b]   # already mean-normalized
        ssd_vals.append(float(np.sum((f - g) ** 2)))

    if ssd_vals:
        lo  = min(ssd_vals)
        rng = max(ssd_vals) - lo
        rng = rng if rng > 1e-8 else 1.0
    else:
        lo, rng = 0.0, 1.0

    for (idx_a, idx_b), ssd in zip(shown, ssd_vals):
        kp1 = valid_kps1[idx_a]
        kp2 = valid_kps2[idx_b]

        # OpenCV: (x=col, y=row)
        pt1 = (int(round(kp1['y'])),       int(round(kp1['x'])))
        pt2 = (int(round(kp2['y'])) + w1,  int(round(kp2['x'])))

        # Invert t: lower SSD → higher t → greener colour
        t = 1.0 - (ssd - lo) / rng
        if t > 0.66:
            color = (50, 220, 50)    # green  — best matches
        elif t > 0.33:
            color = (50, 220, 220)   # yellow — medium
        else:
            color = (50, 50, 220)    # red    — worst matches

        cv2.line(canvas,   pt1, pt2, color, 1, cv2.LINE_AA)
        cv2.circle(canvas, pt1, 4, (80,  80, 220), -1)
        cv2.circle(canvas, pt2, 4, (220, 80,  80), -1)

    return canvas


# =========================================================================== #
#  4. TOP-LEVEL PIPELINE  — called from app.py                                #
# =========================================================================== #

def detect_and_match_ssd(
    img1: np.ndarray,
    img2: np.ndarray,
    patch_size:    int   = 21,
    num_keypoints: int   = 200,
    ratio_thresh:  float = 0.75,
) -> Dict[str, Any]:
    """
    Full SSD pipeline:
      1. SIFT detection          → sift_detector.detect_sift_features_fast()
      2. Patch extraction        → extract_patch_descriptors()  (pure NumPy)
      3. SSD matching            → match_ssd()                  (pure NumPy, vectorized)

    Parameters
    ----------
    img1 / img2    : float32 grayscale (H × W), pixel values 0–255
    patch_size     : odd integer — size of descriptor patch
    num_keypoints  : top-N SIFT keypoints to use per image
    ratio_thresh   : Lowe's ratio threshold for SSD (lower = stricter)

    Returns
    -------
    dict with matches, scores, keypoints, descriptors, timing
    """
    from sift_detector import detect_sift_features_fast

    t_start = time.perf_counter()

    if patch_size % 2 == 0:
        patch_size += 1

    # ── SIFT detection ─────────────────────────────────────────────────── #
    
    # Detect all SIFT keypoints in both images

    kps1_all, _ = detect_sift_features_fast(img1)
    
    kps2_all, _ = detect_sift_features_fast(img2)
    
   # Keep only the top-N keypoints ranked by contrast 
   # (most distinctive first)
   # High contrast = more unique = better for matching
    # Top-N by SIFT contrast (most distinctive first)
    kps1 = sorted(kps1_all, key=lambda k: k['contrast'], reverse=True)[:num_keypoints]
    kps2 = sorted(kps2_all, key=lambda k: k['contrast'], reverse=True)[:num_keypoints]

    # ── Patch descriptors ──────────────────────────────────────────────── #
    print(f"[SSD Pipeline] Extracting patches (patch_size={patch_size}) …")
    desc1, valid_kps1 = extract_patch_descriptors(img1, kps1, patch_size)
    desc2, valid_kps2 = extract_patch_descriptors(img2, kps2, patch_size)
    print(f"               → desc1={len(desc1)}, desc2={len(desc2)}")

    if not desc1 or not desc2:
        return {
            'matches':            [],
            'num_matches':        0,
            'computational_time': time.perf_counter() - t_start,
            'keypoints1':         valid_kps1,
            'keypoints2':         valid_kps2,
            'descriptors1':       desc1,
            'descriptors2':       desc2,
        }

    # ── SSD Matching ───────────────────────────────────────────────────── #
    print(f"[SSD Pipeline] Matching ratio_thresh={ratio_thresh} …")
    raw_matches = match_ssd(desc1, desc2, ratio_thresh)

    # Annotate each match with its true SSD score
    matches_out = []
    for idx_a, idx_b in raw_matches:
        kp1       = valid_kps1[idx_a]
        kp2       = valid_kps2[idx_b]
        f         = desc1[idx_a]   # already mean-normalized
        g         = desc2[idx_b]   # already mean-normalized
        ssd_score = float(np.sum((f - g) ** 2))

        matches_out.append({
            'idx1':      idx_a,
            'idx2':      idx_b,
            'point1':    [int(round(kp1['x'])), int(round(kp1['y']))],
            'point2':    [int(round(kp2['x'])), int(round(kp2['y']))],
            'ncc_score': ssd_score,   # key kept as ncc_score so frontend works unchanged
        })

    total = time.perf_counter() - t_start
    print(f"[SSD Pipeline] Done — {len(matches_out)} matches in {total:.4f}s")

    return {
        'matches':            matches_out,
        'num_matches':        len(matches_out),
        'computational_time': total,
        'keypoints1':         valid_kps1,
        'keypoints2':         valid_kps2,
        'descriptors1':       desc1,
        'descriptors2':       desc2,
    }