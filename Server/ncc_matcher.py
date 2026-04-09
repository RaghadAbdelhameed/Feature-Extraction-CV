"""
NCC Feature Matcher — from scratch.

Detection  : SIFT keypoints via sift_detector.detect_sift_features_fast()
             (OpenCV used inside sift_detector for grayscale conversion only)
Descriptors: Raw pixel patches extracted around each SIFT keypoint — pure NumPy
Matching   : NCC with ratio test (colleague's exact algorithm) — pure NumPy
             SSD with ratio test (placeholder for future UI option) — pure NumPy
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
    """
    Extract a raw pixel patch around every SIFT keypoint as its descriptor.

    SIFT keypoint dict convention:  'x' = row,  'y' = col

    Parameters
    ----------
    image      : 2-D float32 grayscale  (H × W),  pixel values 0–255
    keypoints  : list of dicts with 'x' (row) and 'y' (col)
    patch_size : odd integer square patch size

    Returns
    -------
    descriptors : list of flat float64 arrays  (patch_size² each)
    valid_kps   : matching subset of the input keypoints (border-safe only)
    """
    if patch_size % 2 == 0:
        patch_size += 1
    half = patch_size // 2
    h, w = image.shape[:2]

    descriptors: List[np.ndarray] = []
    valid_kps:   List[Dict]       = []

    for kp in keypoints:
        row = int(round(kp['x']))   # SIFT: x = row
        col = int(round(kp['y']))   # SIFT: y = col

        # Discard keypoints whose patch would fall outside the image
        if (row - half < 0 or row + half >= h or
                col - half < 0 or col + half >= w):
            continue

        patch = image[row - half: row + half + 1,
                      col - half: col + half + 1].astype(np.float64)
        descriptors.append(patch.ravel())
        valid_kps.append(kp)

    return descriptors, valid_kps


# =========================================================================== #
#  2. NCC MATCHING  — colleague's exact algorithm, pure NumPy                 #
# =========================================================================== #

def match_ncc(
    descriptors_A: List[np.ndarray],
    descriptors_B: List[np.ndarray],
    ratio_thresh: float = 0.9
) -> List[Tuple[int, int]]:
    """
    Normalised Cross-Correlation matching with ratio test.

    For each descriptor f in A:
      1. Normalise f  → f_norm = (f − mean) / (std + ε)
      2. Do the same for every g in B
      3. NCC(f, g) = sum(f_norm * g_norm)
      4. Keep if  best_ncc > ratio_thresh
              AND best_ncc / second_best_ncc > 1.1

    Returns list of (idx_in_A, idx_in_B) pairs.
    """
    matches: List[Tuple[int, int]] = []
    start = time.time()

    for i, f in enumerate(descriptors_A):
        f_mean = np.mean(f)
        f_std  = np.std(f)
        f_norm = (f - f_mean) / (f_std + 1e-8)

        ncc_scores = []
        for g in descriptors_B:
            g_mean = np.mean(g)
            g_std  = np.std(g)
            g_norm = (g - g_mean) / (g_std + 1e-8)
            ncc    = np.sum(f_norm * g_norm)
            ncc_scores.append(ncc)

        ncc_scores = np.array(ncc_scores)

        if len(ncc_scores) < 2:
            continue

        idx        = np.argsort(-ncc_scores)   # descending
        best_idx   = idx[0]
        second_idx = idx[1]
        best       = ncc_scores[best_idx]
        second     = ncc_scores[second_idx]

        if best > ratio_thresh and (best / (second + 1e-8)) > 1.1:
            matches.append((i, int(best_idx)))

    print(f"[NCC] match time: {time.time()-start:.3f}s  found: {len(matches)}")
    return matches


# =========================================================================== #
#  3. SSD MATCHING  — pure NumPy  (ready for future UI toggle)                #
# =========================================================================== #

def match_ssd(
    descriptors_A: List[np.ndarray],
    descriptors_B: List[np.ndarray],
    ratio_thresh: float = 0.75
) -> List[Tuple[int, int]]:
    """
    Sum of Squared Differences matching with ratio test.
    Lower SSD = better match.

    Keep if  best_ssd / second_best_ssd  < ratio_thresh
    """
    matches: List[Tuple[int, int]] = []
    start = time.time()

    for i, f in enumerate(descriptors_A):
        ssd_scores = np.array([np.sum((f - g) ** 2) for g in descriptors_B])

        if len(ssd_scores) < 2:
            continue

        idx        = np.argsort(ssd_scores)   # ascending (lower = better)
        best_idx   = idx[0]
        second_idx = idx[1]
        best       = ssd_scores[best_idx]
        second     = ssd_scores[second_idx]

        if second > 1e-8 and (best / second) < ratio_thresh:
            matches.append((i, int(best_idx)))

    print(f"[SSD] match time: {time.time()-start:.3f}s  found: {len(matches)}")
    return matches


# =========================================================================== #
#  4. VISUALISATION  — OpenCV used only for drawing                           #
# =========================================================================== #

def visualize_matches(
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

    Colour by relative NCC score:
      Green  → top third
      Yellow → middle third
      Red    → bottom third
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

    # Compute NCC scores for colour mapping
    shown = match_pairs[:max_lines]
    ncc_vals = []
    for idx_a, idx_b in shown:
        f  = descriptors1[idx_a]
        g  = descriptors2[idx_b]
        fn = (f - f.mean()) / (f.std() + 1e-8)
        gn = (g - g.mean()) / (g.std() + 1e-8)
        ncc_vals.append(float(np.sum(fn * gn)))

    if ncc_vals:
        lo  = min(ncc_vals)
        rng = max(ncc_vals) - lo
        rng = rng if rng > 1e-8 else 1.0
    else:
        lo, rng = 0.0, 1.0

    for (idx_a, idx_b), score in zip(shown, ncc_vals):
        kp1 = valid_kps1[idx_a]
        kp2 = valid_kps2[idx_b]

        # OpenCV: (x=col, y=row)
        pt1 = (int(round(kp1['y'])),       int(round(kp1['x'])))
        pt2 = (int(round(kp2['y'])) + w1,  int(round(kp2['x'])))

        t = (score - lo) / rng
        if t > 0.66:
            color = (50, 220, 50)    # green
        elif t > 0.33:
            color = (50, 220, 220)   # yellow
        else:
            color = (50, 50, 220)    # red

        cv2.line(canvas,   pt1, pt2, color, 1, cv2.LINE_AA)
        cv2.circle(canvas, pt1, 4, (80,  80, 220), -1)
        cv2.circle(canvas, pt2, 4, (220, 80,  80), -1)

    return canvas


# =========================================================================== #
#  5. TEMPLATE MATCHING  — pure NumPy sliding-window NCC                      #
# =========================================================================== #

def ncc_template_match(
    template: np.ndarray,
    search:   np.ndarray
) -> Dict[str, Any]:
    """Sliding-window NCC template matching, pure NumPy (stride tricks)."""
    start = time.perf_counter()
    t_h, t_w = template.shape[:2]
    s_h, s_w = search.shape[:2]

    if t_h > s_h or t_w > s_w:
        return {'error': f'Template ({t_h}×{t_w}) larger than search ({s_h}×{s_w})'}

    out_h, out_w = s_h - t_h + 1, s_w - t_w + 1
    n_pix        = t_h * t_w

    tmpl_std = template.std()
    if tmpl_std < 1e-8:
        return {'error': 'Template is uniform (zero variance)'}
    tmpl_norm = ((template - template.mean()) / tmpl_std).ravel()

    from numpy.lib.stride_tricks import as_strided
    s    = search.strides
    view = as_strided(search,
                      shape=(out_h, out_w, t_h, t_w),
                      strides=(s[0], s[1], s[0], s[1]))
    patches = view.reshape(out_h * out_w, n_pix).astype(np.float64)
    pmeans  = patches.mean(axis=1, keepdims=True)
    pstds   = patches.std(axis=1,  keepdims=True)
    pstds   = np.where(pstds < 1e-8, 1.0, pstds)
    pnorm   = (patches - pmeans) / pstds
    scores  = pnorm.dot(tmpl_norm) / n_pix
    ncc_map = scores.reshape(out_h, out_w)
    best    = int(np.argmax(scores))

    return {
        'ncc_map':            ncc_map,
        'best_location':      (best // out_w, best % out_w),
        'best_ncc':           float(scores[best]),
        'template_shape':     (t_h, t_w),
        'computational_time': time.perf_counter() - start,
    }


# =========================================================================== #
#  6. TOP-LEVEL PIPELINE  — called from app.py                                #
# =========================================================================== #

def detect_and_match_features(
    img1: np.ndarray,
    img2: np.ndarray,
    patch_size:    int   = 21,
    num_keypoints: int   = 200,
    ratio_thresh:  float = 0.85,
    method:        str   = 'ncc',
) -> Dict[str, Any]:
    """
    Full pipeline:
      1. SIFT detection          → sift_detector.detect_sift_features_fast()
      2. Patch extraction        → extract_patch_descriptors()  (pure NumPy)
      3. NCC or SSD matching     → match_ncc() / match_ssd()    (pure NumPy)

    img1 / img2   : float32 grayscale (H × W), pixel values 0–255
    method        : 'ncc' or 'ssd'
    """
    from sift_detector import detect_sift_features_fast

    t_start = time.perf_counter()

    if patch_size % 2 == 0:
        patch_size += 1

    # ── SIFT detection ─────────────────────────────────────────────────── #
    print(f"[Pipeline] SIFT on img1 {img1.shape} …")
    kps1_all, _ = detect_sift_features_fast(img1)
    print(f"           → {len(kps1_all)} keypoints")

    print(f"[Pipeline] SIFT on img2 {img2.shape} …")
    kps2_all, _ = detect_sift_features_fast(img2)
    print(f"           → {len(kps2_all)} keypoints")

    # Top-N by SIFT contrast (highest contrast = most distinctive)
    kps1 = sorted(kps1_all, key=lambda k: k['contrast'], reverse=True)[:num_keypoints]
    kps2 = sorted(kps2_all, key=lambda k: k['contrast'], reverse=True)[:num_keypoints]

    # ── Patch descriptors ──────────────────────────────────────────────── #
    print(f"[Pipeline] Patches patch_size={patch_size} …")
    desc1, valid_kps1 = extract_patch_descriptors(img1, kps1, patch_size)
    desc2, valid_kps2 = extract_patch_descriptors(img2, kps2, patch_size)
    print(f"           → desc1={len(desc1)}, desc2={len(desc2)}")

    if not desc1 or not desc2:
        return {
            'matches': [], 'num_matches': 0,
            'computational_time': time.perf_counter() - t_start,
            'keypoints1': valid_kps1, 'keypoints2': valid_kps2,
            'descriptors1': desc1,    'descriptors2': desc2,
        }

    # ── Matching ───────────────────────────────────────────────────────── #
    print(f"[Pipeline] {method.upper()} matching ratio_thresh={ratio_thresh} …")
    if method == 'ssd':
        raw_matches = match_ssd(desc1, desc2, ratio_thresh)
    else:
        raw_matches = match_ncc(desc1, desc2, ratio_thresh)

    # Annotate each match with its NCC score for the frontend
    matches_out = []
    for idx_a, idx_b in raw_matches:
        kp1  = valid_kps1[idx_a]
        kp2  = valid_kps2[idx_b]
        f    = desc1[idx_a];  fn = (f - f.mean()) / (f.std() + 1e-8)
        g    = desc2[idx_b];  gn = (g - g.mean()) / (g.std() + 1e-8)
        matches_out.append({
            'idx1':      idx_a,
            'idx2':      idx_b,
            'point1':    [int(round(kp1['x'])), int(round(kp1['y']))],
            'point2':    [int(round(kp2['x'])), int(round(kp2['y']))],
            'ncc_score': float(np.sum(fn * gn)),
        })

    total = time.perf_counter() - t_start
    print(f"[Pipeline] Done — {len(matches_out)} matches in {total:.4f}s")

    return {
        'matches':            matches_out,
        'num_matches':        len(matches_out),
        'computational_time': total,
        'keypoints1':         valid_kps1,
        'keypoints2':         valid_kps2,
        'descriptors1':       desc1,
        'descriptors2':       desc2,
    }
