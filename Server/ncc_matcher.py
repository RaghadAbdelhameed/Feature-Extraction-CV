import numpy as np
import time
from typing import List, Tuple
from utils import base64_to_image_array, image_array_to_base64, detect_harris_corners

def ncc_matching_robust(template: np.ndarray, search_image: np.ndarray) -> dict:
    """
    Robust NCC matching using vectorized operations for template matching.
    """
    start_time = time.time()
    
    t_h, t_w = template.shape
    s_h, s_w = search_image.shape
    
    # Check dimensions
    if t_h > s_h or t_w > s_w:
        return {'error': f'Template ({t_h}x{t_w}) larger than search image ({s_h}x{s_w})'}
    
    # Calculate output dimensions
    out_h = s_h - t_h + 1
    out_w = s_w - t_w + 1
    
    if out_h <= 0 or out_w <= 0:
        return {'error': f'Invalid output dimensions ({out_h}x{out_w})'}
    
    # Extract all patches from search image
    patches = np.zeros((out_h * out_w, t_h * t_w), dtype=np.float32)
    
    idx = 0
    for i in range(out_h):
        for j in range(out_w):
            patch = search_image[i:i+t_h, j:j+t_w]
            patches[idx] = patch.flatten()
            idx += 1
    
    # Normalize template
    template_mean = np.mean(template)
    template_std = np.std(template)
    template_norm = (template - template_mean) / (template_std + 1e-8)
    template_flat = template_norm.flatten()
    
    # Normalize all patches (vectorized)
    patch_means = np.mean(patches, axis=1, keepdims=True)
    patch_stds = np.std(patches, axis=1, keepdims=True)
    patches_norm = (patches - patch_means) / (patch_stds + 1e-8)
    
    # Compute NCC scores for all patches (vectorized)
    ncc_scores = np.sum(patches_norm * template_flat, axis=1) / (t_h * t_w)
    
    # Reshape NCC scores back to 2D map
    ncc_map = ncc_scores.reshape(out_h, out_w)
    
    # Find best match
    best_idx = np.argmax(ncc_scores)
    best_row = best_idx // out_w
    best_col = best_idx % out_w
    
    computational_time = time.time() - start_time
    
    return {
        'ncc_map': ncc_map,
        'best_location': (best_row, best_col),
        'best_ncc': float(ncc_map[best_row, best_col]),
        'computational_time': computational_time,
        'template_shape': [t_h, t_w],
        'search_shape': [s_h, s_w]
    }

def extract_keypoint_descriptors(image: np.ndarray, keypoints: List[Tuple[int, int]], 
                                 patch_size: int = 15) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
    """
    Extract normalized patch descriptors around keypoints.
    Returns list of flattened, normalized patches.
    """
    descriptors = []
    valid_keypoints = []
    half_patch = patch_size // 2
    h, w = image.shape
    
    for r, c in keypoints:
        r_int, c_int = int(r), int(c)
        # Check boundaries
        if (r_int - half_patch >= 0 and r_int + half_patch < h and
            c_int - half_patch >= 0 and c_int + half_patch < w):
            
            # Extract patch
            patch = image[r_int-half_patch:r_int+half_patch+1, 
                        c_int-half_patch:c_int+half_patch+1]
            
            # Normalize patch
            patch_mean = np.mean(patch)
            patch_std = np.std(patch)
            patch_norm = (patch - patch_mean) / (patch_std + 1e-8)
            
            descriptors.append(patch_norm.flatten())
            valid_keypoints.append((r_int, c_int))
    
    return descriptors, valid_keypoints

def match_ncc(descriptors_A, descriptors_B, ratio_thresh=0.9):
    """
    Match descriptors using NCC with ratio test.
    EXACT LOGIC as provided by user.
    """
    matches = []
    
    start = time.time()
    
    for i, f in enumerate(descriptors_A):
        # Normalize f
        f_mean = np.mean(f)
        f_std = np.std(f)
        f_norm = (f - f_mean) / (f_std + 1e-8)
        
        ncc_scores = []
        
        for g in descriptors_B:
            g_mean = np.mean(g)
            g_std = np.std(g)
            g_norm = (g - g_mean) / (g_std + 1e-8)
            
            ncc = np.sum(f_norm * g_norm)
            ncc_scores.append(ncc)
        
        ncc_scores = np.array(ncc_scores)
        
        # Best and second best
        idx = np.argsort(-ncc_scores)  # descending
        best_idx = idx[0]
        second_idx = idx[1]
        
        best = ncc_scores[best_idx]
        second = ncc_scores[second_idx]
        
        # Ratio-like test (since higher is better)
        if best > ratio_thresh and (best / (second + 1e-8)) > 1.1:
            matches.append((i, best_idx))
    
    end = time.time()
    print(f"NCC Matching Time: {end - start:.4f} seconds")
    print(f"Found {len(matches)} matches")
    
    return matches

def detect_and_match_features(image1: np.ndarray, image2: np.ndarray, 
                              patch_size: int = 15, 
                              num_keypoints: int = 50,
                              ratio_thresh: float = 0.9) -> dict:
    """
    Detect keypoints using Harris corner detection and match using NCC
    with the exact matching logic provided.
    """
    overall_start = time.time()
    
    print(f"Image 1 shape: {image1.shape}")
    print(f"Image 2 shape: {image2.shape}")
    
    # Ensure patch size is odd
    if patch_size % 2 == 0:
        patch_size += 1
    print(f"Using patch size: {patch_size}x{patch_size}")
    
    # Detect keypoints using Harris corner detector
    print("\nDetecting keypoints in Image 1...")
    start = time.time()
    keypoints1 = detect_harris_corners(image1, num_keypoints)
    print(f"Detection time: {time.time() - start:.4f} seconds")
    print(f"Found {len(keypoints1)} keypoints")
    
    print("\nDetecting keypoints in Image 2...")
    start = time.time()
    keypoints2 = detect_harris_corners(image2, num_keypoints)
    print(f"Detection time: {time.time() - start:.4f} seconds")
    print(f"Found {len(keypoints2)} keypoints")
    
    # Extract descriptors (normalized patches)
    print("\nExtracting descriptors from Image 1...")
    start = time.time()
    descriptors1, valid_kp1 = extract_keypoint_descriptors(image1, keypoints1, patch_size)
    print(f"Extraction time: {time.time() - start:.4f} seconds")
    print(f"Valid keypoints: {len(valid_kp1)}")
    
    print("\nExtracting descriptors from Image 2...")
    start = time.time()
    descriptors2, valid_kp2 = extract_keypoint_descriptors(image2, keypoints2, patch_size)
    print(f"Extraction time: {time.time() - start:.4f} seconds")
    print(f"Valid keypoints: {len(valid_kp2)}")
    
    # Match descriptors using NCC with ratio test
    print("\nMatching descriptors using NCC...")
    match_indices = match_ncc(descriptors1, descriptors2, ratio_thresh)
    
    # Convert match indices to actual keypoint coordinates and calculate NCC scores
    matches = []
    ncc_scores = []
    
    for i, j in match_indices:
        kp1 = valid_kp1[i]
        kp2 = valid_kp2[j]
        
        # Calculate NCC score for this match
        f = descriptors1[i]
        g = descriptors2[j]
        f_mean = np.mean(f)
        f_std = np.std(f)
        f_norm = (f - f_mean) / (f_std + 1e-8)
        g_mean = np.mean(g)
        g_std = np.std(g)
        g_norm = (g - g_mean) / (g_std + 1e-8)
        ncc = np.sum(f_norm * g_norm)
        
        matches.append({
            'point1': [int(kp1[0]), int(kp1[1])],
            'point2': [int(kp2[0]), int(kp2[1])],
            'ncc_score': float(ncc)
        })
        ncc_scores.append(float(ncc))
    
    total_time = time.time() - overall_start
    
    print(f"\nTotal computational time: {total_time:.4f} seconds")
    print(f"Number of matches found: {len(matches)}")
    
    if len(ncc_scores) > 0:
        print(f"NCC scores - Min: {min(ncc_scores):.4f}, "
              f"Max: {max(ncc_scores):.4f}, "
              f"Mean: {np.mean(ncc_scores):.4f}")
    
    return {
        'matches': matches,
        'num_matches': len(matches),
        'computational_time': total_time,
        'keypoints1': [[int(k[0]), int(k[1])] for k in valid_kp1],
        'keypoints2': [[int(k[0]), int(k[1])] for k in valid_kp2],
        'ncc_scores': ncc_scores
    }

def visualize_match_result(img1: np.ndarray, img2: np.ndarray, matches: list, keypoints1: list, keypoints2: list) -> np.ndarray:
    """
    Create a visualization of feature matches between two images.
    Returns combined image with matches drawn.
    """
    import cv2
    
    # Convert to uint8 for visualization
    if img1.dtype != np.uint8:
        if img1.max() <= 1.0:
            img1_viz = (img1 * 255).astype(np.uint8)
        else:
            img1_viz = img1.astype(np.uint8)
    else:
        img1_viz = img1
    
    if img2.dtype != np.uint8:
        if img2.max() <= 1.0:
            img2_viz = (img2 * 255).astype(np.uint8)
        else:
            img2_viz = img2.astype(np.uint8)
    else:
        img2_viz = img2
    
    # Convert grayscale to BGR for colored lines
    img1_color = cv2.cvtColor(img1_viz, cv2.COLOR_GRAY2BGR)
    img2_color = cv2.cvtColor(img2_viz, cv2.COLOR_GRAY2BGR)
    
    # Combine images side by side
    h1, w1 = img1_color.shape[:2]
    h2, w2 = img2_color.shape[:2]
    
    combined_height = max(h1, h2)
    combined_width = w1 + w2
    
    combined = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)
    combined[:h1, :w1] = img1_color
    combined[:h2, w1:w1+w2] = img2_color
    
    # Draw matches
    for match in matches[:50]:  # Limit to 50 matches for visibility
        pt1 = (match['point1'][1], match['point1'][0])  # (x, y)
        pt2 = (match['point2'][1] + w1, match['point2'][0])  # (x + offset, y)
        
        # Draw line with color based on NCC score
        ncc_score = match['ncc_score']
        # Green for high scores, yellow for medium, red for low
        if ncc_score > 0.8:
            color = (0, 255, 0)  # Green
        elif ncc_score > 0.6:
            color = (0, 255, 255)  # Yellow
        else:
            color = (0, 0, 255)  # Red
        
        cv2.line(combined, pt1, pt2, color, 1)
        # Draw circles at keypoints
        cv2.circle(combined, pt1, 3, (0, 0, 255), -1)
        cv2.circle(combined, pt2, 3, (255, 0, 0), -1)
    
    return combined