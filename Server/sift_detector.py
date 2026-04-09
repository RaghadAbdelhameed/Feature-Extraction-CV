import numpy as np
import cv2
from scipy import ndimage
import warnings
from scipy.ndimage import maximum_filter, minimum_filter

# Import general scale-space and image tools from utils
from utils import (
    normalize_image, 
    build_gaussian_pyramid, 
    compute_dog_pyramid, 
    remove_duplicates_fast
)

warnings.filterwarnings('ignore')

# --- SIFT SPECIFIC FUNCTIONS ---

def find_keypoints_vectorized(dog_pyramid, contrast_threshold=0.015, edge_threshold=15, sigma=1.6):
    """Vectorized SIFT keypoint detection using morphological filters"""

    keypoints = []
    
    for octave_idx, dog_octave in enumerate(dog_pyramid):
        for scale_idx in range(1, len(dog_octave)-1):
            current = dog_octave[scale_idx]
            prev = dog_octave[scale_idx-1]
            nxt = dog_octave[scale_idx+1]
            
            if current.shape[0] < 3 or current.shape[1] < 3:
                continue
            
            # Find local maxima/minima in 2D
            local_max = (current == maximum_filter(current, size=3)) & (current > 0)
            local_min = (current == minimum_filter(current, size=3)) & (current < 0)
            
            for coords in np.concatenate([np.argwhere(local_max), np.argwhere(local_min)]):
                i, j = coords
                if 0 < i < current.shape[0]-1 and 0 < j < current.shape[1]-1:
                    # Check against adjacent scales (3D extrema check)
                    val = current[i, j]
                    if (val > prev[i, j] and val > nxt[i, j]) or (val < prev[i, j] and val < nxt[i, j]):
                        kp = refine_keypoint_fast(dog_octave, octave_idx, scale_idx, i, j, 
                                                 contrast_threshold, edge_threshold, sigma)
                        if kp:
                            keypoints.append(kp)
    return keypoints

def refine_keypoint_fast(dog_octave, octave_idx, scale_idx, x, y, contrast_threshold, edge_threshold, sigma):
    """Reject low contrast and edge responses using Hessian matrix (Lowe's method)"""
    if x < 2 or x >= dog_octave[scale_idx].shape[0]-2 or y < 2 or y >= dog_octave[scale_idx].shape[1]-2:
        return None
    
    current = dog_octave[scale_idx]
    if abs(current[x, y]) < contrast_threshold:
        return None
    
    dxx = current[x, y+1] - 2*current[x, y] + current[x, y-1]
    dyy = current[x+1, y] - 2*current[x, y] + current[x-1, y]
    dxy = (current[x+1, y+1] - current[x+1, y-1] - current[x-1, y+1] + current[x-1, y-1]) / 4.0
    
    trace = dxx + dyy
    det = dxx * dyy - dxy * dxy
    
    if det <= 0: return None
    
    edge_ratio = (trace**2) / det
    if edge_ratio > (edge_threshold + 1)**2 / edge_threshold:
        return None
    
    scale = sigma * (2 ** octave_idx) * (2 ** (scale_idx / 3.0))
    return {
        'x': x * (2 ** octave_idx), 'y': y * (2 ** octave_idx),
        'scale': scale, 'octave': octave_idx, 'scale_idx': scale_idx,
        'contrast': abs(current[x, y])
    }

def compute_orientations_vectorized(keypoints, gaussian_pyramid):
    """Assign dominant orientation to SIFT keypoints"""
    orientations = []
    for kp in keypoints:
        oct_idx, scale_idx = kp['octave'], min(kp['scale_idx'], len(gaussian_pyramid[kp['octave']])-1)
        image = gaussian_pyramid[oct_idx][scale_idx]
        x, y = int(kp['x'] / (2**oct_idx)), int(kp['y'] / (2**oct_idx))
        
        if 5 > x or x >= image.shape[0]-5 or 5 > y or y >= image.shape[1]-5:
            orientations.append(0); continue

        region = image[x-5:x+6, y-5:y+6]
        gy, gx = np.gradient(region)
        mag = np.sqrt(gx**2 + gy**2)
        ang = np.arctan2(gy, gx) * 180 / np.pi
        ang[ang < 0] += 360
        
        hist, _ = np.histogram(ang, bins=36, range=(0, 360), weights=mag)
        hist = ndimage.gaussian_filter1d(hist, sigma=1)
        orientations.append(np.argmax(hist) * 10)
    return orientations

def compute_descriptors_vectorized(keypoints, orientations, gaussian_pyramid):
    """Build 128-D SIFT descriptors"""
    descriptors, valid_indices = [], []
    for idx, kp in enumerate(keypoints):
        oct_idx, scale_idx = kp['octave'], min(kp['scale_idx'], len(gaussian_pyramid[kp['octave']])-1)
        image = gaussian_pyramid[oct_idx][scale_idx]
        x, y = int(kp['x'] / (2**oct_idx)), int(kp['y'] / (2**oct_idx))
        
        if 8 > x or x >= image.shape[0]-8 or 8 > y or y >= image.shape[1]-8:
            continue

        window = image[x-8:x+8, y-8:y+8]
        gy, gx = np.gradient(window)
        mag, ang = np.sqrt(gx**2 + gy**2), (np.arctan2(gy, gx) * 180 / np.pi - orientations[idx]) % 360
        
        desc = []
        for i in range(0, 16, 4):
            for j in range(0, 16, 4):
                cell_mag, cell_ang = mag[i:i+4, j:j+4], ang[i:i+4, j:j+4]
                hist, _ = np.histogram(cell_ang, bins=8, range=(0, 360), weights=cell_mag)
                desc.extend(hist)
        
        desc = np.array(desc)
        norm = np.linalg.norm(desc)
        if norm > 0:
            desc = np.minimum(desc / norm, 0.2)
            desc /= (np.linalg.norm(desc) + 1e-7)
            descriptors.append(desc)
            valid_indices.append(idx)
    return descriptors, valid_indices

def detect_sift_features_fast(image, contrast_threshold=0.012, edge_threshold=18, sigma=1.6):
    """Main pipeline for SIFT feature extraction with adjustable parameters"""
    # Convert to grayscale if it's a color image
    if len(image.shape) == 3: 
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    else:
        gray_image = image.astype(np.float32)
        
    gray_image = normalize_image(gray_image)
    
    gaussian_pyramid = build_gaussian_pyramid(gray_image, sigma=sigma)
    dog_pyramid = compute_dog_pyramid(gaussian_pyramid)
    
    keypoints = find_keypoints_vectorized(dog_pyramid, contrast_threshold, edge_threshold, sigma=sigma)
    keypoints = remove_duplicates_fast(keypoints)
    
    if not keypoints: return [], []
    
    orientations = compute_orientations_vectorized(keypoints, gaussian_pyramid)
    descriptors, valid_indices = compute_descriptors_vectorized(keypoints, orientations, gaussian_pyramid)
    return [keypoints[i] for i in valid_indices], descriptors