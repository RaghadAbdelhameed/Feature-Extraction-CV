import numpy as np
import cv2
import time

# Import the common functions from your utils file 
from utils import load_image, preprocess_for_gradients, compute_spatial_gradients, encode_image_to_base64, custom_convolve2d

def run_harris_detector(image_path, method='harris', block_size=3, k=0.04, threshold_ratio=0.01):
    """
    Self-contained Harris Corner Detector.
    Returns computation time, point count, and the output image as a Base64 string.
    """
    # 1. Read and preprocess (Using Utils)
    original_img = load_image(image_path)
    gray_img_float = preprocess_for_gradients(original_img, apply_blur=True)
    
    start_time = time.time()

    # 2. Calculate Gradients (Using Utils)
    Ix, Iy = compute_spatial_gradients(gray_img_float)

    # 3. Structure Tensor Components 
    Ixx, Iyy, Ixy = Ix**2, Iy**2, Ix * Iy
    
    # --- Gaussian Window Generation ---
    # Calculate standard deviation (sigma) based on the block_size
    sigma = 0.3 * ((block_size - 1) * 0.5 - 1) + 0.8
    
    # Create a 1D grid originating from the center
    ax = np.arange(-(block_size // 2), block_size // 2 + 1)
    
    # Calculate the 1D Gaussian curve
    kernel_1d = np.exp(-(ax ** 2) / (2 * sigma ** 2))
    
    # Create the 2D Gaussian filter matrix by taking the outer product
    window = np.outer(kernel_1d, kernel_1d)
    
    # Normalize the kernel so the sum of all elements equals 1
    window /= np.sum(window)
    # ------------------------------------------------------
    
    Sxx = custom_convolve2d(Ixx, window, mode='same', boundary='symm')
    Syy = custom_convolve2d(Iyy, window, mode='same', boundary='symm')
    Sxy = custom_convolve2d(Ixy, window, mode='same', boundary='symm')

    # 4. Corner Response
    if method == 'harris':
        det_M = (Sxx * Syy) - (Sxy**2)
        trace_M = Sxx + Syy
        R = det_M - k * (trace_M**2)
    elif method == 'shi-tomasi':
        # get the minimum eigenvalue of the structure tensor
        R = 0.5 * ((Sxx + Syy) - np.sqrt((Sxx - Syy)**2 + 4 * (Sxy**2)))
    else:
        raise ValueError("Method must be 'harris' or 'shi-tomasi'")

    # 5. Thresholding & strict Distance-Based NMS 
    R_max = R.max()
    threshold_value = threshold_ratio * R_max
    
    # Get all coordinates that pass the initial minimum threshold
    y_coords, x_coords = np.where(R > threshold_value)
    scores = R[y_coords, x_coords]
    
    # Sort points by their score in descending order (highest first)
    sorted_indices = np.argsort(scores)[::-1]
    x_coords = x_coords[sorted_indices]
    y_coords = y_coords[sorted_indices]
    
    final_points = []
    min_distance = 10  # Minimum pixel distance allowed between two corners
    min_dist_sq = min_distance ** 2  # Pre-calculate squared distance for speed
    
    # Greedy NMS: keep the best point, reject anything too close to it
    for i in range(len(x_coords)):
        px, py = x_coords[i], y_coords[i]
        
        # Automatically add the highest scoring point first
        if not final_points:
            final_points.append((px, py))
            continue
            
        # Convert our accepted points to a NumPy array to do fast vectorized math
        accepted_arr = np.array(final_points)
        
        # Calculate the squared distance from the current point to all accepted points
        dx = accepted_arr[:, 0] - px
        dy = accepted_arr[:, 1] - py
        distances_sq = (dx ** 2) + (dy ** 2)
        
        # If the minimum distance to ANY accepted point is valid, keep the point
        if np.all(distances_sq >= min_dist_sq):
            final_points.append((px, py))

    computation_time = time.time() - start_time

    # 6. Draw the filtered points directly onto a copy of the original image
    output_img = original_img.copy()
    for x, y in final_points:
        cv2.circle(output_img, (int(x), int(y)), radius=5, color=(0, 0, 255), thickness=-1)

    # 7. Convert the marked-up image to Base64 (Using Utils)
    # The util function already formats it with "data:image/jpeg;base64,..."
    img_base64_string = encode_image_to_base64(output_img, ext='.jpg')

    # 8. Return exactly what the webapp needs
    return {
        "computation_time_seconds": round(computation_time, 4),
        "total_points": len(final_points),
        "result_image_base64": img_base64_string
    }