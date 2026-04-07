import numpy as np
import cv2
import time
from scipy.signal import convolve2d
from scipy.ndimage import maximum_filter

# Import the common functions from your utils file
from utils import load_image, preprocess_for_gradients, compute_spatial_gradients, encode_image_to_base64

def run_harris_detector(image_path, block_size=3, k=0.04, threshold_ratio=0.1):
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
    window = np.ones((block_size, block_size))
    Sxx = convolve2d(Ixx, window, mode='same', boundary='symm')
    Syy = convolve2d(Iyy, window, mode='same', boundary='symm')
    Sxy = convolve2d(Ixy, window, mode='same', boundary='symm')

    # 4. Harris Corner Response
    det_M = (Sxx * Syy) - (Sxy**2)
    trace_M = Sxx + Syy
    R = det_M - k * (trace_M**2)

    # 5. Thresholding & NMS
    R_max = R.max()
    threshold_value = threshold_ratio * R_max
    corner_map = R > threshold_value
    local_max = maximum_filter(R, size=40) == R
    corner_map = corner_map & local_max

    y_coords, x_coords = np.where(corner_map)
    computation_time = time.time() - start_time

    # 6. Draw the points directly onto a copy of the original image
    output_img = original_img.copy()
    for x, y in zip(x_coords, y_coords):
        # Drawing purely Red circles (BGR format)
        cv2.circle(output_img, (int(x), int(y)), radius=5, color=(0, 0, 255), thickness=-1)

    # 7. Convert the marked-up image to Base64 (Using Utils)
    # The util function already formats it with "data:image/jpeg;base64,..."
    img_base64_string = encode_image_to_base64(output_img, ext='.jpg')

    # 8. Return exactly what the webapp needs
    return {
        "computation_time_seconds": round(computation_time, 4),
        "total_points": len(x_coords),
        "result_image_base64": img_base64_string
    }