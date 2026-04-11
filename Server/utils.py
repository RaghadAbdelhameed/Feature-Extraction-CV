import cv2
import numpy as np
import base64
from PIL import Image
import io
from scipy import ndimage
from scipy.spatial import KDTree
from numpy.lib.stride_tricks import sliding_window_view

def custom_convolve2d(image, kernel, mode='same', boundary='symm'):
    """
    A NumPy-based implementation of 2D convolution to replace scipy.signal.convolve2d.
    """
    # 1. Flip the kernel horizontally and vertically. 
    # Strict mathematical convolution requires a flipped kernel (unlike cross-correlation).
    # This matches the exact behavior of scipy.signal.convolve2d.
    k_flipped = np.flip(kernel)
    
    # 2. Determine the amount of padding needed based on the kernel size.
    # Assuming odd-sized kernels (like 3x3 or 5x5).
    pad_h = k_flipped.shape[0] // 2
    pad_w = k_flipped.shape[1] // 2
    
    # 3. Apply symmetric padding to handle edge cases
    if boundary == 'symm':
        pad_mode = 'symmetric'
    else:
        pad_mode = 'constant'
        
    padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode=pad_mode)
    
    # 4. Extract rolling windows of the exact shape as our kernel across the image
    # This turns a (H, W) image into a (H, W, kernel_H, kernel_W) matrix of views.
    windows = sliding_window_view(padded_image, window_shape=k_flipped.shape)
    
    # 5. Multiply the windows by the kernel and sum them up over the kernel axes
    output = np.sum(windows * k_flipped, axis=(2, 3))
    
    return output

def apply_gaussian_filter(matrix, kernel_size):
    """
    Generates a 2D Gaussian kernel and convolves it with the input matrix.
    Can be used for both initial image smoothing and structure tensor integration.
    """
    # Calculate standard deviation (sigma) based on the kernel_size
    sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
    
    # Create a 1D grid originating from the center
    ax = np.arange(-(kernel_size // 2), kernel_size // 2 + 1)
    
    # Calculate the 1D Gaussian curve
    kernel_1d = np.exp(-(ax ** 2) / (2 * sigma ** 2))
    
    # Create the 2D Gaussian filter matrix by taking the outer product
    kernel_2d = np.outer(kernel_1d, kernel_1d)
    
    # Normalize the kernel so the sum of all elements equals 1
    kernel_2d /= np.sum(kernel_2d)
    
    # Apply the convolution
    return custom_convolve2d(matrix, kernel_2d, mode='same', boundary='symm')

def load_image(image_path):
    """
    Safely reads an image from the given path.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image at {image_path}. Please verify the file exists and is a valid image.")
    return img

def preprocess_for_gradients(image, apply_blur=True, blur_ksize=(5, 5)):
    """
    Converts a BGR image to grayscale from scratch, applies optional custom 
    Gaussian smoothing, and normalizes pixel values to a 0.0 - 1.0 float range.
    """
    
    # --- 1. Grayscale Conversion ---
    b = image[:, :, 0]
    g = image[:, :, 1]
    r = image[:, :, 2]
    
    # Apply the standard luminosity weights
    gray_img = (0.114 * b) + (0.587 * g) + (0.299 * r)
    
    # --- 2. Gaussian Blur (Using new reusable function) ---
    if apply_blur:
        # Extract the integer size from the tuple (assuming a square kernel like (5, 5))
        k_size = blur_ksize[0] if isinstance(blur_ksize, tuple) else blur_ksize
        gray_img = apply_gaussian_filter(gray_img, kernel_size=k_size)
        
    # --- 3. Normalization ---
    gray_img_float = gray_img.astype(np.float64) / 255.0
    
    return gray_img_float

def compute_spatial_gradients(gray_img_float):
    """
    Computes the X and Y image gradients using a 3x3 Sobel operator.
    """
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Ky = np.array([[-1, -2, -1], [ 0,  0,  0], [ 1,  2,  1]])

    Ix = custom_convolve2d(gray_img_float, Kx, mode='same', boundary='symm')
    Iy = custom_convolve2d(gray_img_float, Ky, mode='same', boundary='symm')
    
    return Ix, Iy

def encode_image_to_base64(img_array, ext='.jpg'):
    """
    Encodes an OpenCV image (numpy array) into a Base64 data URI string.
    Ready to be sent via JSON and directly plugged into an HTML <img src="..."> tag.
    """
    success, buffer = cv2.imencode(ext, img_array)
    if not success:
        raise ValueError("Failed to encode image to base64.")
    
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    # Determine mime type based on extension
    mime_type = "image/jpeg" if ext.lower() in ['.jpg', '.jpeg'] else "image/png"
    
    return f"data:{mime_type};base64,{img_base64}"

# ========== NCC UTILITY FUNCTIONS ==========

def base64_to_image_array(base64_str):
    """
    Convert base64 string to numpy image array (grayscale float32)
    """
    # Remove data URL prefix if present
    if 'base64,' in base64_str:
        base64_str = base64_str.split('base64,')[1]
    
    img_data = base64.b64decode(base64_str)
    img = Image.open(io.BytesIO(img_data)).convert('L')
    return np.array(img, dtype=np.float32)

def image_array_to_base64(img_array, format='PNG'):
    """
    Convert numpy image array to base64 string
    """
    # Convert to uint8 if needed
    if img_array.dtype != np.uint8:
        if img_array.max() <= 1.0:
            img_array = (img_array * 255).astype(np.uint8)
        else:
            img_array = img_array.astype(np.uint8)
    
    img = Image.fromarray(img_array)
    buffered = io.BytesIO()
    img.save(buffered, format=format)
    return base64.b64encode(buffered.getvalue()).decode()

def normalize_image(img: np.ndarray) -> np.ndarray:
    """Normalize image to zero mean and unit variance."""
    mean = np.mean(img)
    std = np.std(img)
    if std < 1e-10:
        return img - mean
    return (img - mean) / std

def detect_harris_corners(image: np.ndarray, num_corners: int = 50) -> list:
    """
    Detect corners using Harris detector.
    Returns list of (row, col) coordinates.
    """
    # Convert to uint8 if needed
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            img_uint8 = (image * 255).astype(np.uint8)
        else:
            img_uint8 = image.astype(np.uint8)
    else:
        img_uint8 = image
    
    # Use OpenCV's Harris detector for better performance
    gray = np.float32(img_uint8)
    dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
    
    # Dilate to mark corners
    dst = cv2.dilate(dst, None)
    
    # Get top corners
    dst_flat = dst.flatten()
    indices = np.argsort(dst_flat)[-num_corners:]
    
    corners = []
    h, w = dst.shape
    for idx in indices:
        row = idx // w
        col = idx % w
        corners.append((row, col))
    
    return corners

# ========== SIFT UTILITY FUNCTIONS ==========
def gaussian_blur(image, sigma):
    """Fast Gaussian blur using scipy's optimized function."""
    if sigma == 0:
        return image
    return ndimage.gaussian_filter(image, sigma, mode='reflect')

def build_gaussian_pyramid(image, sigma=1.6, num_octaves=None, num_levels=5):
    """Builds a Gaussian pyramid useful for scale-space extrema detection."""
    if num_octaves is None:
        num_octaves = int(np.log2(min(image.shape))) - 2
    
    pyramid = []
    k = 2 ** (1.0/3)
    
    for octave in range(num_octaves):
        octave_images = []
        if octave == 0:
            current = image
        else:
            current = pyramid[octave-1][3][::2, ::2]
        
        octave_images.append(gaussian_blur(current, sigma))
        
        for level in range(1, num_levels):
            sigma_current = sigma * (k ** level)
            octave_images.append(gaussian_blur(octave_images[0], sigma_current))
        
        pyramid.append(octave_images)
    return pyramid

def compute_dog_pyramid(gaussian_pyramid):
    """Computes Difference of Gaussians (DoG) from a Gaussian pyramid."""
    dog_pyramid = []
    for octave in gaussian_pyramid:
        dog_octave = [octave[i+1] - octave[i] for i in range(len(octave)-1)]
        dog_pyramid.append(dog_octave)
    return dog_pyramid

def remove_duplicates_fast(keypoints, distance_threshold=5):
    """Fast duplicate removal using spatial grid hashing."""
    if len(keypoints) <= 1: return keypoints
    keypoints_sorted = sorted(keypoints, key=lambda x: x['contrast'], reverse=True)
    grid = {}
    unique_keypoints = []

    for kp in keypoints_sorted:
        cell_x, cell_y = int(kp['x'] // distance_threshold), int(kp['y'] // distance_threshold)
        is_duplicate = False
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                cell = (cell_x + dx, cell_y + dy)
                if cell in grid:
                    for existing in grid[cell]:
                        if np.hypot(kp['x'] - existing['x'], kp['y'] - existing['y']) < distance_threshold:
                            is_duplicate = True; break
                if is_duplicate: break
            if is_duplicate: break
        
        if not is_duplicate:
            unique_keypoints.append(kp)
            grid.setdefault((cell_x, cell_y), []).append(kp)
    return unique_keypoints

def match_features(desc1, desc2, ratio_threshold=0.75):
    """Generic feature matcher using KDTree and Lowe's ratio test."""
    if not desc1 or not desc2: return []
    tree = KDTree(desc2)
    distances, indices = tree.query(desc1, k=2)
    return [(i, idx[0]) for i, (d, idx) in enumerate(zip(distances, indices)) if d[0] < ratio_threshold * d[1]]

def draw_keypoints_opencv(image, keypoints):
    """Generic keypoint drawing utility for OpenCV."""
    output_image = image.copy()
    if len(output_image.shape) == 2:
        output_image = cv2.cvtColor(output_image, cv2.COLOR_GRAY2BGR)

    for kp in keypoints:
        center = (int(kp['y']), int(kp['x']))
        radius = max(1, int(kp['scale']))
        cv2.circle(output_image, center, radius, (0, 255, 0), 1)
        cv2.circle(output_image, center, 1, (0, 0, 255), -1)
    return output_image