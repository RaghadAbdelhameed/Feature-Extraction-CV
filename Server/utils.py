import cv2
import numpy as np
import base64
from scipy.signal import convolve2d
from PIL import Image
import io
from scipy import ndimage
from scipy.spatial import KDTree

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
    Converts a BGR image to grayscale, applies optional smoothing, 
    and normalizes pixel values to a 0.0 - 1.0 float range.
    """
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if apply_blur:
        gray_img = cv2.GaussianBlur(gray_img, blur_ksize, 0)
        
    gray_img_float = gray_img.astype(np.float64) / 255.0
    return gray_img_float

def compute_spatial_gradients(gray_img_float):
    """
    Computes the X and Y image gradients using a 3x3 Sobel operator.
    (Used heavily in Harris, Shi-Tomasi, and SIFT).
    """
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Ky = np.array([[-1, -2, -1], [ 0,  0,  0], [ 1,  2,  1]])

    Ix = convolve2d(gray_img_float, Kx, mode='same', boundary='symm')
    Iy = convolve2d(gray_img_float, Ky, mode='same', boundary='symm')
    
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