import cv2
import numpy as np
import base64
from scipy.signal import convolve2d
from PIL import Image
import io

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