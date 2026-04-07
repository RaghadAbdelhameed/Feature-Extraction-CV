import cv2
import numpy as np
import base64
from scipy.signal import convolve2d

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