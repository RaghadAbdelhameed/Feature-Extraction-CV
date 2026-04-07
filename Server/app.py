from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import uuid
import numpy as np
import cv2
import sys

# Import our standalone function
from harris_detector import run_harris_detector
# Import NCC functions
from ncc_matcher import ncc_matching_robust, detect_and_match_features, visualize_match_result
# Import utils
from utils import base64_to_image_array, image_array_to_base64, detect_harris_corners

app = Flask(__name__)
CORS(app) 

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Add print to console with flush
def log(message):
    print(message, flush=True)
    sys.stdout.flush()

@app.route('/api/harris', methods=['POST'])
def process_harris():
    # 1. Check for the image
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # 2. Get parameters from the frontend (with defaults just in case)
    method = request.form.get('method', 'harris')
    block_size = int(request.form.get('block_size', 3))
    k_value = float(request.form.get('k', 0.04))
    threshold = float(request.form.get('threshold_ratio', 0.1))

    # 3. Save the image to the uploads folder temporarily WITH A UNIQUE NAME
    unique_filename = f"upload_{uuid.uuid4().hex}.png"
    temp_path = os.path.join(UPLOAD_FOLDER, unique_filename)
    file.save(temp_path)

    try:
        # 4. Run the isolated Harris logic
        result_data = run_harris_detector(
            image_path=temp_path,
            method=method,
            block_size=block_size,
            k=k_value,
            threshold_ratio=threshold
        )
        
        # 5. Clean up: Delete the temporary file after processing is done
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        # 6. Send back the dictionary containing the time, points, and Base64 image
        return jsonify(result_data), 200

    except Exception as e:
        # Clean up the file even if the script crashes so the folder doesn't fill up
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({"error": str(e)}), 500

# ========== NCC ENDPOINTS ==========

@app.route('/api/ncc/feature-match', methods=['POST', 'OPTIONS'])
def ncc_feature_match():
    """Endpoint for feature matching using NCC"""
    if request.method == 'OPTIONS':
        return jsonify({}), 200
    
    try:
        log("="*60)
        log("NCC FEATURE MATCH REQUEST RECEIVED")
        log("="*60)
        
        data = request.json
        
        if 'image1' not in data or 'image2' not in data:
            return jsonify({"error": "Missing image1 or image2"}), 400
        
        # Get parameters with defaults
        patch_size = data.get('patch_size', 15)
        num_keypoints = data.get('num_keypoints', 50)
        ratio_thresh = data.get('ratio_thresh', 0.9)
        
        log(f"Parameters: patch_size={patch_size}, num_keypoints={num_keypoints}, ratio_thresh={ratio_thresh}")
        
        # Ensure patch size is odd
        if patch_size % 2 == 0:
            patch_size += 1
            log(f"Adjusted patch_size to odd: {patch_size}")
        
        # Convert base64 to numpy arrays
        log("Converting base64 images to arrays...")
        img1 = base64_to_image_array(data['image1'])
        img2 = base64_to_image_array(data['image2'])
        
        log(f"Image1 shape: {img1.shape}, dtype: {img1.dtype}")
        log(f"Image2 shape: {img2.shape}, dtype: {img2.dtype}")
        log(f"Image1 range: [{img1.min():.2f}, {img1.max():.2f}]")
        log(f"Image2 range: [{img2.min():.2f}, {img2.max():.2f}]")
        
        # Perform feature matching
        log("\nStarting feature matching...")
        result = detect_and_match_features(
            img1, img2,
            patch_size=patch_size,
            num_keypoints=num_keypoints,
            ratio_thresh=ratio_thresh
        )
        
        # Create visualization of matches
        log("\nCreating visualization...")
        viz_image = visualize_match_result(
            img1, img2, 
            result['matches'], 
            result['keypoints1'], 
            result['keypoints2']
        )
        
        # Convert visualization to base64
        viz_b64 = image_array_to_base64(viz_image)
        
        log(f"\nSUCCESS: Found {result['num_matches']} matches")
        log("="*60)
        
        return jsonify({
            "success": True,
            "matches": result['matches'][:100],  # Limit to first 100 matches
            "num_matches": result['num_matches'],
            "computational_time": result['computational_time'],
            "visualization": f"data:image/png;base64,{viz_b64}",
            "num_keypoints1": len(result['keypoints1']),
            "num_keypoints2": len(result['keypoints2'])
        }), 200
        
    except Exception as e:
        log(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/ncc/template-match', methods=['POST', 'OPTIONS'])
def ncc_template_match():
    """Endpoint for NCC template matching"""
    if request.method == 'OPTIONS':
        return jsonify({}), 200
    
    try:
        data = request.json
        
        if 'template_image' not in data or 'search_image' not in data:
            return jsonify({"error": "Missing template_image or search_image"}), 400
        
        # Convert base64 to numpy arrays
        template_img = base64_to_image_array(data['template_image'])
        search_img = base64_to_image_array(data['search_image'])
        
        # Perform NCC matching
        result = ncc_matching_robust(template_img, search_img)
        
        if 'error' in result:
            return jsonify({"error": result['error']}), 400
        
        # Normalize NCC map for visualization
        ncc_map = result['ncc_map']
        ncc_map_normalized = (ncc_map - np.min(ncc_map)) / (np.max(ncc_map) - np.min(ncc_map) + 1e-8)
        ncc_map_uint8 = (ncc_map_normalized * 255).astype(np.uint8)
        
        # Convert NCC map to base64
        ncc_map_b64 = image_array_to_base64(ncc_map_uint8)
        
        # Create visualization of best match
        search_img_uint8 = search_img.astype(np.uint8) if search_img.max() <= 255 else (search_img / search_img.max() * 255).astype(np.uint8)
        search_img_color = cv2.cvtColor(search_img_uint8, cv2.COLOR_GRAY2BGR)
        
        best_row, best_col = result['best_location']
        t_h, t_w = result['template_shape']
        
        # Draw rectangle around matched region
        cv2.rectangle(search_img_color, 
                     (best_col, best_row), 
                     (best_col + t_w, best_row + t_h), 
                     (0, 255, 0), 2)
        
        # Convert result image to base64
        result_img_b64 = image_array_to_base64(search_img_color)
        
        return jsonify({
            "success": True,
            "best_location": result['best_location'],
            "best_ncc": result['best_ncc'],
            "computational_time": result['computational_time'],
            "ncc_map": f"data:image/png;base64,{ncc_map_b64}",
            "result_image": f"data:image/png;base64,{result_img_b64}"
        }), 200
        
    except Exception as e:
        log(f"ERROR in template matching: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/ncc/debug', methods=['POST', 'OPTIONS'])
def ncc_debug():
    """Debug endpoint to see what's happening with images"""
    if request.method == 'OPTIONS':
        return jsonify({}), 200
    
    try:
        data = request.json
        
        if 'image1' not in data or 'image2' not in data:
            return jsonify({"error": "Missing images"}), 400
        
        # Convert images
        img1 = base64_to_image_array(data['image1'])
        img2 = base64_to_image_array(data['image2'])
        
        # Detect corners
        corners1 = detect_harris_corners(img1, 50)
        corners2 = detect_harris_corners(img2, 50)
        
        # Get image info
        info = {
            "image1_shape": img1.shape,
            "image2_shape": img2.shape,
            "image1_min_max": [float(np.min(img1)), float(np.max(img1))],
            "image2_min_max": [float(np.min(img2)), float(np.max(img2))],
            "corners1_count": len(corners1),
            "corners2_count": len(corners2),
            "sample_corners1": corners1[:5] if corners1 else [],
            "sample_corners2": corners2[:5] if corners2 else []
        }
        
        log(f"Debug info: {info}")
        
        return jsonify({"success": True, "debug_info": info}), 200
        
    except Exception as e:
        log(f"Debug error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/ncc/health', methods=['GET'])
def ncc_health():
    """Health check endpoint for NCC service"""
    return jsonify({"status": "healthy", "service": "NCC Matcher"}), 200



if __name__ == '__main__':
    print("\n" + "="*60)
    print("Starting NCC Feature Matching Server")
    print("="*60)
    print("Server running on http://localhost:5000")
    print("Endpoints:")
    print("  - POST /api/ncc/feature-match")
    print("  - POST /api/ncc/template-match")
    print("  - POST /api/ncc/debug")
    print("  - GET  /api/ncc/health")
    print("="*60 + "\n")
    app.run(debug=True, port=5000)