"""
Flask backend — Feature Extraction CV
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os, uuid, sys, time
import numpy as np
import cv2

from harris_detector import run_harris_detector
from sift_detector   import detect_sift_features_fast
from ncc_matcher     import (
    detect_and_match_features,
    ncc_template_match,
    visualize_matches,
)
from utils import (
    base64_to_image_array,
    image_array_to_base64,
    draw_keypoints_opencv,
    encode_image_to_base64,
)

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def log(msg):
    print(msg, flush=True)


# =========================================================================== #
#  HARRIS                                                                      #
# =========================================================================== #

@app.route('/api/harris', methods=['POST'])
def process_harris():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    method     = request.form.get('method', 'harris')
    block_size = int(request.form.get('block_size', 3))
    k_value    = float(request.form.get('k', 0.04))
    threshold  = float(request.form.get('threshold_ratio', 0.1))

    unique_name = f"upload_{uuid.uuid4().hex}.png"
    temp_path   = os.path.join(UPLOAD_FOLDER, unique_name)
    file.save(temp_path)
    try:
        result = run_harris_detector(
            image_path=temp_path, method=method,
            block_size=block_size, k=k_value, threshold_ratio=threshold
        )
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


# =========================================================================== #
#  SIFT                                                                        #
# =========================================================================== #

@app.route('/api/sift', methods=['POST'])
def process_sift():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    try:
        contrast_thr = float(request.form.get('contrast_thr', 0.04))
        edge_thr     = float(request.form.get('edge_thr', 10.0))
        sigma        = float(request.form.get('sigma', 1.6))
    except ValueError:
        return jsonify({'error': 'Invalid parameter types'}), 400

    file       = request.files['image']
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img        = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({'error': 'Invalid image file'}), 400

    t0 = time.time()
    keypoints, _ = detect_sift_features_fast(
        img, contrast_threshold=contrast_thr, edge_threshold=edge_thr, sigma=sigma
    )
    computation_time = time.time() - t0

    result_img    = draw_keypoints_opencv(img, keypoints)
    result_base64 = encode_image_to_base64(result_img, ext='.jpg')

    return jsonify({
        'result_image_base64':      result_base64,
        'computation_time_seconds': computation_time,
        'total_keypoints':          len(keypoints),
    })


# =========================================================================== #
#  NCC / SSD  FEATURE MATCHING                                                 #
# =========================================================================== #

@app.route('/api/ncc/feature-match', methods=['POST', 'OPTIONS'])
def ncc_feature_match():
    if request.method == 'OPTIONS':
        return jsonify({}), 200
    try:
        log("=" * 60)
        log("FEATURE-MATCH REQUEST")

        data = request.json
        if 'image1' not in data or 'image2' not in data:
            return jsonify({"error": "Missing image1 or image2"}), 400

        patch_size    = int(data.get('patch_size',    21))
        num_keypoints = int(data.get('num_keypoints', 200))
        ratio_thresh  = float(data.get('ratio_thresh', 0.85))
        method        = str(data.get('method', 'ncc')).lower()   # 'ncc' or 'ssd'

        if patch_size % 2 == 0:
            patch_size += 1

        log(f"method={method}, patch_size={patch_size}, "
            f"num_keypoints={num_keypoints}, ratio_thresh={ratio_thresh}")

        # Decode — float32 grayscale, pixel range 0–255
        img1 = base64_to_image_array(data['image1'])
        img2 = base64_to_image_array(data['image2'])
        log(f"img1={img1.shape}  img2={img2.shape}")

        result = detect_and_match_features(
            img1, img2,
            patch_size=patch_size,
            num_keypoints=num_keypoints,
            ratio_thresh=ratio_thresh,
            method=method,
        )

        # Build visualisation (needs descriptors for colour mapping)
        viz = visualize_matches(
            img1, img2,
            [(m['idx1'], m['idx2']) for m in result['matches']],
            result['keypoints1'],
            result['keypoints2'],
            result['descriptors1'],
            result['descriptors2'],
        )
        viz_b64 = image_array_to_base64(viz)

        log(f"SUCCESS: {result['num_matches']} matches  "
            f"time={result['computational_time']:.4f}s")
        log("=" * 60)

        return jsonify({
            "success":            True,
            "matches":            result['matches'][:100],
            "num_matches":        result['num_matches'],
            "computational_time": result['computational_time'],
            "visualization":      f"data:image/png;base64,{viz_b64}",
            "num_keypoints1":     len(result['keypoints1']),
            "num_keypoints2":     len(result['keypoints2']),
        }), 200

    except Exception as e:
        log(f"ERROR: {e}")
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# =========================================================================== #
#  NCC TEMPLATE MATCHING                                                       #
# =========================================================================== #

@app.route('/api/ncc/template-match', methods=['POST', 'OPTIONS'])
def ncc_template_match_endpoint():
    if request.method == 'OPTIONS':
        return jsonify({}), 200
    try:
        data = request.json
        if 'template_image' not in data or 'search_image' not in data:
            return jsonify({"error": "Missing template_image or search_image"}), 400

        template_img = base64_to_image_array(data['template_image'])
        search_img   = base64_to_image_array(data['search_image'])
        result       = ncc_template_match(template_img, search_img)

        if 'error' in result:
            return jsonify({"error": result['error']}), 400

        ncc_map = result['ncc_map']
        mn, mx  = ncc_map.min(), ncc_map.max()
        ncc_u8  = ((ncc_map - mn) / (mx - mn + 1e-8) * 255).astype(np.uint8)
        ncc_b64 = image_array_to_base64(ncc_u8)

        # Draw bounding box
        si       = np.stack([search_img.astype(np.uint8)] * 3, axis=-1)
        r, c     = result['best_location']
        t_h, t_w = result['template_shape']
        r1       = min(r + t_h, si.shape[0] - 1)
        c1       = min(c + t_w, si.shape[1] - 1)
        si[r,    c:c1+1] = [0, 255, 0]
        si[r1,   c:c1+1] = [0, 255, 0]
        si[r:r1+1, c   ] = [0, 255, 0]
        si[r:r1+1, c1  ] = [0, 255, 0]
        res_b64  = image_array_to_base64(si)

        return jsonify({
            "success":            True,
            "best_location":      list(result['best_location']),
            "best_ncc":           result['best_ncc'],
            "computational_time": result['computational_time'],
            "ncc_map":            f"data:image/png;base64,{ncc_b64}",
            "result_image":       f"data:image/png;base64,{res_b64}",
        }), 200

    except Exception as e:
        log(f"ERROR in template matching: {e}")
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# =========================================================================== #
#  HEALTH                                                                      #
# =========================================================================== #

@app.route('/api/ncc/health', methods=['GET'])
def ncc_health():
    return jsonify({"status": "healthy", "service": "NCC Matcher"}), 200


# =========================================================================== #
#  ENTRY POINT                                                                 #
# =========================================================================== #

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  Feature Extraction CV — Backend")
    print("  http://localhost:5000")
    print("  POST /api/harris")
    print("  POST /api/sift")
    print("  POST /api/ncc/feature-match   (method: ncc | ssd)")
    print("  POST /api/ncc/template-match")
    print("  GET  /api/ncc/health")
    print("=" * 60 + "\n")
    app.run(debug=True, port=5000)
