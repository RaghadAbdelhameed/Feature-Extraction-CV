from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import uuid

# Import our standalone function
from harris_detector import run_harris_detector

app = Flask(__name__)
CORS(app) 

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/api/harris', methods=['POST'])
def process_harris():
    # 1. Check for the image
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # 2. Get parameters from the frontend (with defaults just in case)
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

if __name__ == '__main__':
    app.run(debug=True, port=5000)