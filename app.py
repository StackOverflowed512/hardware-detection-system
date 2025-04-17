from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from object_detection import HardwareDetector

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max-limit

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize detector
detector = HardwareDetector()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process image
        image = cv2.imread(filepath)
        result_image, hardware_items = detector.detect_hardware(image)
        
        # Save result image
        result_filename = 'result_' + filename
        result_filepath = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
        cv2.imwrite(result_filepath, result_image)
        
        # Prepare results
        results = {
            'original_image': filename,
            'result_image': result_filename,
            'items': [
                {
                    'id': item['id'],
                    'type': item['type'],
                    'dimensions': item['dimensions']
                }
                for item in hardware_items
            ]
        }
        
        return jsonify(results)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)
