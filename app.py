from flask import Flask, request, jsonify
from model import predict_mask
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    file_path = 'uploads/image.png'
    file.save(file_path)

    result = predict_mask(file_path)
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
