import torch
from transformers import SamModel, SamProcessor
from PIL import Image
import numpy as np
import os
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/predict', methods=['POST'])
def predict():
    # Your prediction logic here
    return jsonify({'message': 'Prediction successful'})

# Load the SAM model and processor
try:
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    model = SamModel.from_pretrained("facebook/sam-vit-base").to("cuda" if torch.cuda.is_available() else "cpu")
    print("Model and processor loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")

# Function to process an image and get predictions
def predict_mask(image_path):
    try:
        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        image = image.resize((512, 512))
        inputs = processor(images=image, return_tensors="pt").to(model.device)
        
        h, w = image.size
        input_points = torch.tensor([[[[w // 2, h // 2]]]], dtype=torch.float32, device=model.device)
        input_labels = torch.tensor([[[1]]], dtype=torch.float32, device=model.device)

        inputs["input_points"] = input_points
        inputs["input_labels"] = input_labels

        # Perform inference
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Extract predictions
        predicted_mask = outputs.pred_masks.squeeze().cpu().numpy()
        binary_mask = (predicted_mask > 0.5).astype(np.uint8)
        
        return {
            "success": True,
            "mask_shape": binary_mask.shape,
            "mask_sum": int(np.sum(binary_mask)),
            "has_cancer": bool(np.sum(binary_mask) > 100),
            "binary_mask": binary_mask
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

# If running directly, process the test image
if __name__ == "__main__":
    result = predict_mask("image.png")
    if result["success"]:
        print("Prediction result\nHas cancer =", result['has_cancer'])
    else:
        print("Prediction failed:", result["error"])