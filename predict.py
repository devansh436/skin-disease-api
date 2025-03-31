import sys
import json
import numpy as np
from PIL import Image
import os

# Uncomment the appropriate library based on what your CNN was built with
# For TensorFlow/Keras
import tensorflow as tf

# For PyTorch
# import torch
# from torchvision import transforms

def preprocess_image(image_path, target_size=(224, 224)):
    """
    Preprocess the image for the CNN model
    """
    try:
        # Open and resize image
        img = Image.open(image_path)
        img = img.convert('RGB')  # Ensure 3 channels
        img = img.resize(target_size)
        
        # Convert to numpy array and normalize
        img_array = np.array(img) / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array, None
    except Exception as e:
        return None, str(e)

def load_model(model_path):
    """
    Load the saved CNN model
    """
    try:
        # For TensorFlow/Keras
        model = tf.keras.models.load_model(model_path)
        
        # For PyTorch
        # model = torch.load(model_path)
        # model.eval()
        
        return model, None
    except Exception as e:
        return None, str(e)

def main():
    # Validate command line arguments
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No image path provided"}))
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    if not os.path.exists(image_path):
        print(json.dumps({"error": f"Image file not found: {image_path}"}))
        sys.exit(1)
    
    try:
        # Load the model
        model_path = 'model'  # Update this path to where your model is saved
        model, error = load_model(model_path)
        
        if error:
            print(json.dumps({"error": f"Failed to load model: {error}"}))
            sys.exit(1)
        
        # Preprocess the image
        img_array, error = preprocess_image(image_path)
        
        if error:
            print(json.dumps({"error": f"Failed to process image: {error}"}))
            sys.exit(1)
        
        # Make prediction
        # For TensorFlow/Keras
        predictions = model.predict(img_array)
        
        # For PyTorch
        # with torch.no_grad():
        #     img_tensor = torch.from_numpy(img_array).permute(0, 3, 1, 2).float()
        #     outputs = model(img_tensor)
        #     predictions = outputs.numpy()
        
        # Process and return the prediction results
        # Modify this section based on your model's output format
        
        # Example for classification model:
        class_names = ['class1', 'class2', 'class3']  # Replace with your classes
        prediction_index = np.argmax(predictions[0])
        confidence = float(predictions[0][prediction_index])
        predicted_class = class_names[prediction_index]
        
        result = {
            "class": predicted_class,
            "confidence": confidence,
            "predictions": predictions[0].tolist()
        }
        
        print(json.dumps(result))
        
    except Exception as e:
        print(json.dumps({"error": f"Prediction failed: {str(e)}"}))
        sys.exit(1)

if __name__ == "__main__":
    main()