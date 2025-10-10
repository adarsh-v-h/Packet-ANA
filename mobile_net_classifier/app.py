import os
import json
import numpy as np
from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# --- Configuration (UPDATED) ---
IMG_SIZE = 224  # Standard input size for MobileNetV2
INPUT_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
# *** CHANGE HERE: Added 'IMAGE' class ***
NUM_CLASSES = 5 
CLASSES = ['HTTP', 'DNS', 'SSH', 'UNKNOWN', 'IMAGE'] 
# *******************************
MODEL_PATH = 'mobilenet_v2_classifier.h5'

app = Flask(__name__)
model = None

def build_and_load_model():
    """
    Builds the MobileNetV2 model using transfer learning.
    This structure must reflect the 5 classes we now want to predict.
    """
    print("Building MobileNetV2 base model...")
    # Load MobileNetV2 base, excluding the final classification layer
    base_model = MobileNetV2(
        input_shape=INPUT_SHAPE,
        include_top=False,
        weights='imagenet' # Use weights pre-trained on ImageNet
    )

    # Freeze the base model layers
    base_model.trainable = False

    # Add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    
    # Add a dense layer for our final classification (5 classes)
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)

    # This is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    # NOTE: In a real system, we would load the weights here:
    # model.load_weights(MODEL_PATH)
    
    # For demonstration, we simply return the structured model.
    print(f"Model successfully built for {NUM_CLASSES} classes.")
    return model

@app.before_request
def initialize_model():
    """Initializes the model once when the first request comes in."""
    global model
    if model is None:
        try:
            model = build_and_load_model()
        except Exception as e:
            print(f"FATAL: Could not load/build TensorFlow model: {e}")
            # If the model fails to load, we allow the request to proceed but it will fail later
            pass 

@app.route('/classify', methods=['POST'])
def classify_data():
    """Receives the tensor list, converts it back to a tensor, and performs inference."""
    if model is None:
        return jsonify({"status": "error", "error": "Model failed to initialize on server."}), 500
        
    try:
        data = request.get_json()
        if not data or 'tensor' not in data:
            return jsonify({"error": "No tensor data provided"}), 400
        
        # 1. Convert the JSON list back to a numpy array
        tensor_list = data['tensor']
        flattened_tensor = np.array(tensor_list, dtype=np.float32)

        # Sanity check: ensure the tensor has the expected flat size
        if flattened_tensor.size != IMG_SIZE * IMG_SIZE * 3:
            return jsonify({
                "error": "Tensor size mismatch",
                "expected": IMG_SIZE * IMG_SIZE * 3,
                "received": flattened_tensor.size
            }), 400

        # 2. Reshape to the required MobileNetV2 input shape (Batch, H, W, Channels)
        image_tensor = flattened_tensor.reshape((1, IMG_SIZE, IMG_SIZE, 3))

        # 3. Perform MobileNetV2 specific preprocessing 
        # (scales input pixels between -1 and 1)
        preprocessed_input = preprocess_input(image_tensor)

        # 4. Run Inference
        predictions = model.predict(preprocessed_input)

        # 5. Get the highest confidence prediction
        predicted_class_index = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_index])
        # Use the updated CLASSES list
        predicted_class = CLASSES[predicted_class_index]

        response_data = {
            "status": "success",
            "predicted_class": predicted_class,
            "confidence": f"{confidence:.4f}",
            "full_prediction": {c: f"{float(p):.4f}" for c, p in zip(CLASSES, predictions[0])}
        }
        
        return jsonify(response_data)

    except Exception as e:
        print(f"Inference Error: {e}")
        return jsonify({"error": f"Internal server error during inference: {str(e)}"}), 500

if __name__ == '__main__':
    # Initialize model on startup for faster first request
    initialize_model()
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
