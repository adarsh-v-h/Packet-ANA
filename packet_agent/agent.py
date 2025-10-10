import json
import time
import random
import numpy as np
import requests
import base64
from io import BytesIO
from PIL import Image
from flask import Flask, jsonify, request

# Configuration
IMG_SIZE = 224
IMG_CHANNELS = 3
TENSOR_FLATTENED_SIZE = IMG_SIZE * IMG_SIZE * IMG_CHANNELS # 150528
CLASSIFIER_URL = "http://mobile-net-classifier:5000/classify"

# Define the simulated traffic types (Now 5 classes)
TRAFFIC_TYPES = {
    "HTTP": b"GET /index.html HTTP/1.1\r\nHost: example.com\r\n\r\n",
    "DNS": b"\xaa\xaa\x01\x00\x00\x01\x00\x00\x00\x00\x00\x00\x03www\x06google\x03com\x00\x00\x01\x00\x00",
    "SSH": b"SSH-2.0-OpenSSH_7.4p1 Debian-10+deb9u3\r\n" + b"A" * 50,
    "UNKNOWN": b"Z" * 150000,
    # NEW SIMULATION CLASS: IMAGE - Represents complex, high-entropy data like an image.
    "IMAGE": np.random.bytes(TENSOR_FLATTENED_SIZE) 
}

app = Flask(__name__)

def process_raw_to_tensor(raw_bytes, mode='SIMULATED'):
    """
    Converts raw bytes (from simulation or image) into a normalized tensor list.
    """
    if len(raw_bytes) > TENSOR_FLATTENED_SIZE:
        packet_bytes = raw_bytes[:TENSOR_FLATTENED_SIZE]
    else:
        padding_needed = TENSOR_FLATTENED_SIZE - len(raw_bytes)
        packet_bytes = raw_bytes + b'\x00' * padding_needed

    np_array = np.frombuffer(packet_bytes, dtype=np.uint8)
    normalized_array = np_array.astype(np.float32) / 255.0
    tensor_list = normalized_array.tolist()
    return tensor_list

def send_to_classifier(tensor_list):
    """Handles the network request to the MobileNetV2 service."""
    response = requests.post(
        CLASSIFIER_URL,
        json={'tensor': tensor_list},
        timeout=10
    )
    response.raise_for_status()
    return response.json()

@app.route('/api/classify_random', methods=['GET'])
def classify_random_packet():
    """Triggers a simulation and classification for one of the five types."""
    
    # Choose from the five structured types defined in TRAFFIC_TYPES
    traffic_type_name, raw_bytes = random.choice(list(TRAFFIC_TYPES.items()))
    
    print(f"[AGENT] Simulating traffic type: {traffic_type_name}")

    tensor_list = process_raw_to_tensor(raw_bytes, 'SIMULATED')
    
    print(f"[AGENT] Sending simulated {traffic_type_name} packet data to classifier...")

    try:
        classification_result = send_to_classifier(tensor_list)
        predicted_class = classification_result.get('predicted_class', 'N/A')
        confidence = classification_result.get('confidence', 'N/A')
        
        print(f"[AGENT] Classification Result: {predicted_class}")

        return jsonify({
            "status": "success",
            "source": "Simulation",
            "simulated_type": traffic_type_name,
            "predicted_class": predicted_class,
            "confidence": confidence,
            "tensor_data": tensor_list
        })

    except requests.exceptions.ConnectionError:
        return jsonify({
            "status": "error", 
            "error": "Could not connect to the Classifier service. Is it running?"
        }), 503
    except requests.exceptions.RequestException as e:
        return jsonify({
            "status": "error", 
            "error": f"Request failed: {str(e)}"
        }), 500

@app.route('/api/classify_upload', methods=['POST'])
def classify_uploaded_image():
    """Receives a Base64 image, processes it, and classifies it."""
    try:
        data = request.get_json()
        if not data or 'image_data' not in data:
            return jsonify({"status": "error", "error": "No image data provided."}), 400
        
        # Decode Base64 string to image bytes
        image_b64 = data['image_data'].split(',')[1] 
        image_bytes = base64.b64decode(image_b64)
        
        # Use PIL (Pillow) to open, resize, and convert the image
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        img = img.resize((IMG_SIZE, IMG_SIZE))
        
        # Convert image to raw byte array
        img_byte_array = np.array(img).tobytes()
        
        print(f"[AGENT] Processing uploaded image, converted to {len(img_byte_array)} bytes.")
        
        # Convert raw image bytes to the flattened 150528-element tensor
        tensor_list = process_raw_to_tensor(img_byte_array, 'UPLOAD')

        # Send to classifier
        classification_result = send_to_classifier(tensor_list)
        
        predicted_class = classification_result.get('predicted_class', 'N/A')
        confidence = classification_result.get('confidence', 'N/A')
        
        print(f"[AGENT] Classification Result for Upload: {predicted_class}")

        return jsonify({
            "status": "success",
            "source": "Upload",
            "predicted_class": predicted_class,
            "confidence": confidence,
            "tensor_data": tensor_list 
        })

    except requests.exceptions.ConnectionError:
        return jsonify({
            "status": "error", 
            "error": "Could not connect to the Classifier service."
        }), 503
    except Exception as e:
        print(f"Error processing image: {e}")
        return jsonify({
            "status": "error", 
            "error": f"Error processing image or request: {str(e)}"
        }), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)