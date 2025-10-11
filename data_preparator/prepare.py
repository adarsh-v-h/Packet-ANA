import os
import numpy as np
import random
from scapy.all import Ether, IP, TCP, Raw

# --- Configuration (Must match the classifier's expected input) ---
IMG_SIZE = 224
IMG_CHANNELS = 3
TENSOR_FLATTENED_SIZE = IMG_SIZE * IMG_SIZE * IMG_CHANNELS # 150528

# Define the root directory for saving prepared data (mounted to the Docker volume)
DATA_OUTPUT_ROOT = '/app/processed_data' 
LABELS = ['HTTP', 'DNS', 'SSH', 'UNKNOWN'] 
NUM_SAMPLES_PER_CLASS = 10 # Generate 10 mock samples for now

def generate_mock_packet_bytes(label):
    """Generates mock packet bytes based on the desired label."""
    if label == 'HTTP':
        # Simulate a typical HTTP request packet structure
        packet = Ether()/IP(src="10.0.0.1", dst="1.1.1.1")/TCP(dport=80)/Raw(load=b"GET /index.html HTTP/1.1\r\n")
        raw_bytes = bytes(packet)
    elif label == 'DNS':
        # Simulate a DNS query packet structure
        packet = Ether()/IP(src="10.0.0.1", dst="8.8.8.8")/Raw(load=os.urandom(64))
        raw_bytes = bytes(packet)
    else:
        # Default/Unknown with random padding
        raw_bytes = os.urandom(random.randint(100, 1500))
    
    # Pad/Truncate bytes to match the required flattened tensor size
    num_bytes = len(raw_bytes)
    target_size = TENSOR_FLATTENED_SIZE
    
    if num_bytes < target_size:
        padded_bytes = raw_bytes + b'\x00' * (target_size - num_bytes)
    else:
        padded_bytes = raw_bytes[:target_size]
        
    return padded_bytes

def save_tensor_to_disk(tensor_list, label, sample_id):
    """Saves the tensor as a numpy file in the correct labeled directory."""
    
    # Create the directory structure: processed_data/HTTP/
    label_dir = os.path.join(DATA_OUTPUT_ROOT, label)
    os.makedirs(label_dir, exist_ok=True)
    
    # Convert the list back to a numpy array (1D) for saving
    tensor_array = np.array(tensor_list, dtype=np.float32)
    
    # Save the numpy array file
    filepath = os.path.join(label_dir, f"{label}_{sample_id}.npy")
    np.save(filepath, tensor_array)
    
    print(f"Saved: {filepath} ({len(tensor_array)} elements)")


def main_data_preparation():
    print("--- Data Preparation Service Starting ---")
    
    # Ensure the root output directory exists
    os.makedirs(DATA_OUTPUT_ROOT, exist_ok=True)
    
    for label in LABELS:
        print(f"\nProcessing mock data for class: {label}")
        for i in range(NUM_SAMPLES_PER_CLASS):
            
            # 1. Get raw packet bytes (Simulated PCAP reading here)
            raw_bytes = generate_mock_packet_bytes(label)
            
            # 2. Convert to list of floats (the 'tensor')
            tensor_list = [float(b) for b in raw_bytes]
            
            # 3. Save the tensor file
            save_tensor_to_disk(tensor_list, label, i + 1)
            
    print("\n--- Data Preparation Complete ---")

if __name__ == '__main__':
    main_data_preparation()
