import os
import numpy as np
from scapy.all import rdpcap, IP, TCP, UDP, Raw

# --- Configuration (Must match the classifier's expected input) ---
# Note: The Classifier (MobileNetV2) expects a tensor of size 224x224x3 (150528 elements)
IMG_SIZE = 224
IMG_CHANNELS = 3
TENSOR_FLATTENED_SIZE = IMG_SIZE * IMG_SIZE * IMG_CHANNELS # 150528

# Define the root directory for saving prepared data (mounted to the Docker volume)
DATA_OUTPUT_ROOT = '/app/processed_data' 
INPUT_PCAP_FILE = 'monday_traffic.pcap' # Assuming you place your downloaded file here

# Define the classification mapping based on common ports
# Only packets matching these protocols will be saved. Others are skipped.
PROTOCOL_MAP = {
    # Port 80
    80: 'HTTP',
    # Port 53 (DNS)
    53: 'DNS',
    # Port 22 (SSH)
    22: 'SSH',
    # We will exclude 'UNKNOWN' for now, focusing only on labeled data for training
}
# Define the final classes for the model
CLASSES = ['HTTP', 'DNS', 'SSH']
NUM_PACKETS_TO_PROCESS = 10000 # Limit processing for faster debugging

def get_protocol_label(packet):
    """
    Analyzes the packet to determine its application-layer protocol based on ports.
    
    Args:
        packet: A Scapy packet object.
    
    Returns:
        The protocol label string (e.g., 'HTTP', 'DNS', 'SSH') or None if not classified.
    """
    try:
        if IP in packet:
            # Check for TCP protocols
            if TCP in packet:
                dport = packet[TCP].dport
                sport = packet[TCP].sport
                # Check both source and destination port
                if dport in PROTOCOL_MAP:
                    return PROTOCOL_MAP[dport]
                if sport in PROTOCOL_MAP:
                    return PROTOCOL_MAP[sport]
            
            # Check for UDP protocols
            elif UDP in packet:
                dport = packet[UDP].dport
                sport = packet[UDP].sport
                # Check both source and destination port for DNS (Port 53)
                if dport == 53 or sport == 53:
                    return 'DNS' # DNS usually uses UDP
        
        return None
    except Exception as e:
        # print(f"Error parsing packet: {e}")
        return None


def extract_and_convert(packet_bytes):
    """
    Converts raw packet bytes into the flattened tensor format required by the model.
    """
    raw_bytes = bytes(packet_bytes)
    num_bytes = len(raw_bytes)
    target_size = TENSOR_FLATTENED_SIZE
    
    # 1. Pad or Truncate bytes to match the required flattened tensor size (150528)
    if num_bytes < target_size:
        # Pad with zeros if packet is too small
        padded_bytes = raw_bytes + b'\x00' * (target_size - num_bytes)
    else:
        # Truncate if packet is too large
        padded_bytes = raw_bytes[:target_size]
        
    # 2. Convert raw byte values (0-255) to a list of floats
    tensor_list = [float(b) for b in padded_bytes]
    
    return tensor_list


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
    
    # print(f"Saved: {filepath} ({len(tensor_array)} elements)")


def main_data_preparation():
    print("--- Data Preparation Service Starting ---")
    print(f"Target Tensor Size: {TENSOR_FLATTENED_SIZE} elements")

    # 1. Check for input file
    if not os.path.exists(INPUT_PCAP_FILE):
        print(f"\nERROR: Input file '{INPUT_PCAP_FILE}' not found in the directory.")
        print("Please download a PCAP file (e.g., from CIC-IDS 2017) and name it 'monday_traffic.pcap' in the data_preparator/ directory.")
        return

    # 2. Load the PCAP file
    print(f"Loading PCAP file: {INPUT_PCAP_FILE}...")
    try:
        packets = rdpcap(INPUT_PCAP_FILE)
        print(f"Successfully loaded {len(packets)} packets.")
    except Exception as e:
        print(f"FATAL ERROR: Could not read PCAP file. Is it corrupted? Error: {e}")
        return

    # Initialize counters for each class
    sample_counts = {cls: 0 for cls in CLASSES}
    
    # 3. Process packets
    for i, packet in enumerate(packets):
        if i >= NUM_PACKETS_TO_PROCESS:
            print(f"Stopping after processing limit of {NUM_PACKETS_TO_PROCESS} packets.")
            break

        # A. Determine the protocol label
        label = get_protocol_label(packet)
        
        # B. If classified and has raw data, process it
        if label and label in CLASSES:
            try:
                # Extract the raw packet bytes (including link/network/transport headers)
                raw_packet_bytes = bytes(packet)
                
                # Convert the bytes to the tensor format
                tensor_list = extract_and_convert(raw_packet_bytes)
                
                # C. Save the result
                sample_counts[label] += 1
                save_tensor_to_disk(tensor_list, label, sample_counts[label])
                
            except Exception as e:
                print(f"Warning: Failed to process or save packet {i}. Error: {e}")
                
    # 4. Summary
    print("\n--- Data Preparation Complete ---")
    print("Summary of samples saved:")
    for cls, count in sample_counts.items():
        print(f"  {cls}: {count} samples")
    print(f"Output saved to the shared volume at: {DATA_OUTPUT_ROOT}")


if __name__ == '__main__':
    main_data_preparation()
