import os
import glob
import numpy as np
from scapy.all import rdpcap # Scapy is essential for reading PCAP files

# --- Configuration (Must match train.py and agent.py) ---
IMG_SIZE = 224
IMG_CHANNELS = 3
TENSOR_FLATTENED_SIZE = IMG_SIZE * IMG_SIZE * IMG_CHANNELS # 150528

# Define input and output directories inside the Docker container
INPUT_PCAP_DIR = '/app/raw_pcaps'
OUTPUT_TENSOR_DIR = '/app/processed_data'

# --- Core Functions ---

def process_raw_to_tensor(raw_bytes):
    """
    Converts raw bytes to a 1D tensor of fixed size (150528 elements).
    Pads with zeros or truncates to fit the required size.
    This function is identical to the one used in train.py for consistency.
    """
    padding_size = TENSOR_FLATTENED_SIZE - len(raw_bytes)
    
    if padding_size > 0:
        # Pad with zeros to fill the space
        final_bytes = raw_bytes + b'\x00' * padding_size
    else:
        # Truncate to fit
        final_bytes = raw_bytes[:TENSOR_FLATTENED_SIZE]

    # Convert to numpy array of pixel values (0-255)
    tensor = np.frombuffer(final_bytes, dtype=np.uint8)
    return tensor


def process_pcap_to_tensors(filepath, class_label):
    """
    Reads a single PCAP/Log file, extracts raw data,
    converts it to a tensor, and saves the resulting tensors in a list.
    """
    print(f"Reading file: {os.path.basename(filepath)}...")
    tensors = []
    
    if filepath.endswith(('.pcap', '.cap', '.libpcap', '.dmp')):
        # --- Handle standard PCAP files using scapy ---
        try:
            packets = rdpcap(filepath)
            
            for packet in packets:
                raw_bytes = bytes(packet) 
                if len(raw_bytes) > 0:
                    tensor = process_raw_to_tensor(raw_bytes)
                    tensors.append(tensor)
            print(f"  -> Extracted {len(tensors)} tensors from PCAP.")
            
        except Exception as e:
            print(f"Error processing PCAP {filepath}: {e}")
            return []
            
    elif filepath.endswith(('.log', '.txt')):
        # --- Handle raw log/payload files (treats entire file as one or more payloads) ---
        try:
            with open(filepath, 'rb') as f:
                # Read the entire file content as raw bytes
                raw_bytes = f.read()
            
            if len(raw_bytes) > 0:
                # If the log file is massive, we can slice it into multiple tensors.
                # For Snort log files, it's often best to treat them as individual large packets.
                
                # --- SIMPLE MODE: Treat file content as a single packet payload ---
                tensor = process_raw_to_tensor(raw_bytes)
                tensors.append(tensor)
                print(f"  -> Extracted 1 tensor from log file (treating as single large payload).")
                
                # OPTIONAL: Advanced mode to split a massive log file into chunks
                # for i in range(0, len(raw_bytes), TENSOR_FLATTENED_SIZE):
                #     chunk = raw_bytes[i:i + TENSOR_FLATTENED_SIZE]
                #     tensor = process_raw_to_tensor(chunk)
                #     tensors.append(tensor)
                # print(f"  -> Extracted {len(tensors)} tensors from log file (chunked).")
                
            
        except Exception as e:
            print(f"Error processing Log file {filepath}: {e}")
            return []
            
    else:
        print(f"Skipping file: {os.path.basename(filepath)} (Unknown extension).")

    return tensors


def main():
    """Main function to find data files and convert them to NPY files."""
    os.makedirs(OUTPUT_TENSOR_DIR, exist_ok=True)
    print("--- Data Preparator Service Started ---")
    print(f"Looking for data files in: {INPUT_PCAP_DIR}")
    
    all_tensors = []
    
    # We look for files grouped by class name 
    for class_label in ['HTTP', 'DNS', 'SSH']:
        # Search for files starting with the class name, regardless of extension
        search_pattern = os.path.join(INPUT_PCAP_DIR, f"{class_label}*.*")
        data_files = glob.glob(search_pattern)

        if not data_files:
            print(f"Warning: No data files found for class: {class_label}. Skipping.")
            continue
            
        class_tensors = []
        for data_file in data_files:
            tensors_from_file = process_pcap_to_tensors(data_file, class_label)
            class_tensors.extend(tensors_from_file)

        if class_tensors:
            # Convert the list of tensors to a single NumPy array
            final_array = np.array(class_tensors, dtype=np.uint8)
            output_filepath = os.path.join(OUTPUT_TENSOR_DIR, f"{class_label}_real.npy")
            
            # Save the final NumPy tensor file
            np.save(output_filepath, final_array)
            print(f"\nSuccessfully created and saved tensor file: {output_filepath}")
            print(f"Shape: {final_array.shape}\n")
            all_tensors.append(final_array)
            
    if not all_tensors:
        print("No tensors were generated. Please ensure data files are placed in the raw_pcaps directory.")
    else:
        total_samples = sum(a.shape[0] for a in all_tensors)
        print(f"--- Data Preparation Complete. Total tensors created: {total_samples} ---")

if __name__ == '__main__':
    main()
