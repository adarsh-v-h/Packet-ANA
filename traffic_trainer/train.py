import os
import glob
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.utils import to_categorical

# --- Configuration ---
IMG_SIZE = 224
IMG_CHANNELS = 3
INPUT_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
TENSOR_FLATTENED_SIZE = IMG_SIZE * IMG_SIZE * IMG_CHANNELS # 150528

# The full set of classes the final model must predict
FINAL_CLASSES = ['HTTP', 'DNS', 'SSH', 'UNKNOWN', 'IMAGE'] 
NUM_CLASSES = len(FINAL_CLASSES)
CLASS_TO_INDEX = {cls: i for i, cls in enumerate(FINAL_CLASSES)}

MODEL_SAVE_PATH = '/app/data/model/mobilenet_v2_classifier.h5'
DATA_ROOT = '/app/data/processed_data' # Still reference for future use, but not strictly needed now

# Training Parameters
EPOCHS = 10  
BATCH_SIZE = 32

# Data Augmentation Parameters
# This ensures we get a high number of training samples for robust learning
AUGMENTATION_FACTOR = 50 
NOISE_LEVEL = 0.05       # 5% of pixels will be randomly altered for augmentation
SAMPLES_PER_REAL_CLASS = 100 # Base number of simulated samples for HTTP/DNS/SSH

# --- Synthetic Data Definitions based on Packet Agent ---
TRAFFIC_PATTERNS = {
    # Structured protocols (will look like distinct visual patterns)
    "HTTP": b"GET /index.html HTTP/1.1\r\nHost: example.com\r\nUser-Agent: Simulated-Client/1.0\r\nConnection: Close\r\n\r\n",
    "DNS": b"\xaa\xaa\x01\x00\x00\x01\x00\x00\x00\x00\x00\x00\x03www\x06google\x03com\x00\x00\x01\x00\x00",
    "SSH": b"SSH-2.0-OpenSSH_7.4p1 Debian-10+deb9u3\r\n" + b"\xAB" * 1500,
}


def process_raw_to_tensor(raw_bytes):
    """
    Converts raw bytes to a 1D tensor of fixed size (150528 elements).
    Pads with zeros or truncates to fit the required size.
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


def augment_tensor(tensor):
    """
    Applies noise injection to a single tensor.
    """
    # 1. Noise Injection
    mask = np.random.choice([0, 1], size=tensor.shape, p=[1 - NOISE_LEVEL, NOISE_LEVEL])
    noise = np.random.randint(0, 256, size=tensor.shape, dtype=np.uint8)
    augmented_tensor = np.where(mask == 1, noise, tensor)
    
    return augmented_tensor.astype(np.float32)

def load_and_augment_data():
    """
    Generates synthetic structured data (HTTP, DNS, SSH) and unstructured data 
    (UNKNOWN, IMAGE), then applies augmentation.
    """
    all_data = []
    all_labels = []

    print("--- Generating Structured Data (HTTP, DNS, SSH) ---")
    
    # 1. Generate and Augment Structured Data
    for class_name, raw_pattern in TRAFFIC_PATTERNS.items():
        base_tensor = process_raw_to_tensor(raw_pattern)
        
        # Generate the base sample and augmented versions
        for i in range(SAMPLES_PER_REAL_CLASS * AUGMENTATION_FACTOR):
            # The first N samples use the base pattern exactly (no augmentation)
            if i < SAMPLES_PER_REAL_CLASS:
                tensor_to_add = base_tensor
            else:
                # Apply augmentation (noise) to the base tensor
                tensor_to_add = augment_tensor(base_tensor)
            
            all_data.append(tensor_to_add)
            all_labels.append(CLASS_TO_INDEX[class_name])

        print(f"Generated and augmented data for {class_name}. Total samples: {SAMPLES_PER_REAL_CLASS * AUGMENTATION_FACTOR}")


    # --- 2. Generate Unstructured/Noise Data (UNKNOWN/IMAGE) ---
    
    # We generate a large number of pure random noise samples 
    SAMPLES_PER_SYNTHETIC_CLASS = 2500 * AUGMENTATION_FACTOR 
    
    print(f"Generating {SAMPLES_PER_SYNTHETIC_CLASS} synthetic UNKNOWN/IMAGE samples...")
    
    # UNKNOWN class: Pure random noise
    unknown_index = CLASS_TO_INDEX['UNKNOWN']
    for _ in range(SAMPLES_PER_SYNTHETIC_CLASS):
        # Generate a tensor of random bytes
        noise_tensor = np.random.randint(0, 256, size=(TENSOR_FLATTENED_SIZE,), dtype=np.uint8)
        all_data.append(noise_tensor)
        all_labels.append(unknown_index)
        
    # IMAGE class: Another set of random noise
    image_index = CLASS_TO_INDEX['IMAGE']
    for _ in range(SAMPLES_PER_SYNTHETIC_CLASS):
        # Generate another noise tensor for the IMAGE class
        noise_tensor = np.random.randint(0, 256, size=(TENSOR_FLATTENED_SIZE,), dtype=np.uint8)
        all_data.append(noise_tensor)
        all_labels.append(image_index)


    # --- 3. Final Preprocessing and Split ---
    X = np.array(all_data).astype(np.float32)
    y = np.array(all_labels)

    # Reshape the flattened data to the MobileNetV2 input shape (N, 224, 224, 3)
    X = X.reshape(-1, IMG_SIZE, IMG_SIZE, IMG_CHANNELS)
    
    # Apply MobileNetV2-specific pre-processing (scales pixels between -1 and 1)
    X = preprocess_input(X)
    
    # Convert labels to one-hot encoding
    y = to_categorical(y, num_classes=NUM_CLASSES)

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"\nTotal Dataset Size (after augmentation): {X.shape[0]}")
    print(f"Training Samples: {X_train.shape[0]}")
    print(f"Validation Samples: {X_val.shape[0]}")
    
    return X_train, X_val, y_train, y_val


def build_model():
    """Builds the MobileNetV2 model structure for 5 classes."""
    print("Building MobileNetV2 base model...")
    # Load MobileNetV2 base, excluding the final classification layer
    base_model = MobileNetV2(
        input_shape=INPUT_SHAPE,
        include_top=False,
        weights='imagenet' 
    )

    # Freeze the base model layers (Transfer Learning)
    base_model.trainable = False

    # Add a custom classifier head for our 5 classes
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

def main():
    """Main training function."""
    
    # 1. Load Data
    X_train, X_val, y_train, y_val = load_and_augment_data()
    
    if X_train.shape[0] == 0:
        print("Error: No training data available. Cannot proceed.")
        return

    # 2. Build Model
    model = build_model()
    # model.summary()

    # 3. Train Model
    print("--- Starting Model Training ---")
    
    model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        verbose=1
    )

    # 4. Save Model
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    model.save(MODEL_SAVE_PATH)
    print(f"\n--- Training Complete ---")
    print(f"Model saved to: {MODEL_SAVE_PATH}")


if __name__ == '__main__':
    # Ensure CPU/GPU usage is correctly set up for training
    try:
        # Check for GPU devices
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
                tf.config.experimental.set_memory_growth(gpus[0], True)
                print(f"Using GPU: {gpus[0].name}")
            except RuntimeError as e:
                print(f"Runtime error during GPU setup: {e}")
        else:
            print("Using CPU for training.")
    except Exception as e:
        print(f"Error during TensorFlow device setup: {e}")

    main()
