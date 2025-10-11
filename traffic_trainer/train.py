import os
import glob
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# --- Configuration ---
IMG_SIZE = 224
IMG_CHANNELS = 3
INPUT_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
TENSOR_FLATTENED_SIZE = IMG_SIZE * IMG_SIZE * IMG_CHANNELS # 150528

# The full set of classes the final model must predict
FINAL_CLASSES = ['HTTP', 'DNS', 'SSH', 'UNKNOWN', 'IMAGE'] 
NUM_CLASSES = len(FINAL_CLASSES)
MODEL_SAVE_PATH = '/app/data/model/mobilenet_v2_classifier.h5'
DATA_ROOT = '/app/data/processed_data'

# Training Parameters
EPOCHS = 5
BATCH_SIZE = 32

def build_model():
    """Builds the MobileNetV2 model structure for 5 classes."""
    print("Building MobileNetV2 base model...")
    # Load MobileNetV2 base, excluding the final classification layer
    base_model = MobileNetV2(
        input_shape=INPUT_SHAPE,
        include_top=False,
        weights='imagenet' # Use weights pre-trained on ImageNet
    )

    # Freeze the base model layers
    base_model.trainable = False

    # Add a custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)

    # Final model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def load_and_preprocess_data():
    """
    Loads NumPy tensor files from the processed_data directory and creates 
    synthetic 'UNKNOWN' samples to balance the dataset.
    """
    print("--- Loading and Preprocessing Data ---")
    data_tensors = []
    data_labels = []
    
    # 1. Load Benign Traffic (HTTP, DNS, SSH)
    print(f"Loading data from {DATA_ROOT}...")
    for i, class_name in enumerate(['HTTP', 'DNS', 'SSH']):
        class_dir = os.path.join(DATA_ROOT, class_name)
        
        # Check if the directory exists (it should if prepare.py ran)
        if not os.path.isdir(class_dir):
            print(f"Warning: Directory not found for class '{class_name}'. Skipping.")
            continue
            
        files = glob.glob(os.path.join(class_dir, '*.npy'))
        
        for file in files:
            try:
                # Load the flattened tensor
                tensor = np.load(file).astype(np.float32)
                
                # Check for correct size
                if tensor.size != TENSOR_FLATTENED_SIZE:
                    print(f"Skipping {file}: Incorrect size {tensor.size}")
                    continue
                
                # Reshape from (150528,) to (224, 224, 3)
                image_tensor = tensor.reshape(INPUT_SHAPE)
                
                # Apply MobileNetV2 preprocessing (scales pixel values from 0-255 to -1 to 1)
                preprocessed_tensor = preprocess_input(image_tensor)
                
                data_tensors.append(preprocessed_tensor)
                data_labels.append(class_name)
                
            except Exception as e:
                print(f"Error loading {file}: {e}")

    # 2. Generate Synthetic 'UNKNOWN' Samples
    # We will create an equal number of 'UNKNOWN' samples (random noise) 
    # to match the largest existing class count to address imbalance.
    
    # Count current samples
    unique_labels, counts = np.unique(data_labels, return_counts=True)
    label_counts = dict(zip(unique_labels, counts))
    max_count = max(counts) if counts.size > 0 else 100
    
    # 'UNKNOWN' samples are generated as random noise tensors
    unknown_count = max_count * 2 # Generate double the largest class for robust UNKNOWN detection
    
    print(f"Generating {unknown_count} synthetic 'UNKNOWN' (Random Noise) samples...")
    for _ in range(unknown_count):
        # Generate random bytes (0-255) for the UNKNOWN class
        random_bytes = np.random.randint(0, 256, size=TENSOR_FLATTENED_SIZE, dtype=np.uint8)
        
        # Reshape and preprocess like real data
        image_tensor = random_bytes.astype(np.float32).reshape(INPUT_SHAPE)
        preprocessed_tensor = preprocess_input(image_tensor)
        
        data_tensors.append(preprocessed_tensor)
        data_labels.append('UNKNOWN')

    # 3. Handle 'IMAGE' Class (Simple Placeholder for now)
    # Since we have no real training data for user-uploaded images, 
    # we will use some of the UNKNOWN noise to serve as a placeholder for the 
    # IMAGE class for initial training stability.
    image_placeholder_count = max_count // 2
    print(f"Using {image_placeholder_count} 'UNKNOWN' samples as 'IMAGE' class placeholders...")
    for i in range(image_placeholder_count):
        # Re-use UNKNOWN samples for IMAGE class placeholder
        data_tensors.append(data_tensors[unknown_count - 1 - i])
        data_labels.append('IMAGE')
    
    # 4. Final Conversion and Splitting
    X = np.array(data_tensors)
    
    # Map label names to one-hot encoded vectors based on FINAL_CLASSES order
    y_raw = np.array(data_labels)
    y = np.zeros((y_raw.size, NUM_CLASSES), dtype=np.float32)
    for i, class_name in enumerate(FINAL_CLASSES):
        # Set the column corresponding to the class name to 1.0
        y[y_raw == class_name, i] = 1.0
        
    print(f"Total samples loaded/generated: {X.shape[0]}")
    
    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Validation set size: {X_val.shape[0]}")
    
    return X_train, X_val, y_train, y_val


def main_training():
    """Main function to run the model training."""
    
    # 1. Load Data
    X_train, X_val, y_train, y_val = load_and_preprocess_data()
    
    # Only proceed if we have training data
    if X_train.shape[0] == 0:
        print("FATAL: No training data available. Cannot proceed.")
        return

    # 2. Build Model
    model = build_model()
    model.summary()

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
                # Restrict TensorFlow to only use the first GPU
                tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
                # Enable memory growth (avoids allocating all GPU memory at once)
                tf.config.experimental.set_memory_growth(gpus[0], True)
                print(f"Using GPU: {gpus[0].name}")
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)
        else:
            print("No GPU found. Using CPU for training.")
            
    except Exception as e:
        print(f"TensorFlow configuration error: {e}")

    main_training()
