# Use the same base image as the classifier (TensorFlow Python)
FROM tensorflow/tensorflow:2.15.0-gpu

WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the training script
COPY train.py .

# Command to run the training script (will be run manually by the user)
CMD ["python", "train.py"]