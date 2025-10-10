FROM python:3.10-slim

WORKDIR /app

# Install dependencies: Flask, TensorFlow, and NumPy
# NOTE: TensorFlow is a large dependency.
RUN pip install flask numpy tensorflow tensorflow-hub

# Copy the application file
COPY app.py .

# Install the ImageNet weights during the build process
# This prevents runtime errors due to failed downloads and makes the service start faster.
# This RUN command executes a Python script that forces the Keras download.
RUN python -c "import tensorflow as tf; \
    from tensorflow.keras.applications import MobileNetV2; \
    MobileNetV2(weights='imagenet'); \
    print('ImageNet weights downloaded and cached successfully.')"

# Expose the API port (5000)
EXPOSE 5000

# Start the Flask application
CMD ["python", "app.py"]