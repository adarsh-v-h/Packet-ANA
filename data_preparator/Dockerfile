# Dockerfile for the Data Preparator Service
# This service is responsible for ingesting raw PCAP data and converting it 
# into the 224x224x3 image tensor format required for model training.

FROM python:3.9-slim

WORKDIR /app

# Install necessary libraries: 
# scapy for packet analysis
# numpy for numerical processing
# tensorflow is added here in case we eventually run training logic
RUN pip install --no-cache-dir \
    scapy \
    numpy \
    tensorflow==2.10.0  # Use a stable version for data processing

# Copy the data preparation script into the container
COPY prepare.py .

# Command to run the preparation script
# This service is designed to run its task and then exit (not run indefinitely like the others)
CMD ["python", "prepare.py"]