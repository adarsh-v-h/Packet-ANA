Network Traffic Classifier Simulation (MobileNetV2 via Docker Compose)
This project demonstrates a proof-of-concept architecture for classifying network traffic using a deep learning model (MobileNetV2) integrated within a containerized environment using Docker Compose.

The core idea is to treat the raw byte data of a network packet as pixel data, convert it into a fixed-size "image" tensor (224×224×3), and feed it to a pre-trained image classification model for protocol identification.

🚀 Architecture Overview
The system is composed of three distinct services running in separate Docker containers, communicating over the internal Docker network:

traffic_gen (Simulated Environment)

Role: Simulates the continuous presence of network activity, logging arbitrary traffic events to standard output.

Output: Generates log messages to show activity.

packet_agent (Data Processor & Client)

Role: Simulates packet capture, performs the crucial Byte-to-Image Tensor Conversion, and acts as a client to the Classifier API.

Logic: Truncates/pads raw packet data to 150,528 bytes (224×224×3), normalizes it to 0−1, and sends the flattened list via a POST request to the classifier service.

classifier (Inference Server)

Role: Hosts the Keras/MobileNetV2 model and provides a REST API for inference.

Logic: Receives the 150,528-element tensor, reshapes it to (1,224,224,3), runs the prediction, and returns the predicted class (HTTP, DNS, SSH, or UNKNOWN).

🛠️ Prerequisites
To run this project, you must have the following installed:

Docker: (Version 20.10 or later)

Docker Compose: (Usually included in Docker Desktop)

📂 Project Structure
The files are organized into three service directories and the root configuration file:

.
├── docker-compose.yml
├── mobile_net_classifier/
│   ├── Dockerfile
│   └── app.py            # Flask API, loads MobileNetV2
├── packet_agent/
│   ├── Dockerfile
│   └── agent.py          # Byte-to-Tensor conversion, HTTP client
└── traffic_gen/
    ├── Dockerfile
    └── traffic_gen.py    # Generates simulated network log messages

🚀 Getting Started
Follow these steps to build and run the entire simulation.

1. Build the Docker Images
From the root directory containing docker-compose.yml, run the build command. This will download the base images and install all Python dependencies (including TensorFlow in the classifier service).

docker compose build

2. Run the Services
Start all three services simultaneously in detached mode (-d).

docker compose up -d

3. Observe the Classification
To see the entire pipeline working, stream the logs from all three services. The packet_agent service will show the results of its communication with the classifier.

docker compose logs -f

You will see three streams of output:

traffic_gen: Simple logs simulating network activity.

classifier: Logs indicating that the model has been initialized and is receiving requests.

packet_agent: This is the most important output. It shows the simulated traffic type, the tensor size, the request being sent, and the final classification result (e.g., Predicted Class: DNS).

Expected packet_agent output snippet:

[AGENT] Sending simulated HTTP packet data (150528 values) to classifier...
Classification Result for simulated HTTP:
  -> Predicted Class: HTTP (Confidence: 0.9812)
  -> (Simulation Match: ✅ Correct)

4. Stop and Clean Up
When you are finished, stop the running containers and remove the network:

docker compose down

📝 Key Components
Classifier Model (mobile_net_classifier/app.py)
The classifier uses a pre-trained MobileNetV2 base model, commonly used in computer vision, but adapted for 4 specific classes (HTTP, DNS, SSH, UNKNOWN).

It's important to note that without actual training data (packet captures labeled as images), the model uses random, untrained weights for the final classification layer. The simulated accuracy in the logs is based on the model's random initial guesses.

The preprocess_input function is crucial, as it scales the 0−1 normalized tensor into the required MobileNetV2 input range of [−1,1].

Packet-to-Tensor Conversion (packet_agent/agent.py)
This function is the heart of the "Traffic as Image" concept:

Fixed Size: Raw packet bytes are padded or truncated to exactly 150,528 bytes.

Normalization: Bytes (0−255) are converted to floating-point numbers and normalized to the 0−1 range before being sent to the classifier.