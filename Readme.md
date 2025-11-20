# 🧠 Network Traffic Classifier Simulation (MobileNetV2 via Docker Compose)

A proof-of-concept system that treats raw network packet bytes as image data and classifies them using a deep-learning model inside a fully containerized environment. The architecture demonstrates how network traffic can be reinterpreted as pixel tensors and processed through MobileNetV2 for protocol identification.

---

## 📘 Overview

The core idea is simple but ambitious:

1. Capture or simulate packet bytes.  
2. Reshape them into a `224×224×3` tensor.  
3. Normalize, preprocess, and send them to a ML inference server.  
4. Receive a classification: **HTTP**, **DNS**, **SSH**, or **UNKNOWN**.

Three microservices (all Dockerized) work together to create a full simulation of network activity → tensor generation → deep-learning inference.

---

## ⚙️ Architecture

The system is composed of **three Docker containers** communicating over an internal network:

### 🛰️ `traffic_gen` — Simulated Environment
- Generates continuous fake network activity logs.
- Represents ambient network traffic.

### 🧩 `packet_agent` — Packet → Tensor Processor
- Simulates a packet capture agent.
- Converts raw bytes into the exact tensor size MobileNetV2 requires:  
  `224 × 224 × 3 = 150,528` values.
- Normalizes bytes from `[0–255] → [0–1]`.
- Sends the tensor to the classifier via HTTP POST.

### 🧠 `classifier` — ML Inference API
- Flask + TensorFlow/Keras server.
- Loads MobileNetV2 with a custom classification head.
- Receives tensor → reshapes → runs inference → returns predicted class.

---

## 📂 Project Structure
.
├── docker-compose.yml
├── mobile_net_classifier/
│ ├── Dockerfile
│ └── app.py
├── packet_agent/
│ ├── Dockerfile
│ └── agent.py
└── traffic_gen/
├── Dockerfile
└── traffic_gen.py


---

## 🚀 Getting Started

### 1️⃣ Build the project
```bash
docker compose build
```
**Run all services**
```bash
docker compose up -d
```
**View logs**
```bash
docker compose logs -f
```
**Stop everything**
```bash
docker compose down
```
