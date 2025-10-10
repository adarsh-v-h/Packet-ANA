from flask import Flask, send_file, jsonify, request
import requests
import os

app = Flask(__name__)
# Internal Docker network service address for the packet-agent
PACKET_AGENT_RANDOM_URL = "http://packet-agent:5001/api/classify_random"
PACKET_AGENT_UPLOAD_URL = "http://packet-agent:5001/api/classify_upload"

# 1. Serve the main HTML page
@app.route('/')
def index():
    return send_file('index.html')

# 2. Proxy endpoint for the browser to call the packet-agent (Simulation)
@app.route('/classify_proxy', methods=['GET'])
def classify_proxy():
    try:
        response = requests.get(PACKET_AGENT_RANDOM_URL, timeout=15)
        response.raise_for_status() 
        return jsonify(response.json())
        
    except requests.exceptions.RequestException as e:
        return jsonify({
            "status": "error", 
            "error": "Backend (Packet Agent) is unreachable for simulation. Check Docker logs."
        }), 500

# 3. Proxy endpoint for the browser to call the packet-agent (Upload)
@app.route('/classify_upload', methods=['POST'])
def classify_upload_proxy():
    try:
        # Forward the JSON data (which contains the Base64 image) from the browser to the agent
        data = request.get_json()
        response = requests.post(PACKET_AGENT_UPLOAD_URL, json=data, timeout=20)
        response.raise_for_status() 
        return jsonify(response.json())
        
    except requests.exceptions.RequestException as e:
        return jsonify({
            "status": "error", 
            "error": f"Backend (Packet Agent) failed during upload processing: {str(e)}"
        }), 500


if __name__ == '__main__':
    # Run on port 8080 as defined in docker-compose.yml
    app.run(host='0.0.0.0', port=8080)