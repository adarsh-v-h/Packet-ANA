import time
import random
from datetime import datetime

# Define a list of simulated network events
PROTOCOLS = ["HTTP", "DNS", "SSH", "P2P", "ICMP"]
SOURCES = [f"192.168.1.{i}" for i in range(10, 20)]
DESTINATIONS = ["external-web.com", "google-dns.net", "remote-server-01", "unknown-host.net"]

def generate_traffic_event():
    """Generates a log message simulating a network event."""
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    protocol = random.choice(PROTOCOLS)
    source_ip = random.choice(SOURCES)
    destination_host = random.choice(DESTINATIONS)
    
    if protocol == "HTTP":
        action = f"GET /api/v1/data"
    elif protocol == "DNS":
        action = f"Query for {destination_host}"
    elif protocol == "SSH":
        action = "Initiate secure connection"
    else:
        action = "Send payload"
        
    log_message = f"[{timestamp}] [NET-LOG] {source_ip} -> {destination_host} | Protocol: {protocol} | Action: {action}"
    
    return log_message

def main_traffic_loop():
    """The main loop that continuously prints simulated traffic."""
    
    print("--- Traffic Generator Service Starting ---")
    
    while True:
        log = generate_traffic_event()
        print(log)
        
        # Wait a short, random time before generating the next event
        time.sleep(random.uniform(0.1, 0.5))

if __name__ == "__main__":
    main_traffic_loop()
