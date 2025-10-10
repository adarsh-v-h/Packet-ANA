# Use a lightweight Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy the traffic generator script
COPY traffic_gen.py .

# Command to run the generator script continuously
CMD ["python", "traffic_gen.py"]