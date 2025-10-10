FROM python:3.10-slim

WORKDIR /app

# Install dependencies: Flask and requests
RUN pip install flask requests

# Copy the frontend application files: app.py (the server) and index.html (the dashboard)
COPY app.py index.html ./

# Expose the port (8080)
EXPOSE 8080

# Start the Flask app
CMD ["python", "app.py"]

