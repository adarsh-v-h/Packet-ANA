FROM python:3.10-slim

WORKDIR /app

# Install dependencies:
# - flask and requests for the API functionality.
# - Pillow (PIL) for image decoding and resizing (required for the new upload feature).
# - numpy for tensor processing.
RUN pip install flask requests Pillow numpy

# Copy the application files
COPY agent.py .

# Expose the API port (5001)
EXPOSE 5001

# Start the Flask API server
CMD ["python", "agent.py"]