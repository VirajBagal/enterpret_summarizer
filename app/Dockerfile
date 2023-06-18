# Base image with Python
FROM python:3.9-slim

# Install g++
RUN apt-get update && apt-get install build-essential -y && apt-get install manpages-dev -y

# Set the working directory
WORKDIR /app

# Copy only the requirements file
COPY requirements.txt requirements.txt

# Install Python dependencies, if any
RUN pip install -r requirements.txt

# create cache directory
WORKDIR /root
RUN mkdir -p .cache/huggingface/hub
COPY models--google--flan-t5-large .cache/huggingface/hub/models--google--flan-t5-large

WORKDIR /app

COPY lora_weights lora_weights
COPY inference_utils.py inference_utils.py
COPY app.py app.py

CMD ["uvicorn", "app:app", "--host", "0.0.0.0"]

