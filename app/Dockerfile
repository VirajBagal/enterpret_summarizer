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

# download weights in cache
COPY download_snapshot.py download_snapshot.py
RUN python3 download_snapshot.py

COPY inference_utils.py inference_utils.py
COPY app_v2.py app_v2.py

CMD ["uvicorn", "app_v2:app", "--host", "0.0.0.0"]

