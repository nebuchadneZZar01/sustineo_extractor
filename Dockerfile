FROM python:3.10
ARG DEBIAN_FRONTEND=noninteractive
WORKDIR /app
ADD . /app

# Installing apt-utils to avoid further errors
RUN apt-get update && apt-get install apt-utils

# Installing opencv dependencies for docker
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y

# Installing Google Tesseract-OCR package
RUN apt-get install tesseract-ocr tesseract-ocr-ita -y

# Upgrading pip
RUN python -m pip install --upgrade pip

# Installing python requirements
RUN pip install -r requirements.txt

ENTRYPOINT ["python", "main.py"]