FROM python
RUN mkdir /app
ADD . /app
WORKDIR /app
# Installing opencv dependencies for docker
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y
# Installing Google Tesseract-OCR package
RUN apt-get install tesseract-ocr tesseract-ocr-ita -y
# Upgrading pip
RUN python -m pip install --upgrade pip
# Installing python requirements
RUN pip install -r requirements.txt
ENTRYPOINT ["python", "main.py"]