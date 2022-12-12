FROM python:3.10-slim

WORKDIR /home/app
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY utils.py .
COPY validation.py .
COPY app.py .

ENV MODEL_URL="https://drive.google.com/drive/folders/1ItBIA7whRu3G9FHn3VObTdLNiCA-xy_V?usp=share_link"

CMD ["python3", "app.py"]