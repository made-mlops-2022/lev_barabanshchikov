FROM python:3.10-slim

COPY requirements.txt /requirements.txt
RUN pip install --upgrade pip && \
    pip install -r /requirements.txt --no-cache-dir && \
    rm /requirements.txt