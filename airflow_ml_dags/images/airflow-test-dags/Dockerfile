ARG AIRFLOW_BASE_IMAGE="apache/airflow:2.5.0-python3.10"
FROM ${AIRFLOW_BASE_IMAGE}

WORKDIR /testing

RUN pip install --user --no-cache-dir \
    apache-airflow-providers-docker==3.3.0

COPY requirements.txt /requirements.txt
RUN pip install --upgrade pip && \
    pip install -r /requirements.txt --no-cache-dir