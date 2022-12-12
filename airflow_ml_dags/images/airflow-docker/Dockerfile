ARG AIRFLOW_BASE_IMAGE="apache/airflow:2.5.0-python3.10"
FROM ${AIRFLOW_BASE_IMAGE}

RUN pip install --user --no-cache-dir \
    apache-airflow-providers-docker==3.3.0

USER root
ENV PYTHONPATH=/home/airflow/.local/lib/python3.10/site-packages