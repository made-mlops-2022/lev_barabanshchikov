from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.dates import days_ago

from config import DEFAULT_ARGS, DATA_VOLUME

with DAG(
        dag_id="generate_data",
        description="Generate training data",
        schedule_interval="@daily",
        start_date=days_ago(3),
        default_args=DEFAULT_ARGS
) as dag:
    start = EmptyOperator(task_id="downloading_started")

    generate = DockerOperator(
        task_id="airflow-get-data",
        image="airflow-get-data",
        command="get_data.py /data/raw/{{ ds }}",
        do_xcom_push=False,
        network_mode="bridge",
        auto_remove="True",
        volumes=[DATA_VOLUME]
    )

    end = EmptyOperator(task_id="downloading_completed")

    start >> generate >> end
