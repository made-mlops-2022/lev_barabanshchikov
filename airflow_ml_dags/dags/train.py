from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.operators.empty import EmptyOperator
from airflow.sensors.filesystem import FileSensor
from airflow.utils.dates import days_ago

from config import DEFAULT_ARGS, DATA_VOLUME, ARTIFACTS_VOLUME, MLFLOW_PARAMS

with DAG(
        dag_id="train",
        description="Train and validate the model",
        schedule_interval="@weekly",
        start_date=days_ago(3),
        default_args=DEFAULT_ARGS
) as dag:
    start = EmptyOperator(task_id="start_training")

    features_sensor = FileSensor(
        task_id="features-sensor",
        poke_interval=10,
        retries=100,
        filepath="data/raw/{{ ds }}/data.csv"
    )

    target_sensor = FileSensor(
        task_id="target-sensor",
        poke_interval=10,
        retries=100,
        filepath="data/raw/{{ ds }}/target.csv"
    )

    preprocess_data = DockerOperator(
        task_id="preprocess",
        image="airflow-preprocess",
        command=""  # TODO: from here to 52 pts
    )

    split_data = DockerOperator(
        task_id="split-data",
        image="airflow-split-data",
        command="--input_dir /data/processed/{{ ds }} --output_dir /data/split/{{ ds }}",
        network_mode="bridge",
        do_xcom_push=False,
        volumes=[DATA_VOLUME]
    )
