from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sensors.filesystem import FileSensor
from airflow.utils.dates import days_ago

from config import DEFAULT_ARGS, MLFLOW_PARAMS, DATA_MOUNT, ARTIFACTS_MOUNT

with DAG(
        dag_id="predict",
        description="Prediction model",
        schedule_interval="@daily",
        start_date=days_ago(3),
        default_args=DEFAULT_ARGS
) as dag:
    start = EmptyOperator(task_id="start")

    features_sensor = FileSensor(
        task_id="features-sensor",
        poke_interval=10,
        retries=5,
        filepath="data/raw/{{ ds }}/data.csv"
    )

    predict = DockerOperator(
        task_id="predict",
        image="airflow-predict",
        command="--input_dir /data/raw/{{ ds }} --output_dir /data/predictions/{{ ds }} "
                "--transformers_dir /data/transformers",
        network_mode="host",
        do_xcom_push=False,
        private_environment=MLFLOW_PARAMS,
        mount_tmp_dir=False,
        mounts=[DATA_MOUNT, ARTIFACTS_MOUNT]
    )

    end = EmptyOperator(task_id="end")

    start >> features_sensor >> predict >> end
