from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.operators.empty import EmptyOperator
from airflow.sensors.filesystem import FileSensor
from airflow.utils.dates import days_ago

from config import DEFAULT_ARGS, DATA_MOUNT, ARTIFACTS_MOUNT, MLFLOW_PARAMS

with DAG(
        dag_id="train",
        description="Train and validate the model",
        schedule_interval="@weekly",
        start_date=days_ago(3),
        default_args=DEFAULT_ARGS
) as dag:
    start = EmptyOperator(task_id="start-training")

    features_sensor = FileSensor(
        task_id="features-sensor",
        poke_interval=10,
        retries=5,
        filepath="data/raw/{{ ds }}/data.csv"
    )

    target_sensor = FileSensor(
        task_id="target-sensor",
        poke_interval=10,
        retries=5,
        filepath="data/raw/{{ ds }}/target.csv"
    )

    preprocess_data = DockerOperator(
        task_id="preprocess",
        image="airflow-preprocess",
        command="--input_dir /data/raw/{{ ds }} --output_dir /data/processed/{{ ds }} "
                "--save_transformer_dir /data/transformers/{{ ds }}",
        network_mode="bridge",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[DATA_MOUNT]
    )

    split_data = DockerOperator(
        task_id="split-data",
        image="airflow-split-data",
        command="--input_dir /data/processed/{{ ds }} --output_dir /data/split/{{ ds }}",
        network_mode="bridge",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[DATA_MOUNT]
    )

    train = DockerOperator(
        task_id="train",
        image="airflow-train",
        command="--data_dir /data/split/{{ ds }} --save_model_dir /data/models/{{ ds }}",
        network_mode="host",
        do_xcom_push=False,
        private_environment=MLFLOW_PARAMS,
        mount_tmp_dir=False,
        mounts=[DATA_MOUNT, ARTIFACTS_MOUNT]
    )

    validate = DockerOperator(
        task_id="validate",
        image="airflow-validate",
        command="--data_dir /data/split/{{ ds }} --metrics_dir /data/metrics/{{ ds }}",
        network_mode="host",
        do_xcom_push=False,
        private_environment=MLFLOW_PARAMS,
        mount_tmp_dir=False,
        mounts=[DATA_MOUNT, ARTIFACTS_MOUNT]
    )

    end = EmptyOperator(task_id="end-training")

    start >> [features_sensor, target_sensor] >> preprocess_data >> split_data >> train >> validate >> end
