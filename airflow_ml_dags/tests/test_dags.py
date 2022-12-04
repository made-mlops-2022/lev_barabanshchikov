import sys

import pytest
from airflow.models import DagBag

sys.path.append("dags/")


@pytest.fixture()
def dagbag():
    return DagBag(dag_folder="/testing/dags", include_examples=False)


def test_get_data_loaded(dagbag):
    dag = dagbag.get_dag(dag_id="get_data")
    assert dagbag.import_errors == {}
    assert dag is not None
    assert len(dag.tasks) == 3


def test_get_data_structure(dagbag):
    dag = dagbag.get_dag(dag_id="get_data")
    structure = {
        "downloading_started": ["airflow-get-data"],
        "airflow-get-data": ["downloading-completed"],
        "downloading-completed": [],
    }
    for name, task in dag.task_dict.items():
        assert set(structure[name]) == task.downstream_task_ids


def test_train_loaded(dagbag):
    dag = dagbag.get_dag(dag_id="train")
    assert dagbag.import_errors == {}
    assert dag is not None
    assert len(dag.tasks) == 8


def test_train_structure(dagbag):
    dag = dagbag.get_dag(dag_id="train")
    structure = {
        "start-training": ["features-sensor", "target-sensor"],
        "sensor-features": ["preprocess"],
        "sensor-targets": ["preprocess"],
        "preprocess": ["split-data"],
        "split-data": ["train"],
        "train": ["validate"],
        "validate": ["end-training"],
        "end-training": [],
    }
    for name, task in dag.task_dict.items():
        assert set(structure[name]) == task.downstream_task_ids


def test_predict_loaded(dagbag):
    dag = dagbag.get_dag(dag_id="predict")
    assert dagbag.import_errors == {}
    assert dag is not None
    assert len(dag.tasks) == 4


def test_predict_structure(dagbag):
    dag = dagbag.get_dag(dag_id="predict")
    structure = {
        "start": ["features-sensor"],
        "features-sensor": ["predict"],
        "predict": ["end"],
        "end": [],
    }
    for name, task in dag.task_dict.items():
        assert set(structure[name]) == task.downstream_task_ids
