# ДЗ1

## Барабанщиков Лев, Технопарк ML-21

## Python

Рекомендованная версия: 3.10

## Установка

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Запуск

### Train

```bash
python3 ml_project/train_pipeline.py
```

Также можно конфигурировать различные параметры. Например,

```bash
python3 ml_project/train_pipeline.py model="logistic_regression" 
```

позволяет сменить модель обучения на логистическую регрессию. А параметр metric

```bash
python3 ml_project/train_pipeline.py model="logistic_regression" metric="roc_auc"
```

позволит сменить так же и метрику.

### Predict

```bash
python3 ml_project/predict.py
```

Данная команда по умолчанию возьмет самую новую обученную модель из артефактов. Но это можно изменить:

```bash
python3 ml_project/predict.py predict.run_name="Имя нужной папки в папке outputs"
```

## Структура проекта

------------

    ├── README.md                   <- The top-level README.
    ├── data                        <- The datasets for modelling.
        └── raw                     <- .csv datasets
    ├── notebooks                   <- Jupyter notebooks (e.g. EDA)
    ├── requirements.txt            <- The requirements file for reproducing the analysis environment, e.g.
    │                                   generated with `pip freeze > requirements.txt`
    ├── ml_project                  <- Source code for use in this project.
    │   ├── configs                 <- Configuration files
    │   │   ├── common
    │   │   ├── dataset         
    │   │   ├── features   
    │   │   ├── metric    
    │   │   ├── mlflow
    │   │   ├── model
    │   │   ├── prediction
    │   │   └── config.yaml
    │   │
    │   ├── dataset                 <- Dataset class
    │   │   └── dataset.py
    │   │
    │   ├── mlflow_logger           <- MLflow logger class for mlflow support
    │   │   └── mlflow_logger.py
    │   │ 
    │   ├── preprocessing           <- Data preprocessing scripts
    │   │   └── transformer.py
    │   │
    │   ├── utils                   <- Helpful functions for pipelines
    │   │   └── tech_magic.py
    │   │
    │   ├── entities                <- Dataclasses for config validation
    │   │   ├── config.py
    │   │   ├── ds_params.py
    │   │   ├── feature_params.py
    │   │   ├── metric_params.py
    │   │   ├── mlflow_params.py
    │   │   ├── model_params.py
    │   │   └── predict_params.py
    │   │
    │   ├── models                   <- Helpful functions for pipelines implementing their logic
    │   │   ├── train_model.py
    │   │   └── predict_model.py
    │   │
    │   ├── train_pipeline.py       <- Train pipeline main script
    │   └── predict.py              <- Inference pipeline main script
    │
    ├── tests                       <- Tests for pipelines and functions
    ├── setup.cfg                   <- Store pytest configurations
    └── setup.py                    <- TODO: Makes project pip installable (pip install -e .) so src can be imported

--------