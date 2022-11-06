import logging
import mlflow


class MlflowLogger:
    def __init__(self, exp_name: str, run_name: str, uri: str):
        mlflow.warnings.filterwarnings("ignore")
        mlflow.set_tracking_uri(uri)
        self.mlflow_run = None
        mlflow.set_experiment(exp_name)
        experiment = mlflow.get_experiment_by_name(exp_name)
        mlflow.start_run(run_name=run_name, experiment_id=experiment.experiment_id)
        self.mlflow_run = mlflow.active_run()

        if self.mlflow_run:
            self.log_dict, self.current_epoch = {}, 0
            logging.info(f"[Mlflow] logging has been initiated")
        else:
            logging.warning(f"[Mlflow] MLflow is not running!")

    def log_artifact(self, *args, **kwargs):
        if self.mlflow_run:
            mlflow.log_artifact(*args, **kwargs)

    def log_artifacts(self, *args, **kwargs):
        if self.mlflow_run:
            mlflow.log_artifacts(*args, **kwargs)

    def end_run(self):
        if self.mlflow_run:
            mlflow.end_run()

    def log_metric(self, *args, **kwargs):
        if self.mlflow_run:
            mlflow.log_metric(*args, **kwargs)

    def log_metrics(self, *args, **kwargs):
        if self.mlflow_run:
            mlflow.log_metrics(*args, **kwargs)

    def log_param(self, *args, **kwargs):
        if self.mlflow_run:
            mlflow.log_param(*args, **kwargs)

    def log_params(self, *args, **kwargs):
        if self.mlflow_run:
            mlflow.log_params(*args, **kwargs)

    def log_sklearn_model(self, *args, **kwargs):
        if self.mlflow_run:
            mlflow.sklearn.log_model(*args, **kwargs)
