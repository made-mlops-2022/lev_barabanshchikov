import importlib
from pathlib import Path
from typing import Any, Union
import pickle

from ml_project.entities import Config


def load_pickle(model_path: str):
    with open(model_path, "rb") as fin:
        model = pickle.load(fin)
    return model


def load_obj(obj_path: str, default_obj_path: str = "") -> Any:
    obj_path_list = obj_path.rsplit(".", 1)
    obj_path = obj_path_list.pop(0) if len(obj_path_list) > 1 else default_obj_path
    obj_name = obj_path_list[0]
    module_obj = importlib.import_module(obj_path)
    if not hasattr(module_obj, obj_name):
        raise AttributeError(f"Object `{obj_name}` cannot be loaded from `{obj_path}`")
    return getattr(module_obj, obj_name)


def get_last_artifacts_path(cfg: Config) -> Union[Path, str]:
    project_dir = Path(cfg.common.project_dir)
    artifacts_dir = project_dir / cfg.common.artifacts_dir

    folders = sorted(
        list(
            filter(
                lambda p: "train_pipeline.log"
                in list(map(lambda x: x.name, p.iterdir())),
                artifacts_dir.iterdir(),
            )
        )
    )
    if folders:
        last_artifacts_path = artifacts_dir / folders[-1]
    else:
        last_artifacts_path = ""
    return last_artifacts_path
