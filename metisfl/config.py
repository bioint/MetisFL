import os
from pathlib import Path

METIS_HOME_FOLDER = ".metisfl"
DRIVER_DIR_NAME = "driver"
LEARNER_DIR_NAME = "learner_{}"
CONTROLLER_DIR_NAME = "controller"
MODEL_SAVE_DIR_NAME = "model_definition"

TRAIN = "train"
VALIDATION = "validation"
TEST = "test"
TASK_KEYS = [TRAIN, VALIDATION, TEST]


DATASET_RECEIPE_FILENAMES = {
    "train": "model_train_dataset_ops.pkl",
    "validation": "model_validation_dataset_ops.pkl",
    "test": "model_test_dataset_ops.pkl"
}


def get_project_home() -> str:
    path = os.path.join(Path.home(), METIS_HOME_FOLDER)
    return _get_path_safe(path)

def get_driver_path() -> str:
    path = get_project_home()
    path = os.path.join(path, DRIVER_DIR_NAME)
    return _get_path_safe(path)

def get_controller_path():
    path = get_project_home()
    path = os.path.join(path, "controller")
    return _get_path_safe(path)

def get_learner_path(learner_id: int):
    path = get_project_home()
    path = os.path.join(path, "learner_{}".format(learner_id))
    return _get_path_safe(path)

def get_driver_model_save_dir() -> str:
    path = get_driver_path()
    path = os.path.join(path, MODEL_SAVE_DIR_NAME)
    return _get_path_safe(path)


def _get_path_safe(path: str) -> str:
    if not os.path.exists(path):
        os.makedirs(path)
    return path
