import os
from pathlib import Path

METIS_HOME_FOLDER = ".metisfl"
DRIVER_DIR_NAME = "driver"
LEARNER_DIR_NAME = "learner_{}"
CONTROLLER_DIR_NAME = "controller"
MODEL_SAVE_DIR_NAME = "model_definition"

DEFAULT_LEARNER_HOST = "[::]"
DEFAULT_LEARNER_PORT = 50052
DEFAULT_CONTROLLER_HOSTNAME = "[::]"
DEFAULT_CONTROLLER_PORT = 50051

# FIXME(@panoskyriakis): merge tasks_keys and taks

TRAIN = "train"
VALIDATION = "validation"
TEST = "test"
TASK_KEYS = [TRAIN, VALIDATION, TEST]

KERAS_NN_ENGINE = "keras_nn"
PYTORCH_NN_ENGINE = "pytorch_nn"
NN_ENGINES = [KERAS_NN_ENGINE, PYTORCH_NN_ENGINE]


DATASET_recipe_FILENAMES = {
    "train": "model_train_dataset_ops.pkl",
    "validation": "model_validation_dataset_ops.pkl",
    "test": "model_test_dataset_ops.pkl"
}

LEARNING_TASK = "learning"
EVALUATION_TASK = "evaluation"
INFERENCE_TASK = "inference"
TASKS = [LEARNING_TASK, EVALUATION_TASK, INFERENCE_TASK]

LEARNER_ID_FILE = "learner_id.txt"
AUTH_TOKEN_FILE = "auth_token.txt"

CANCEL_RUNNING_ON_SHUTDOWN = {
    LEARNING_TASK: True,
    EVALUATION_TASK: False,
    INFERENCE_TASK: False
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

def get_learner_id_fp(learner_id):
    learner_id_fp = get_learner_path(learner_id)
    _get_path_safe(learner_id_fp)
    return os.path.join(learner_id_fp, LEARNER_ID_FILE)

def get_auth_token_fp(learner_id):
    learnet_token_path = get_learner_path(learner_id)
    _get_path_safe(learnet_token_path)
    return os.path.join(learnet_token_path, AUTH_TOKEN_FILE)
    
def _get_path_safe(path: str) -> str:
    if not os.path.exists(path):
        os.makedirs(path)
    return path

