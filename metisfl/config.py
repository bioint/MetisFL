import os

from pathlib import Path
from shutil import which, copyfile

from metisfl.utils.metis_logger import MetisLogger

METIS_WORKING_DIR = ".metisfl"
DRIVER_DIR_NAME = "driver"
CONTROLLER_DIR_NAME = "controller"
LEARNER_DIR_NAME = "learner_{}"
MODEL_SAVE_DIR_NAME = "model_definition"
SSL_CONFIG_DIR_NAME = "resources/ssl_config"
GEN_CERTS_SCRIPT_NAME = "gen_certificates.sh"
DEFAULT_SSL_CONFIG_DIR = "resources/ssl_config/default"

CRYPTO_RESOURCES_DIR = "resources/fhe/cryptoparams/"

DEFAULT_CONTROLLER_HOSTNAME = "[::]"
DEFAULT_CONTROLLER_PORT = 50051
DEFAULT_LEARNER_HOST = "[::]"
DEFAULT_LEARNER_PORT = 50052

# FIXME(@panoskyriakis): merge tasks_keys and taks
TRAIN = "train"
VALIDATION = "validation"
TEST = "test"
TASK_KEYS = [TRAIN, VALIDATION, TEST]

KERAS_NN_ENGINE = "keras_nn"
PYTORCH_NN_ENGINE = "pytorch_nn"
NN_ENGINES = [KERAS_NN_ENGINE, PYTORCH_NN_ENGINE]

DATASET_RECIPE_FILENAMES = {
    "train": "model_train_dataset_ops.pkl",
    "validation": "model_validation_dataset_ops.pkl",
    "test": "model_test_dataset_ops.pkl"
}

# TRAIN = LEARNING
# VALIDATION = EVALUATION
# TEST = INFERENCE
# @stripeli @panoskyriakis verify this and change the names accordingly
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
    path = os.path.join(Path.home(), METIS_WORKING_DIR)
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

def get_crypto_resources_dir():
    path = get_project_home()
    # FIXME: finish this; must generate the crypto resources on first call

def get_auth_token_fp(learner_id):
    learnet_token_path = get_learner_path(learner_id)
    _get_path_safe(learnet_token_path)
    return os.path.join(learnet_token_path, AUTH_TOKEN_FILE)

def get_certificates_dir():
    path = get_project_home()
    path = os.path.join(path, SSL_CONFIG_DIR_NAME)
    return _get_path_safe(path)

def get_certificates():
    script_dir = os.path.dirname(__file__)
    gen_cert_script = os.path.join(
        script_dir, SSL_CONFIG_DIR_NAME, GEN_CERTS_SCRIPT_NAME)
    if which("openssl") is None:
        MetisLogger.warning("No openssl found in path. Using default certificates")
        return os.path.join(script_dir, DEFAULT_SSL_CONFIG_DIR)
    
    if os.path.exists(os.path.join(get_certificates_dir(), GEN_CERTS_SCRIPT_NAME)):
        return get_certificates_dir() # already generated
      
    # copy the script to the certificates dir
    copyfile(gen_cert_script, os.path.join(get_certificates_dir(), GEN_CERTS_SCRIPT_NAME))
    
    # generate the certificates
    os.chdir(get_certificates_dir())
    os.system("./{}".format(GEN_CERTS_SCRIPT_NAME))    

def _get_path_safe(path: str) -> str:
    if not os.path.exists(path):
        os.makedirs(path)
    return path
