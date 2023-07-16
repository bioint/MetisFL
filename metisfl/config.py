import os

from pathlib import Path
from shutil import which, copyfile


METIS_WORKING_DIR = ".metisfl"
DRIVER_DIR_NAME = "driver"
CONTROLLER_DIR_NAME = "controller"
LEARNER_DIR_NAME = "learner_{}"
MODEL_SAVE_DIR_NAME = "model_definition"
SSL_PATH = "resources/ssl"
GEN_CERTS_SCRIPT_NAME = "generate.sh"
GEN_CERTS_SCRIPT_NAME_DIR = os.path.join(SSL_PATH, GEN_CERTS_SCRIPT_NAME)
SERVER_CERT_NAME = "server-cert.pem"
SERVER_KEY_NAME = "server-key.pem"
SERVER_CERT_DIR = os.path.join(SSL_PATH, SERVER_CERT_NAME)
SERVER_KEY_DIR = os.path.join(SSL_PATH, SERVER_KEY_NAME)

FHE_RESOURCE_DIR = "resources/fhe"
FHE_CRYPTO_CONTEXT_FILE = "cryptocontext.txt"
FHE_KEY_PUBLIC = "key-public.txt"
FHE_KEY_PRIVATE = "key-private.txt"
FHE_KEY_EVAL_MULT = "key-eval-mult.txt"

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

def get_fhe_dir():
    path = get_project_home()
    path = os.path.join(path, FHE_RESOURCE_DIR)
    return _get_path_safe(path)

def get_fhe_resources():
    path = get_fhe_dir()
    fhe_crypto_context_file = os.path.join(path, FHE_CRYPTO_CONTEXT_FILE)
    fhe_key_public_file = os.path.join(path, FHE_KEY_PUBLIC)
    fhe_key_private_file = os.path.join(path, FHE_KEY_PRIVATE)
    fhe_key_eval_mult_file = os.path.join(path, FHE_KEY_EVAL_MULT)
    return fhe_crypto_context_file, fhe_key_public_file, fhe_key_private_file, fhe_key_eval_mult_file

def get_auth_token_fp(learner_id):
    learnet_token_path = get_learner_path(learner_id)
    _get_path_safe(learnet_token_path)
    return os.path.join(learnet_token_path, AUTH_TOKEN_FILE)

def get_default_certificates():
    script_dir = os.path.dirname(__file__)
    original_certs_path = os.path.join(script_dir, SSL_PATH)
    server_cert = os.path.join(original_certs_path, SERVER_CERT_NAME)
    server_key = os.path.join(original_certs_path, SERVER_KEY_NAME)
    return server_cert, server_key
    
def get_certificates_dir():
    path = get_project_home()
    path = os.path.join(path, SSL_PATH)
    return _get_path_safe(path)

def get_certificates():
    script_dir = os.path.dirname(__file__)
    script_path_original = os.path.join(script_dir, GEN_CERTS_SCRIPT_NAME_DIR)
    
    if which("openssl") is None:
        return get_default_certificates()
    
    cert_dir = get_certificates_dir()
    cert_script_path = os.path.join(cert_dir, GEN_CERTS_SCRIPT_NAME)
    if not os.path.exists(cert_script_path):
        copyfile(script_path_original, cert_script_path)
        cwd = os.getcwd()
        os.chdir(cert_dir)
        os.chmod(GEN_CERTS_SCRIPT_NAME, 0o755)
        os.system("./{}".format(GEN_CERTS_SCRIPT_NAME))
        os.chdir(cwd)
    
    server_cert = os.path.join(cert_dir, SERVER_CERT_NAME)
    server_key = os.path.join(cert_dir, SERVER_KEY_NAME)
    return server_cert, server_key
    

def _get_path_safe(path: str) -> str:
    if not os.path.exists(path):
        os.makedirs(path)
    return path