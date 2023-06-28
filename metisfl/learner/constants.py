import os 
DEFAULT_LEARNER_HOST = "[::]"
DEFAULT_LEARNER_PORT = 50052
DEFAULT_CONTROLLER_HOSTNAME = "[::]"
DEFAULT_CONTROLLER_PORT = 50051
LEARNER_CREDENTIALS_FP =  "/tmp/metis/learner_{}_credentials/"

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

# TODO: @stripeli are these files maintined between runs?
_LEARNER_ID_FP = os.path.join(LEARNER_CREDENTIALS_FP, LEARNER_ID_FILE)
_AUTH_TOKEN_FP = os.path.join(LEARNER_CREDENTIALS_FP, AUTH_TOKEN_FILE)

def get_learner_id_fp(learner_id):
    path = _LEARNER_ID_FP.format(learner_id)
    _ensure_path(path)
    return os.path.join(path, LEARNER_ID_FILE)

def get_auth_token_fp(learner_id):
    path = _AUTH_TOKEN_FP.format(learner_id)
    _ensure_path(path)
    return os.path.join(path, AUTH_TOKEN_FILE)
    
def _ensure_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
