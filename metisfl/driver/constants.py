TRAIN = "train"
VALIDATION = "validation"
TEST = "test"
TASK_KEYS = [TRAIN, VALIDATION, TEST]

MODEL_SAVE_DIR_NAME = "model_definition"
DATASET_RECEIPE_FILENAMES = {
    "train": "model_train_dataset_ops.pkl",
    "validation": "model_validation_dataset_ops.pkl",
    "test": "model_test_dataset_ops.pkl"
}
REMOTE_METIS_CONTROLLER_PATH = "/tmp/metis/controller"
REMOTE_METIS_LEARNER_PATH = "/tmp/metis/workdir_learner_{}"
