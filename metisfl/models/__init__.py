from metisfl.models.model_ops import ModelOps


def get_model_ops_fn(nn_engine) -> ModelOps:
    if nn_engine == "keras":
        from metisfl.models.keras.keras_model_ops import KerasModelOps
        return KerasModelOps
    elif nn_engine == "pytorch":
        from metisfl.models.pytorch.pytorch_model_ops import PyTorchModelOps
        return PyTorchModelOps
    else:
        raise ValueError("Unknown neural engine: {}".format(nn_engine))