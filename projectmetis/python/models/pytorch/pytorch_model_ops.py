import numpy as np
import torch

try:
    from projectmetis.proto import metis_pb2
except (ImportError, ModuleNotFoundError, AttributeError):

    class DummyPB:
        def __getattr__(self, name):
            return self

    metis_pb2 = DummyPB()
from projectmetis.python.logging.metis_logger import MetisLogger
from projectmetis.python.models.model_dataset import ModelDataset
from projectmetis.python.models.model_ops import ModelOps
from projectmetis.python.models.pytorch.quantization import (
    dequantize_state_dict,
    quantize_state_dict,
)


class PyTorchModelOps(ModelOps):
    def __init__(self, model=None, model_filepath="/tmp/model", *args, **kwargs):
        self._model_filepath = model_filepath
        if model is not None:
            self._model = model
        else:
            self._model = self.load_model(self._model_filepath)
        super().__init__(self._model)

    def load_model(self, filepath=None, *args, **kwargs):
        if filepath is None:
            filepath = self._model_filepath
        MetisLogger.info(f"Loading model from: {filepath}")
        m = torch.load(filepath)
        MetisLogger.info(f"Loaded model from: {filepath}")
        return m

    def save_model(self, filepath=None, *args, **kwargs):
        if filepath is None:
            filepath = self._model_filepath
        MetisLogger.info(f"Saving model to: {filepath}")
        torch.save(self._model, filepath=filepath)
        MetisLogger.info(f"Saved model at: {filepath}")

    def set_model_weights(
        self,
        weights_names=None,
        weights_trainable=None,
        weights_values=None,
        *args,
        **kwargs,
    ):
        """Set model weights from lists (weights_names, weights_trainable, weights_values)
        or a state dict / quantized dictionary.
        """
        MetisLogger.info("Applying new PyTorch model weights")
        if isinstance(weights_names, dict) and weights_values is None:
            weights_dict = weights_names
            if "layers" in weights_dict and "nbits" in weights_dict:
                state_dict = dequantize_state_dict(weights_dict)
            else:
                state_dict = weights_dict
            self._model.load_state_dict(state_dict)
        elif weights_names is not None and weights_values is not None:
            state_dict = self._model.state_dict()
            for name, val in zip(weights_names, weights_values):
                if name in state_dict:
                    if isinstance(val, np.ndarray):
                        val_tensor = torch.from_numpy(val)
                    elif isinstance(val, torch.Tensor):
                        val_tensor = val
                    else:
                        val_tensor = torch.as_tensor(val)
                    state_dict[name].copy_(val_tensor)
            self._model.load_state_dict(state_dict)
        MetisLogger.info("Applied new PyTorch model weights")

    def get_model_weights(
        self,
        as_numpy=False,
        quantized=False,
        nbits=4,
        block_size=64,
        *args,
        **kwargs,
    ):
        """Get model weights from PyTorch model.

        If quantized=True: returns quantized state dict dictionary with sub-byte bit packing.
        If as_numpy=True: returns (weights_names, weights_trainable, weights_values).
        Otherwise: returns model.state_dict().
        """
        state_dict = self._model.state_dict()
        if quantized:
            quant_dict, _, _ = quantize_state_dict(
                state_dict, nbits=nbits, block_size=block_size
            )
            return quant_dict

        if as_numpy:
            names = []
            trainables = []
            values = []
            named_params = dict(self._model.named_parameters())
            for name, tensor in state_dict.items():
                names.append(name)
                trainables.append(
                    named_params[name].requires_grad if name in named_params else False
                )
                values.append(tensor.detach().cpu().numpy())
            return names, trainables, values

        return state_dict

    def train_model(
        self,
        train_dataset: ModelDataset,
        total_steps=100,
        batch_size=100,
        validation_dataset: ModelDataset = None,
        verbose=False,
        *args,
        **kwargs,
    ):
        pass

    def evaluate_model(
        self,
        dataset: ModelDataset = None,
        batch_size=100,
        metrics=None,
        verbose=False,
        *args,
        **kwargs,
    ):
        pass

    def infer_model(
        self, dataset: ModelDataset = None, batch_size=100, *args, **kwargs
    ):
        pass

    def construct_optimizer(
        self, hyperparameters_pb: metis_pb2.Hyperparameters = None, *args, **kwargs
    ):
        pass
