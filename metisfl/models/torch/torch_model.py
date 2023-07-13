import cloudpickle
import collections
import inspect
import os
import shutil

import torch
import numpy as np

from metisfl import config
from metisfl.models.metis_model import MetisModel, ModelWeightsDescriptor
from metisfl.utils.metis_logger import MetisLogger


class MetisModelTorch(MetisModel):

    def __init__(self, model: torch.nn.Module):
        assert isinstance(
            model, torch.nn.Module), "MetisModelTorch must be a torch.nn.Module"
        assert hasattr(model, "fit"), "MetisModelTorch requires a .fit method"
        assert hasattr(
            model, "evaluate"), "MetisModelTorch requires an .evaluate method"

        self._backend_model = model
        self.nn_engine = config.PYTORCH_NN_ENGINE

    @staticmethod
    def get_paths(model_dir):
        model_weights_path = os.path.join(model_dir, "model_weights.pt")
        model_def_path = os.path.join(model_dir, "model_def.pkl")
        return model_def_path, model_weights_path

    @staticmethod
    def load(model_dir) -> "MetisModelTorch":
        model_def_path, model_weights_path = MetisModelTorch.get_paths(
            model_dir)
        MetisLogger.info("Loading model from: {}".format(model_dir))
        model_loaded = cloudpickle.load(open(model_def_path, "rb"))
        model_loaded.load_state_dict(torch.load(model_weights_path))
        MetisLogger.info("Loaded model from: {}".format(model_dir))
        return MetisModelTorch(model_loaded)

    def get_weights_descriptor(self) -> ModelWeightsDescriptor:
        weights_names, weights_trainable, weights_values = [], [], []
        for name, param in self._backend_model.named_parameters():
            weights_names.append(name)
            # Trainable variables require gradient computation,
            # for non-trainable the gradient is False.
            weights_trainable.append(param.requires_grad)
            weights_values.append(param.data.numpy(force=True))

        return ModelWeightsDescriptor(weights_names=weights_names,
                                      weights_trainable=weights_trainable,
                                      weights_values=weights_values)

    # The is_initial flag is not used in the torch implementation; simply here for compatibility with the tf version
    def save(self, model_dir, initialize=False):
        model_def_path, model_weights_path = MetisModelTorch.get_paths(
            model_dir)
        MetisLogger.info("Saving model to: {}".format(model_dir))
        # @stripeli why are you removing the model dir on torch.save but not on the respective tf save?
        shutil.rmtree(model_dir)
        os.makedirs(model_dir)
        cloudpickle.register_pickle_by_value(
            inspect.getmodule(self._backend_model))
        cloudpickle.dump(obj=self._backend_model,
                         file=open(model_def_path, "wb+"))
        torch.save(self._backend_model.state_dict(), model_weights_path)
        MetisLogger.info("Saved model at: {}".format(model_dir))

    def set_model_weights(self,
                          model_weights_descriptor: ModelWeightsDescriptor):
        weights_values = model_weights_descriptor.weights_values
        state_dict = collections.OrderedDict({
            k: torch.tensor(np.atleast_1d(v))
            for k, v in zip(self._backend_model.state_dict().keys(), weights_values)
        })
        self._backend_model.load_state_dict(state_dict, strict=True)
