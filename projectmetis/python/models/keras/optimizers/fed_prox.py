from tensorflow.keras.optimizers import Optimizer
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops


class FedProx(Optimizer):
    """Implementation of Perturbed Gradient Descent, i.e., FedProx optimizer"""

    def __init__(self, learning_rate=0.001, mu=0.01, use_locking=False, name="FedProx", **kwargs):
        super().__init__(name, **kwargs)
        self._set_hyper("_lr", learning_rate)
        self._set_hyper("_mu", mu)
        self._lr = learning_rate
        self._mu = mu
        self._use_locking = use_locking

        # Tensor versions of the constructor arguments, created in _prepare().
        self._lr_t = None
        self._mu_t = None

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "learning_rate": self._serialize_hyperparameter("_lr"),
            "proximal_term": self._serialize_hyperparameter("_mu"),
        }

    def _prepare(self, var_list):
        self._lr_t = ops.convert_to_tensor(self._lr, name="learning_rate")
        self._mu_t = ops.convert_to_tensor(self._mu, name="prox_mu")

    def _create_slots(self, var_list):
        # Create slots for the global solution.
        for v in var_list:
            # self._zeros_slot(v, "vstar", self._name)
            self.add_slot(v, "vstar")

    def _resource_apply_dense(self, grad, var, **apply_kwargs):
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        mu_t = math_ops.cast(self._mu_t, var.dtype.base_dtype)
        vstar = self.get_slot(var, "vstar")

        scaled_regularization_term = lr_t * mu_t * (var - vstar)
        scaled_gradient_term = lr_t * grad
        scaled_update = scaled_gradient_term + scaled_regularization_term
        var_update = state_ops.assign_sub(var, scaled_update)

        return control_flow_ops.group(*[var_update, ])