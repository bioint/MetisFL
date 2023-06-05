import tensorflow as tf

from tensorflow.keras.optimizers import Optimizer
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops


class FedProx(Optimizer):
    """Implementation of Perturbed Gradient Descent, i.e., FedProx optimizer"""

    def __init__(self, learning_rate=0.01, proximal_term=0.001, use_locking=False, name="FedProx", **kwargs):
        super().__init__(name, **kwargs)
        self._set_hyper("learning_rate", learning_rate)
        self._set_hyper("proximal_term", proximal_term)
        self.learning_rate = learning_rate
        self.proximal_term = proximal_term
        self._use_locking = use_locking

        self._is_learning_rate_scheduled = False
        if isinstance(learning_rate, tf.keras.optimizers.schedules.LearningRateSchedule):
            self._is_learning_rate_scheduled = True

        # Tensor versions of the constructor arguments, created in _prepare().
        self.learning_rate_t = None
        self.proximal_term_t = None

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "learning_rate": self._serialize_hyperparameter("learning_rate"),
            "proximal_term": self._serialize_hyperparameter("proximal_term"),
        }

    def _prepare(self, var_list):
        if self._is_learning_rate_scheduled:
            self.learning_rate_t = ops.convert_to_tensor(self._lr(self.iterations), name="learning_rate")
        else:
            self.learning_rate_t = ops.convert_to_tensor(self.learning_rate, name="learning_rate")
        self.proximal_term_t = ops.convert_to_tensor(self.proximal_term, name="proximal_term")

    def _create_slots(self, var_list):
        # Create slots for the global solution.
        for v in var_list:
            # self._zeros_slot(v, "vstar", self._name)
            self.add_slot(v, "vstar")

    def _resource_apply_dense(self, grad, var, **apply_kwargs):
        lr_t = math_ops.cast(self.learning_rate_t, var.dtype.base_dtype)
        mu_t = math_ops.cast(self.proximal_term_t, var.dtype.base_dtype)
        vstar = self.get_slot(var, "vstar")

        scaled_regularization_term = lr_t * mu_t * (var - vstar)
        scaled_gradient_term = lr_t * grad
        scaled_update = scaled_gradient_term + scaled_regularization_term
        var_update = state_ops.assign_sub(var, scaled_update)

        return control_flow_ops.group(*[var_update, ])

    def _apply_sparse_shared(self, grad, var, indices, scatter_add):
        lr_t = math_ops.cast(self.learning_rate_t, var.dtype.base_dtype)
        mu_t = math_ops.cast(self.proximal_term_t, var.dtype.base_dtype)
        vstar = self.get_slot(var, "vstar")

        scaled_regularization_term = lr_t * mu_t * (var - vstar)
        scaled_gradient_term = lr_t * grad

        # v_diff = state_ops.assign(vstar, mu_t * (var - vstar), use_locking=self._use_locking)
        v_diff = state_ops.assign(vstar, scaled_regularization_term, use_locking=self._use_locking)

        with ops.control_dependencies([v_diff]):  # run v_diff operation before scatter_add
            # scaled_grad = scatter_add(vstar, indices, grad)
            scaled_update = scatter_add(vstar, indices, scaled_gradient_term)

        # var_update = state_ops.assign_sub(var, lr_t * scaled_grad)
        var_update = state_ops.assign_sub(var, scaled_update)

        return control_flow_ops.group(*[var_update, ])

    def _resource_apply_sparse(self, grad, var, indices, apply_state):
        lr_t = math_ops.cast(self.learning_rate_t, var.dtype.base_dtype)
        mu_t = math_ops.cast(self.proximal_term_t, var.dtype.base_dtype)
        vstar = self.get_slot(var, "vstar")

        scaled_regularization_term = lr_t * mu_t * (var - vstar)
        scaled_gradient_term = lr_t * grad

        # v_diff = state_ops.assign(vstar, mu_t * (var - vstar), use_locking=self._use_locking)
        v_diff = state_ops.assign(vstar, scaled_regularization_term, use_locking=self._use_locking)

        with ops.control_dependencies([v_diff]):  # run v_diff operation before scatter_add
            # scaled_grad = apply_state(vstar, indices, grad)
            # scaled_update = apply_state(vstar, indices, scaled_gradient_term)
            scaled_update = tf.compat.v1.scatter_add(vstar, indices, scaled_gradient_term)

        # var_update = state_ops.assign_sub(var, lr_t * scaled_grad)
        var_update = state_ops.assign_sub(var, scaled_update)

        return control_flow_ops.group(*[var_update, ])
