import tensorflow as tf

import federation.fed_model as fedmodel
from experiments.tf_fedmodels.dcrnn.dcrnn_model import DCRNNModel
from utils.dcrnn import dcrnn_utils


class DcrnnFedModel(fedmodel.FedModelDef):
    """Implements the DCRNN Traffic Forecasting model.
    Paper Reference: https://arxiv.org/pdf/1707.01926.pdf
    """

    def __init__(self, config):
        # Config sections
        self._data_config = config['data']
        self._model_config = config['model']
        self._train_config = config['train']

        # Data params
        batch_size = int(self._data_config.get('batch_size', 64))
        graph_pkl_filename = self._data_config['graph_pkl_filename']
        _, _, adj_mx = dcrnn_utils.load_graph_data(graph_pkl_filename)

        # TODO(canastas): how can we load the scaler in a better way?
        self.scaler = dcrnn_utils.StandardScaler(54.41241757911363, 19.48757953822608)

        # Model params
        self.num_nodes = self._model_config['num_nodes']
        self.horizon = self._model_config['horizon']
        self.input_dim = self._model_config['input_dim']
        self.output_dim = self._model_config['output_dim']

        # Train params
        self.max_grad_norm = float(self._train_config.get('max_grad_norm', 1.))
        self.base_lr = float(self._train_config.get('base_lr', 0.01))

        base_lr = tf.constant_initializer(self.base_lr)
        self._lr = tf.get_variable('learning_rate', shape=(), initializer=base_lr, trainable=False)

        # Model
        with tf.variable_scope('DCRNN'):
            self._model = DCRNNModel(adj_mx, batch_size, **self._model_config)

    def input_tensors_datatype(self, **kwargs):
        return self.inputs

    def output_tensors_datatype(self, **kwargs):
        return self.labels

    @property
    def inputs(self):
        return {'x': self._model.inputs}

    @property
    def labels(self):
        return {'y': self._model.labels}

    @property
    def outputs(self):
        return self._model.outputs

    def model_architecture(self, inputs, labels, global_step=None, batch_size=None, dataset_size=None, **kwargs):
        # Defines the loss operation.
        preds = self._model.outputs
        labels = self._model.labels[..., :self.output_dim]

        loss_fn = dcrnn_utils.masked_mae_loss(self.scaler, 0.)
        loss = loss_fn(preds=preds, labels=labels)

        # Creates the optimizer
        optimizer_name = self._train_config.get('optimizer', 'sgd').lower()
        if optimizer_name == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(self._lr)
        else:
            epsilon = float(self._train_config.get('epsilon', 1e-3))
            optimizer = tf.train.AdamOptimizer(self._lr, epsilon=epsilon)

        # Defines the train operation.
        tvars = tf.trainable_variables()
        grads = tf.gradients(loss, tvars)
        grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)
        global_step = tf.train.get_or_create_global_step()
        train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step, name='train_op')

        # Federated Model Properties
        fed_loss_tensor = fedmodel.FedTensor(tensor=loss, feed_dict={self._model.is_training: False})
        fed_train_step_op = fedmodel.FedOperation(operation=train_op, feed_dict={self._model.is_training: True})
        fed_predictions = fedmodel.FedTensor(tensor=self.outputs, feed_dict={self._model.is_training: False})
        fed_trainable_variables_collection = tf.trainable_variables(scope="DCRNN")

        return fedmodel.ModelArchitecture(loss=fed_loss_tensor,
                                          train_step=fed_train_step_op,
                                          predictions=fed_predictions,
                                          model_federated_variables=fed_trainable_variables_collection)
