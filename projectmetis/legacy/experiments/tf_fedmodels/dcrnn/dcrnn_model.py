import tensorflow as tf
from tensorflow.contrib import legacy_seq2seq
from tensorflow.contrib import rnn

from model.dcrnn_cell import DCGRUCell


class DCRNNModel(object):
    def __init__(self, adj_mx, batch_size, **model_kwargs):
        max_diffusion_step = int(model_kwargs.get('max_diffusion_step', 2))
        cl_decay_steps = int(model_kwargs.get('cl_decay_steps', 1000))
        filter_type = model_kwargs.get('filter_type', 'laplacian')
        horizon = int(model_kwargs.get('horizon', 1))
        num_nodes = int(model_kwargs.get('num_nodes', 1))
        num_rnn_layers = int(model_kwargs.get('num_rnn_layers', 1))
        rnn_units = int(model_kwargs.get('rnn_units'))
        seq_len = int(model_kwargs.get('seq_len'))
        use_curriculum_learning = bool(model_kwargs.get('use_curriculum_learning', False))
        input_dim = int(model_kwargs.get('input_dim', 1))
        output_dim = int(model_kwargs.get('output_dim', 1))

        # Inputs (batch_size, time-steps, num_nodes, input_dim)
        self._inputs = tf.placeholder(tf.float32, shape=(batch_size, seq_len, num_nodes, input_dim), name='inputs')
        # Labels: (batch_size, time-steps, num_nodes, input_dim), same format with input except the temporal dimension.
        self._labels = tf.placeholder(tf.float32, shape=(batch_size, horizon, num_nodes, input_dim), name='labels')

        self._is_training = tf.placeholder(tf.bool, (), name='is_training')

        common_cell_opts = {
            'max_diffusion_step': max_diffusion_step,
            'num_nodes': num_nodes,
            'filter_type': filter_type
        }
        cell = DCGRUCell(rnn_units, adj_mx, **common_cell_opts)
        cell_with_projection = DCGRUCell(rnn_units, adj_mx, num_proj=output_dim, **common_cell_opts)

        encoding_cells_ = [cell] * num_rnn_layers
        decoding_cells_ = [cell] * (num_rnn_layers - 1) + [cell_with_projection]

        encoding_cells = tf.contrib.rnn.MultiRNNCell(encoding_cells_, state_is_tuple=True)
        decoding_cells = tf.contrib.rnn.MultiRNNCell(decoding_cells_, state_is_tuple=True)

        with tf.variable_scope('SEQ2SEQ'):
            global_step = tf.train.get_or_create_global_step()
            GO_SYMBOL = tf.zeros(shape=(batch_size, num_nodes * output_dim))

            inputs = tf.reshape(self._inputs, (batch_size, seq_len, num_nodes * input_dim))
            inputs = tf.unstack(inputs, axis=1)
            labels = tf.reshape(self._labels[..., :output_dim], (batch_size, horizon, num_nodes * output_dim))
            labels = tf.unstack(labels, axis=1)
            labels.insert(0, GO_SYMBOL)

            def _curriculum_loop(prev, i):
                def _do_sample():
                    # Returns either the model's prediction or the previous ground truth.
                    c = tf.random_uniform((), minval=0, maxval=1.)
                    threshold = self._compute_sampling_threshold(global_step, cl_decay_steps)
                    return tf.cond(tf.less(c, threshold), lambda: labels[i], lambda: prev)

                # If in training and curriculum learning is enabled, then runs `_do_sample`. Otherwise, it simply
                # returns the previous prediction.
                return tf.cond(tf.logical_and(self.is_training, use_curriculum_learning), _do_sample, lambda: prev)

            _, enc_state = rnn.static_rnn(encoding_cells, inputs, dtype=tf.float32)
            outputs, final_state = legacy_seq2seq.rnn_decoder(
                labels, enc_state, decoding_cells, loop_function=_curriculum_loop)

        # Project the output to output_dim.
        # Outputs: (batch_size, timesteps, num_nodes, output_dim)
        outputs = tf.stack(outputs[:-1], axis=1)
        self._outputs = tf.reshape(outputs, (batch_size, horizon, num_nodes, output_dim), name='outputs')
        self._merged = tf.summary.merge_all()

    @staticmethod
    def _compute_sampling_threshold(global_step, k):
        """Computes the sampling probability for scheduled sampling using inverse sigmoid.

        :param global_step:
        :param k:
        :return:
        """
        return tf.cast(k / (k + tf.exp(global_step / k)), tf.float32)

    @property
    def inputs(self):
        return self._inputs

    @property
    def outputs(self):
        return self._outputs

    @property
    def labels(self):
        return self._labels

    @property
    def is_training(self):
        return self._is_training

    @property
    def merged(self):
        return self._merged
