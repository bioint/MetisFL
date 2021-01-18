from utils.languagemodels.ner_sequence_tagging_utils import NERTaggingUtils

import ner_sequence_tagging.tf_metrics_guillaumegential as metrics
import federation.fed_model as fedmodel
import tensorflow as tf
import os

dirname = os.path.dirname(__file__)

class NERBioCreativeFedModel(fedmodel.FedModelDef):

	def __init__(self, nwords, nchars, ntags, tags_indices):
		self.dim_word = 300 # word embeddings dimensions
		self.dim_char = 100 # char embeddings dimensions
		self.nwords = nwords
		self.nchars = nchars
		self.ntags = ntags
		self.tags_indices = tags_indices
		self.hidden_size_char = 100  # lstm on chars
		self.hidden_size_lstm = 300  # lstm on word embeddings
		self.learning_rate = 0.001
		# self.lr_decay = 0.9

	def input_tensors_datatype(self):
		return {
			# shape = (batch size, max length of sentence, max length of word)
			'char_ids': tf.placeholder(tf.int32, shape=[None, None, None], name="char_ids"),
			# shape = (batch size, max length of sentence in batch)
			'word_ids': tf.placeholder(tf.int32, shape=[None, None], name="word_ids"),
			# shape = (batch size)
			'sequence_lengths': tf.placeholder(tf.int32, shape=[None], name="sequence_lengths"),
			# shape = (batch_size, max_length of sentence)
			'word_lengths': tf.placeholder(tf.int32, shape=[None, None], name="word_lengths"),
			# shape = 1
			'dropout': tf.placeholder(dtype=tf.float32, shape=[], name="dropout")
		}

	def output_tensors_datatype(self):
		# shape = (batch size, max length of sentence in batch)
		return { 'labels': tf.placeholder(tf.int32, shape=[None, None], name="labels") }

	def model_architecture(self, input_tensors, output_tensors, global_step=None, batch_size=None, dataset_size=None, **kwargs):

		char_ids_input = input_tensors['char_ids']
		word_ids_input = input_tensors['word_ids']
		sequence_lengths_input = input_tensors['sequence_lengths']
		word_lengths_input = input_tensors['word_lengths']
		dropout = input_tensors['dropout']
		labels_output = output_tensors['labels']
		glove_filename_trimmed = os.path.join(dirname, '../../../resources/data/glove/glove.6B.{}d.trimmed.npz'.format(self.dim_word))

		def inference():
			"""
			Add word and char embeddings operations and finally add logits.
			"""
			with tf.variable_scope("words"):
				# Using pre-initialized word embeddings (from glove)
				glove_vectors = NERTaggingUtils.get_trimmed_glove_vectors(glove_filename_trimmed)
				# Do not train word embeddings
				_word_embeddings = tf.Variable(glove_vectors, name="_word_embeddings", dtype=tf.float32, trainable=False)
				word_embeddings = tf.nn.embedding_lookup(_word_embeddings, word_ids_input, name="word_embeddings")

			with tf.variable_scope("chars"):
				# get char embeddings matrix
				_char_embeddings = tf.get_variable(name="_char_embeddings", dtype=tf.float32, shape=[self.nchars, self.dim_char])
				char_embeddings = tf.nn.embedding_lookup(_char_embeddings, char_ids_input, name="char_embeddings")

				# put the time dimension on axis=1
				s = tf.shape(char_embeddings)
				char_embeddings = tf.reshape(char_embeddings, shape=[s[0] * s[1], s[-2], self.dim_char])
				word_lengths = tf.reshape(word_lengths_input, shape=[s[0] * s[1]])

				# bi lstm on chars
				cell_fw = tf.contrib.rnn.LSTMCell(self.hidden_size_char, state_is_tuple=True)
				cell_bw = tf.contrib.rnn.LSTMCell(self.hidden_size_char, state_is_tuple=True)
				_output = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, char_embeddings,
														  sequence_length=word_lengths,
														  dtype=tf.float32)

				# read and concat output
				_, ((_, output_fw), (_, output_bw)) = _output
				output = tf.concat([output_fw, output_bw], axis=-1)

				# shape = (batch size, max sentence length, char hidden size)
				output = tf.reshape(output, shape=[s[0], s[1], 2 * self.hidden_size_char])
				word_embeddings = tf.concat([word_embeddings, output], axis=-1)

				word_embeddings = tf.nn.dropout(word_embeddings, dropout)


			with tf.variable_scope("bi-lstm"):
				cell_fw = tf.contrib.rnn.LSTMCell(self.hidden_size_lstm)
				cell_bw = tf.contrib.rnn.LSTMCell(self.hidden_size_lstm)
				(output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, word_embeddings,
																			sequence_length=sequence_lengths_input,
																			dtype=tf.float32)
				output = tf.concat([output_fw, output_bw], axis=-1)
				output = tf.nn.dropout(output, dropout)

			with tf.variable_scope("proj"):
				W = tf.get_variable("W", dtype=tf.float32, shape=[2 * self.hidden_size_lstm, self.ntags])
				b = tf.get_variable("b", shape=[self.ntags], dtype=tf.float32, initializer=tf.zeros_initializer())

				nsteps = tf.shape(output)[1]
				output = tf.reshape(output, [-1, 2 * self.hidden_size_lstm])
				pred = tf.matmul(output, W) + b
				logits = tf.reshape(pred, [-1, nsteps, self.ntags])

			return logits


		def loss(logits, labels):
			"""
			Add loss operations
			"""
			'''With CRF'''
			log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(logits, labels, sequence_lengths_input)
			loss = tf.reduce_mean(-log_likelihood)

			return transition_params, loss


		def train(total_loss, learning_rate):
			"""
			Add training operations
			"""
			optimizer = tf.train.AdamOptimizer(learning_rate)
			train_op = optimizer.minimize(total_loss)
			return train_op


		def predictions(eval_logits, eval_trans_params):
			crf_params = tf.get_variable("crf", [self.ntags, self.ntags], dtype=tf.float32)
			pred_ids, _ = tf.contrib.crf.crf_decode(eval_logits, eval_trans_params, sequence_lengths_input)
			return pred_ids

		def accuracy(pred_ids):
			weights = tf.sequence_mask(sequence_lengths_input)
			acc, acc_op = tf.metrics.accuracy(labels_output, pred_ids, weights)
			return acc_op

		def precision(pred_ids):
			weights = tf.sequence_mask(sequence_lengths_input)
			prec, prec_op = metrics.precision(labels=labels_output, predictions=pred_ids, num_classes=self.ntags, pos_indices=self.tags_indices, weights=weights)
			return prec_op

		def recall(pred_ids):
			weights = tf.sequence_mask(sequence_lengths_input)
			rec, rec_op = metrics.recall(labels=labels_output, predictions=pred_ids, num_classes=self.ntags, pos_indices=self.tags_indices, weights=weights)
			return rec_op

		def f1_score(pred_ids):
			weights = tf.sequence_mask(sequence_lengths_input)
			f1, f1_op = metrics.f1(labels=labels_output, predictions=pred_ids, num_classes=self.ntags, pos_indices=self.tags_indices, weights=weights)
			return f1_op

		logits = inference()
		loss_components = loss(logits, labels_output)
		transition_params = loss_components[0]
		loss_tensor = loss_components[1]
		train_op = train(loss_tensor, self.learning_rate)
		predictions_tensor = predictions(logits, transition_params)

		accuracy_tensor = accuracy(predictions_tensor)
		precision_tensor = precision(predictions_tensor)
		recall_tensor = recall(predictions_tensor)
		f1_score_tensor = f1_score(predictions_tensor)

		fed_loss_tensor = fedmodel.FedTensor(tensor=loss_tensor, feed_dict={})
		fed_train_op = fedmodel.FedOperation(operation=train_op, feed_dict={})
		fed_predictions_tensor = fedmodel.FedTensor(tensor=predictions_tensor, feed_dict={})
		fed_accuracy_tensor = fedmodel.FedTensor(tensor=accuracy_tensor, feed_dict={})
		fed_precision_tensor = fedmodel.FedTensor(tensor=precision_tensor, feed_dict={})
		fed_recall_tensor = fedmodel.FedTensor(tensor=recall_tensor, feed_dict={})
		fed_f1_score = fedmodel.FedTensor(tensor=f1_score_tensor, feed_dict={})
		fed_trainable_variables_collection = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

		return fedmodel.ModelArchitecture(loss=fed_loss_tensor,
										  train_step=fed_train_op,
										  predictions=fed_predictions_tensor,
										  model_federated_variables=fed_trainable_variables_collection,
										  accuracy=fed_accuracy_tensor)