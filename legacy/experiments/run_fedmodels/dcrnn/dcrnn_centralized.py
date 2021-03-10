import os
import time

import numpy as np
import tensorflow as tf
import yaml
from absl import flags, app, logging

from experiments.run_fedmodels.dcrnn.dcrnn_client import DcrnnDatasetClient
from experiments.tf_fedmodels.dcrnn.dcrnn_fed_model import DcrnnFedModel
from metisdb.metisdb_catalog import MetisCatalog
from metisdb.metisdb_session import MetisDBSession
from metisdb.sqlalchemy_client import SQLAlchemyClient

os.environ["GRPC_VERBOSITY"] = 'ERROR'

DEFAULT_BATCH_SIZE = 64
LEARNER_ID = 'centralized'

FLAGS = flags.FLAGS
flags.DEFINE_string('config_file', None, help='Path to the model config file.')
flags.DEFINE_string('tensorboard_log_dir', None, help='Directory to store tensorboard logs.')

flags.mark_flag_as_required('config_file')


# Train the model
def train(config):
    data_volume_column = '9dof_2mm_vol'
    data_task_column = 'age_at_scan'

    dataset_dir = str(config['data']['dataset_dir'])
    batch_size = int(config['data'].get('batch_size', DEFAULT_BATCH_SIZE))

    train_tmp_filename = '/tmp/dcrnn_train.tfrecord'
    val_tmp_filename = '/tmp/dcrnn_val.tfrecord'

    # Prepares the dataset.
    dcrnn_client = DcrnnDatasetClient(LEARNER_ID, dataset_dir, batch_size)

    logging.info("Starting training tfrecords generation.")
    num_train_data, train_schema = dcrnn_client.generate_tfrecords('', train_tmp_filename, is_training=True)
    logging.info('Train schema: %s', train_schema)

    logging.info("Starting validation tfrecords generation.")
    num_val_data, val_schema = dcrnn_client.generate_tfrecords('', val_tmp_filename, is_validation=True)
    logging.info('Validation schema: %s', val_schema)

    # Updates the catalog.
    sqlalchemy_client = SQLAlchemyClient(postgres_instance=True)
    METIS_CATALOG_CLIENT = MetisCatalog(sqlalchemy_client)
    metis_db_session = MetisDBSession(METIS_CATALOG_CLIENT, [dcrnn_client], is_regression=True)
    metis_db_session.register_tfrecords_volumes(LEARNER_ID, train_tmp_filename, val_tmp_filename)
    metis_db_session.register_tfrecords_schemata(LEARNER_ID, train_schema, val_schema)
    metis_db_session.register_tfrecords_examples_num(LEARNER_ID, num_train_data, num_val_data)
    metis_db_session.register_tfdatasets_schemata(LEARNER_ID)

    # Defines global step
    global_step = tf.train.get_or_create_global_step()

    # Builds the model.
    model = DcrnnFedModel(config)
    model_inputs = model.input_tensors_datatype()
    model_outputs = model.output_tensors_datatype()  # really labels
    model_architecture = model.model_architecture(model_inputs, model_outputs, global_step=global_step)

    train_op = model_architecture.train_step
    loss_tensor = model_architecture.loss
    predictions_tensor = model_architecture.predictions

    # Creates the dataset operations.
    train_data_ops, validation_data_ops, _ = metis_db_session.import_host_data(
        learner_id=LEARNER_ID, batch_size=batch_size, import_train=True, import_validation=True)

    train_dataset_init_op = train_data_ops.dataset_init_op
    next_train_batch_op = train_data_ops.dataset_next

    val_dataset_init_op = validation_data_ops.dataset_init_op
    next_val_batch_op = validation_data_ops.dataset_next

    # Configures the session.
    tf_config = tf.ConfigProto(inter_op_parallelism_threads=10, intra_op_parallelism_threads=10)
    tf_config.allow_soft_placement = True
    tf_config.gpu_options.allow_growth = True

    with tf.Session(config=tf_config) as sess:
        # Initializes graph variables and operations
        sess.run(tf.global_variables_initializer())

        epoch = int(config['train'].get('epoch', 1))
        epochs = int(config['train'].get('epochs', 100))
        while epoch <= epochs:
            # Initializes the train dataset.
            sess.run(train_dataset_init_op)

            # Runs a train epoch.
            losses = []
            train_start = time.perf_counter()
            while True:
                try:
                    batch = sess.run(next_train_batch_op)
                    feed_dict = dict()
                    feed_dict.update(train_op.get_feed_dictionary())
                    for placeholder_name, placeholder_def in model_inputs.items():
                        feed_dict[placeholder_def] = batch[placeholder_name]
                    for placeholder_name, placeholder_def in model_outputs.items():
                        feed_dict[placeholder_def] = batch[placeholder_name]

                    # Executes the train operation on the batch.
                    fetches = {
                        'loss': loss_tensor.get_tensor(),
                        'train_op': train_op.get_operation()
                    }
                    res = sess.run(fetches, feed_dict=feed_dict)
                    losses.append(res['loss'])
                except tf.errors.OutOfRangeError:
                    break
            train_end = time.perf_counter()
            train_elapsed = train_end - train_start
            train_loss = np.mean(losses)

            # Initializes the validation dataset.
            sess.run(val_dataset_init_op)

            # Runs a validation epoch.
            losses = []
            predictions = []
            val_start = time.perf_counter()
            while True:
                try:
                    batch = sess.run(next_val_batch_op)
                    feed_dict = dict()
                    feed_dict.update(predictions_tensor.get_feed_dictionary())
                    for placeholder_name, placeholder_def in model_inputs.items():
                        feed_dict[placeholder_def] = batch[placeholder_name]
                    for placeholder_name, placeholder_def in model_outputs.items():
                        feed_dict[placeholder_def] = batch[placeholder_name]

                    # Executes the train operation on the batch.
                    fetches = {
                        'loss': loss_tensor.get_tensor(),
                        'outputs': predictions_tensor.get_tensor(),
                    }
                    res = sess.run(fetches, feed_dict=feed_dict)
                    losses.append(res['loss'])
                    predictions.append(res['outputs'])
                except tf.errors.OutOfRangeError:
                    break
            val_end = time.perf_counter()
            val_elapsed = val_end - val_start
            val_loss = np.mean(losses)

            global_step = sess.run(tf.train.get_or_create_global_step())
            logging.info('#############################################################################')
            logging.info('Epoch [%d/%d] (global_step %d) finished, took %.2f seconds',
                         epoch, epochs, global_step, (train_elapsed + val_elapsed))
            logging.info('-----------------------------------------------------------------------------')
            logging.info('train_loss=%.4f\ttrain_time=%.2fs', train_loss, train_elapsed)
            logging.info('  val_loss=%.4f\t  val_time=%.2fs', val_loss, val_elapsed)


def main(argv):
    if FLAGS.tensorboard_log_dir:
        if not os.path.exists(FLAGS.tensorboard_log_dir):
            os.makedirs(FLAGS.tensorboard_log_dir)

    with open(FLAGS.config_file, 'r') as f:
        config = yaml.safe_load(f)

    train(config)


if __name__ == "__main__":
    app.run(main)
