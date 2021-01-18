import tensorflow as tf
import numpy as np

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


label = np.asarray([[1,2,3],
                    [4,5,6]]).reshape(2, 3, -1)

sample = np.stack((label + 200).reshape(2, 3, -1))

writer = tf.python_io.TFRecordWriter("toy.tfrecord")

example = tf.train.Example(features=tf.train.Features(feature={
                'label_raw': _bytes_feature(tf.compat.as_bytes(label.tostring())),
                'sample_raw': _bytes_feature(tf.compat.as_bytes(sample.tostring()))}))

writer.write(example.SerializeToString())

writer.close()


def _read_from_tfrecord(example_proto):
    feature = {
        'label_raw': tf.FixedLenFeature([], tf.string),
        'sample_raw': tf.FixedLenFeature([], tf.string)
    }

    features = tf.parse_example([example_proto], features=feature)

    label_1d = tf.decode_raw(features['label_raw'], tf.int64)
    sample_1d = tf.decode_raw(features['sample_raw'], tf.int64)

    label_restored = tf.reshape(label_1d, tf.stack([2, 3, -1]))
    sample_restored = tf.reshape(sample_1d, tf.stack([2, 3, -1]))

    return label_restored, sample_restored

filename = 'toy.tfrecord'
data_path = tf.placeholder(dtype=tf.string, name="tfrecord_file")
dataset = tf.data.TFRecordDataset(data_path)
dataset = dataset.map(_read_from_tfrecord)
iterator = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)
label_tf, sample_tf = iterator.get_next()

iterator_init = iterator.make_initializer(dataset, name="dataset_init")

with tf.Session() as sess:
    sess.run(iterator_init, feed_dict={data_path: filename})

    read_label, read_sample = sess.run([label_tf, sample_tf])

print("After reading:")
print(read_label, read_sample)