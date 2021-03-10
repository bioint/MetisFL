import tensorflow as tf


def kld_loss(preds, dists, collection_name):
	# Prevent over/under-flow
	p = preds + 1e-10
	q = dists + 1e-10

	# Calc KL divergence
	loss = tf.reduce_sum(p * tf.log(p / q), axis=1)
	loss = tf.reduce_mean(loss)

	# Add to loss
	tf.losses.add_loss(loss, loss_collection=collection_name)


def rkld_loss(preds, dists, collection_name):
	# Prevent over/under-flow
	p = preds + 1e-10
	q = dists + 1e-10

	# Calc KL divergence
	loss = tf.reduce_sum(q * tf.log(q / p), axis=1)
	loss = tf.reduce_mean(loss)

	# Add to loss
	tf.losses.add_loss(loss, loss_collection=collection_name)


def skld_loss(preds, dists, collection_name):
	# Prevent over/under-flow
	p = preds + 1e-10
	q = dists + 1e-10

	# Calc KL divergence
	loss1 = tf.reduce_sum(p * tf.log(p / q), axis=1)
	loss2 = tf.reduce_sum(q * tf.log(q / p), axis=1)

	loss = tf.reduce_mean(loss1 + loss2)

	# Add to loss
	tf.losses.add_loss(loss, loss_collection=collection_name)


def js_loss(preds, dists, collection_name):
	# Prevent over/under-flow
	p = dists + 1e-10
	q = preds + 1e-10

	m = 0.5 * (p + q)

	# Calc Jensen-Shannon divergence
	jsd_p = 0.5 * tf.reduce_sum(p * tf.log(p / m), axis=1)
	jsd_q = 0.5 * tf.reduce_sum(q * tf.log(q / m), aixs=1)

	loss = tf.reduce_mean(jsd_p + jsd_q)

	# Add to loss
	tf.losses.add_loss(loss, loss_collection=collection_name)


def bhat_loss(preds, dists, collection_name):
	# Prevent over/under-flow
	p = dists + 1e-10
	q = preds + 1e-10

	# Bhattacharyya constant
	bc = tf.reduce_sum(tf.sqrt(p * q), axis=1)

	# Calc Bhattacharya distance
	bd = -1 * tf.log(bc)

	loss = tf.reduce_mean(bd)

	# Add to loss
	tf.losses.add_loss(loss, loss_collection=collection_name)


def hell_loss(preds, dists, collection_name):
	# Prevent over/under-flow
	p = dists + 1e-10
	q = preds + 1e-10

	# Calc Hellinger distance
	hd = tf.norm(tf.sqrt(p) - tf.sqrt(q), axis=1)
	hd /= 2 ** 0.5

	loss = tf.reduce_mean(hd)

	# Add to loss
	tf.losses.add_loss(loss, loss_collection=collection_name)
