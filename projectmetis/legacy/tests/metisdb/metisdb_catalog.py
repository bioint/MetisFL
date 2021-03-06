import tensorflow as tf
from metisdb.sqlalchemy_client import SQLAlchemyClient
from metisdb.metisdb_catalog import MetisCatalog

psql_sqlalchemy_client = SQLAlchemyClient(postgres_instance=True)
metis_catalog = MetisCatalog(psql_sqlalchemy_client)
psql_sqlalchemy_client2 = SQLAlchemyClient(postgres_instance=True)
metis_catalog2 = MetisCatalog(psql_sqlalchemy_client2)

# Singleton: Ensure that the two catalog object refer to the same object.
assert id(metis_catalog) == id(metis_catalog2), "Not a singleton, two different MetisCatalog objects!"
assert id(metis_catalog.sqlalchemy_client.dbengine) == id(metis_catalog2.sqlalchemy_client.dbengine), \
	"Not a singleton, two different sql engines!"

features = {
	'height': tf.float32,
	'width': tf.int64
}
registered_schema = metis_catalog.register_tfdataset_schema_wcatalog("learner1", features, "height", "width", "height", "width")

features = {
	'height': tf.float32,
	'width': tf.int64,
	'depth': tf.string,
}
registered_schema = metis_catalog.register_tfdataset_schema_wcatalog("learner2", features, "height", "width", "height", "depth")

features = {
	'height': tf.float32,
	'width': tf.int64,
	'depth': tf.string,
	'label': tf.int32,
	'image': tf.string
}
registered_schema = metis_catalog.register_tfdataset_schema_wcatalog("learner3", features, "height", "width", "image", "label")

features = {
	'height': tf.float32,
	'width': tf.int64,
	'depth': tf.string,
	'label': tf.int32,
	'image': tf.string,
	'image2': tf.string
}
registered_schema = metis_catalog.register_tfdataset_schema_wcatalog("learner4", features, "height", "width", "image2", "label")


features = {
	'height': tf.float32,
	'width': tf.int64,
	'label': tf.int32,
}
registered_schema = metis_catalog.register_tfdataset_schema_wcatalog("learner4", features, "height", "width", "label", "width")

metis_catalog.register_tfrecords_volumes_wcatalog("learner1", "/lfs1/stripeli")
metis_catalog.register_tfrecords_volumes_wcatalog("learner1", "/lfs1/stripeli/train", "/lfs1/stripeli/val")
metis_catalog.register_tfrecords_volumes_wcatalog("learner1", "/lfs1/stripeli/train", "/lfs1/stripeli/val", "/lfs1/stripeli/test")

print(metis_catalog.get_learner_training_attrs("learner4"))
print(metis_catalog.get_learner_tfrecord_volume("learner1", train=True, valid=True, test=False))

for i in range(100000):
	metis_catalog.get_learner_tfrecord_volume("learner1", train=True)

metis_catalog.shutdown()
