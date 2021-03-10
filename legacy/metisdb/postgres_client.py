import csv
import os
import psycopg2.extras
import psycopg2
import yaml

from psycopg2 import pool


scriptDirectory = os.path.dirname(os.path.realpath(__file__))


def get_postgres_connection_pool(dbname, minconns=1, maxconns=3):
	"""
	A simple connection pool initialization function
	:param minconns:
	:param maxconns:
	:param dbname:
	:return:
	"""
	configsFilePath = os.path.join(scriptDirectory, '../resources/config/metisdb.settings.yaml')
	fstream = open(configsFilePath).read()
	# See also: https://github.com/yaml/pyyaml/wiki/PyYAML-yaml.load(input)-Deprecation
	# Loader Types:
	# 	- BaseLoader, FullLoader, SafeLoader(recommended)
	settings = yaml.load(fstream, Loader=yaml.SafeLoader)
	dbsettings = settings.get(dbname)
	postgres_host = dbsettings['host']
	postgres_db = dbsettings['dbname']
	postgres_user = dbsettings['username']
	postgres_pass = dbsettings['password']
	psqlpool = pool.ThreadedConnectionPool(minconn=minconns, maxconn=maxconns,
										   host=postgres_host, dbname=postgres_db,
										   user=postgres_user, password=postgres_pass)
	return psqlpool

def get_single_postgres_connection(dbname):
	"""
	A simple connection initialization function
	:param dbname:
	:return:
	"""
	configsFilePath = os.path.join(scriptDirectory, '../resources/config/metisdb.settings.yaml')
	fstream = open(configsFilePath).read()
	settings = yaml.load(fstream, Loader=yaml.SafeLoader)
	dbsettings = settings.get(dbname)
	postgres_host = dbsettings['host']
	postgres_db = dbsettings['dbname']
	postgres_user = dbsettings['username']
	postgres_pass = dbsettings['password']
	conn = psycopg2.connect(host=postgres_host, dbname=postgres_db, user=postgres_user, password=postgres_pass)
	return conn


def insert_metisdb_data(psqlconn, data, tblname, attributes=list()):
	"""
	A simple function for inserting a bunch of accelerometer data inside PostgresDB
	:param psqlconn: given connection to insert data
	:param data: list of tuples
	:param tblname:
	:return:
	"""
	cursor = psqlconn.cursor()
	insert_query = """INSERT INTO {} ({}) VALUES %s""".format(tblname, ','.join(attributes))
	psycopg2.extras.execute_values(cur=cursor, sql=insert_query, argslist=data, template=None, page_size=100)
	psqlconn.commit()


def select_metisdb_data(psqlconn, tblname, where=None, limit=None):
	where_stmt = ""
	limit_stmt = ""

	# TODO extend where with filtering operations
	if where is not None:
		where_stmt = """WHERE {}""".format(where)

	if limit is not None:
		limit_stmt = """LIMIT {}""".format(limit)

	network_input = []
	network_output = []

	need_to_write_data_to_filesystem = False
	filepath = get_tablename_filepath(tblname=tblname)
	if "mnist" in tblname or "cifar" in tblname:
		if filepath is not None and os.path.exists(filepath):
			# TODO this must be removed in order to query directly the database in the future ~ cifar total size 3GBs
			network_input, network_output = retrieve_data_from_filesystem(filepath, no_of_examples=int(limit))
			return network_input, network_output
		else:
			need_to_write_data_to_filesystem = True
			sql_fetch_query = """SELECT image, label FROM {} {} {};""".format(tblname, where_stmt, limit_stmt)
	else:
		sql_fetch_query = """SELECT sentence, tags FROM {} {} {};""".format(tblname, where_stmt, limit_stmt)

	cursor_fetch_size = 10000
	cursor = psqlconn.cursor('metisdb_data_selection_cursor')
	cursor.execute(sql_fetch_query)
	rows = cursor.fetchmany(cursor_fetch_size)
	while rows:
		for row in rows:
			input_mtx = row[0]
			label = row[1]
			network_input.append(input_mtx)
			network_output.append(label)
		rows = cursor.fetchmany(cursor_fetch_size)
	psqlconn.commit()


	if need_to_write_data_to_filesystem:
		write_data_to_filesystem(filepath=filepath, images=network_input, labels=network_output)


	return network_input, network_output


def get_tablename_filepath(tblname):
	if 'extended_mnist' in tblname:
		base_dir = os.path.join(scriptDirectory, '../resources/data/extended_mnist')
	elif 'fmnist' in tblname:
		base_dir = os.path.join(scriptDirectory, '../resources/data/fmnist')
	elif 'mnist' in tblname:
		base_dir = os.path.join(scriptDirectory, '../resources/data/mnist')
	elif 'cifar100' in tblname:
		base_dir = os.path.join(scriptDirectory, '../resources/data/cifar100')
	elif 'cifar10' in tblname:
		base_dir = os.path.join(scriptDirectory, '../resources/data/cifar10')
	else:
		return None

	if not os.path.exists(base_dir):
		os.makedirs(base_dir)
	filepath = os.path.join(base_dir, '{}'.format(tblname))
	return filepath


def write_data_to_filesystem(filepath, images, labels):
	with open(filepath, 'w') as fout:
		data_writer = csv.writer(fout, delimiter=',')
		for idx, img in enumerate(images):
			data_writer.writerow([img, labels[idx]])


def retrieve_data_from_filesystem(filepath, no_of_examples=500):
	images = []
	labels = []
	count = 0
	with open(filepath, 'r') as fin:
		csv_reader = csv.reader(fin, delimiter=',')
		for row in csv_reader:
			images.append([float(x) for x in row[0].strip('[,]').split(',')])
			labels.append(row[1])
			count += 1
			if count >= no_of_examples:
				break

	return images, labels