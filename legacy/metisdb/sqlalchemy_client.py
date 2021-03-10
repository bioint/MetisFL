import os.path
import sqlalchemy.engine
import sqlalchemy.sql

import metisdb.postgres_client as metis_pgclient
import sqlalchemy.pool as pool

from functools import partial
from sqlalchemy.orm import scoped_session
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool


class SQLAlchemyClient(object):

	SQLALCHEMY_DTYPE_INDEX = {
		'big_integer': sqlalchemy.sql.sqltypes.BigInteger,
		'integer': sqlalchemy.sql.sqltypes.BigInteger,
		'float': sqlalchemy.sql.sqltypes.Float,
		'boolean': sqlalchemy.sql.sqltypes.Boolean,
		'string': sqlalchemy.sql.sqltypes.String,
	}

	def __init__(self, dbname='MetisCatalog', sqlite_instance_inmemory=False,
				 sqlite_instance_fs=False, postgres_instance=False, maxconns=20):

		assert (sqlite_instance_inmemory is True or sqlite_instance_fs is True or postgres_instance is True), \
			"Need to specify whether you want a (local) sqlite or (remote) postgres instance."

		self.dbname = dbname
		self.is_sqlite_instance = sqlite_instance_inmemory or sqlite_instance_fs
		self.is_postgres_instance = postgres_instance

		if sqlite_instance_fs:
			# File-based, all tables and related metadata are stored in '{dbname}.db' file at the current directory.
			dbpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "{}.db".format(self.dbname))
			self.dbengine = sqlalchemy.engine.create_engine('sqlite:///{}'.format(dbpath))
		elif sqlite_instance_inmemory:
			# In-memory, since we want the database connection to be shared in a multithreaded scenario, we need to
			# reuse the same connection object among threads. Therefore there are two alternatives.
			# 	1. Define a sqlite creator with cache=shared directly through sqlite3.
			#	2. Define a sqlalchemy connection pooling with StaticPool.
			# References:
			# 	- https://stackoverflow.com/questions/27910829/sqlalchemy-and-sqlite-shared-cache
			# 	- https://docs.sqlalchemy.org/en/13/dialects/sqlite.html#threading-pooling-behavior

			# dbmem, params = 'file::memory:?cache=shared', {'uri': True}
			# creator = lambda: sqlite3.connect(dbmem, **params)
			# self.dbengine = sqlalchemy.engine.create_engine("sqlite:///:memory:", creator=creator)
			self.dbengine = sqlalchemy.engine.create_engine("sqlite://", connect_args={'check_same_thread':False},
															poolclass=StaticPool)
		elif self.is_postgres_instance:
			# Since pool.QueuePool does not feed the function passed to the creator argument with any parameters,
			# we use the partial function tool, to pass those arguments when the function is called.
			connection_pool = pool.QueuePool(
				creator=partial(metis_pgclient.get_single_postgres_connection, dbname=dbname), max_overflow=30,
				pool_size=maxconns)
			self.dbengine = sqlalchemy.engine.create_engine('postgresql+psycopg2://', pool=connection_pool)

		self.session_factory = sessionmaker(bind=self.dbengine, autocommit=True, expire_on_commit=False)
		self.session = scoped_session(self.session_factory)