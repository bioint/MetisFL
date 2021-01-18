import copy
import multiprocessing
import os
import pickle
import yaml

import tensorflow as tf

from utils.devops.network_ops import NetOpsUtil
from utils.devops.remote_tf_servers_ops import RemoteTFServerUtils
from utils.logging.metis_logger import MetisLogger as metis_logger


class FedHost(object):

	def __init__(self, name, fed_parameter_servers, fed_worker_servers, fed_master, cluster_spec=None,
				 dataset_configs=None):
		if not any(isinstance(fed_parameter_server, FedTFParameterServer)
				   for fed_parameter_server in fed_parameter_servers):
			raise TypeError("All the parameter servers passed here must be of type %s " % FedTFParameterServer)
		if not any(isinstance(fed_worker_server, FedWorkerServer) for fed_worker_server in fed_worker_servers):
			raise TypeError("All the worker servers passed here must be of type %s " % FedWorkerServer)
		self.name = name
		self.fed_parameter_servers = fed_parameter_servers
		self.fed_worker_servers = fed_worker_servers
		self.fed_master = fed_master
		self.local_cluster_spec = cluster_spec
		self.dataset_configs = dataset_configs

	@property
	def host_identifier(self):
		ps = ','.join([fed_ps.endpoint for fed_ps in self.fed_parameter_servers])
		ws = ','.join([fed_ws.endpoint for fed_ws in self.fed_worker_servers])
		return "HostTFPS:[{}]|HostTFWorkers:[{}]".format(ps, ws)

	@property
	def host_training_devices(self):
		ps = ','.join([fed_ps.device_name for fed_ps in self.fed_parameter_servers])
		ws = ','.join([fed_ws.device_name for fed_ws in self.fed_worker_servers])
		return "HostTFPSDevices:[{}]|HostTFWorkersDevices:[{}]".format(ps, ws)

	@property
	def cluster_spec(self):
		ps = [fed_ps.endpoint for fed_ps in self.fed_parameter_servers]
		ws = [fed_ws.endpoint for fed_ws in self.fed_worker_servers]
		return tf.train.ClusterSpec({'ps': ps, 'worker': ws})

	def elect_local_cluster_master(self):
		# TODO define which worker will be elected as the next training sessions leader.
		#  While implementing this function, we need change the `is_leader` property of the FedWorkerServer too.
		pass

	def ps_workload_handler(self):
		# TODO define policy on how the workload is distributed across parameter servers within same host.
		pass


class RemoteServerConfigs(object):

	def __init__(self, hostname, login_username, ld_library_path, cuda_home, python_interpreter):
		self.hostname = hostname
		self.login_username = login_username
		self.ld_library_path = ld_library_path
		self.cuda_home = cuda_home
		self.python_interpreter = python_interpreter
		self.remote_pid_filepath = None


class FedHostDatasetConfigs(object):

	def __init__(self, train_dataset_mappings, validation_dataset_mappings, test_dataset_mappings):
		self.train_dataset_mappings = train_dataset_mappings
		self.validation_dataset_mappings = validation_dataset_mappings
		self.test_dataset_mappings = test_dataset_mappings


class FedTFServer(object):
	def __init__(self, endpoint, job_name, task_index, device_name,
				 tf_server, is_remote_server=False, remote_server_configs=None):
		if is_remote_server is True and not isinstance(remote_server_configs, RemoteServerConfigs):
			raise RuntimeError("When connecting to a remote server instance the `remote_server_configs` "
							   "parameter needs to be defined.")
		if is_remote_server is False and not isinstance(tf_server, tf.train.Server):
			raise TypeError("When the federation node is started at localhost as a server node, "
							"then the `tf_server` parameter must be must of type %s " % tf.train.Server)
		self.endpoint = endpoint
		self.hostname = self.endpoint.split(":")[0]
		self.port = self.endpoint.split(":")[1]
		self.job_name = job_name
		self.task_index = task_index
		self.device_name = device_name
		self.grpc_endpoint = "grpc://%s" % self.endpoint
		self.tf_server = tf_server
		self.is_remote_server = is_remote_server
		self.remote_server_configs = remote_server_configs


class FedTFParameterServer(FedTFServer):

	def __init__(self, endpoint, task_index, device_name, tf_server, is_remote_server, remote_server_configs=None):
		FedTFServer.__init__(self, endpoint, "ps", task_index, device_name, tf_server, is_remote_server,
							 remote_server_configs)


class FedWorkerServer(FedTFServer):

	def __init__(self, endpoint, task_index, gpu_id, cpu_id, device_name, tf_server, is_remote_server, is_leader,
				 fed_db, remote_server_configs=None, model_filepath=None, local_batch_size=None,
				 local_epochs_target_update=None, validation_percentage=None, validation_cycle_tombstones=None,
				 validation_cycle_loss_percentage_threshold=None):
		if not isinstance(fed_db, FedDB):
			raise TypeError("`fed_db` parameter must be of type %s" % FedDB)
		FedTFServer.__init__(self, endpoint, "worker", task_index, device_name, tf_server, is_remote_server,
							 remote_server_configs)
		self.gpu_id = gpu_id
		self.cpu_id = cpu_id
		self.is_leader = is_leader
		self.fed_db = fed_db
		self.model_filepath = model_filepath
		self.batch_size = local_batch_size
		self.target_update_epochs = local_epochs_target_update
		self.validation_proportion = validation_percentage / 100
		self.validation_cycle_tombstones = validation_cycle_tombstones
		self.validation_cycle_loss_percentage_threshold = validation_cycle_loss_percentage_threshold


class FedDB(object):
	# TODO add database connection parameters
	pass


class FedEnvironment(object):

	# Servers endpoint specifications.
	ENDPOINT_SPEC = '{}:{}'
	CPU_DEVICE_SPEC = '/job:{}/task:{}/cpu:{}'
	GPU_DEVICE_SPEC = '/job:{}/task:{}/device:GPU:{}'

	# Supported community functions.
	SUPPORTED_COMMUNITY_FUNCTIONS = ['FedAvg', 'DVWMacroWeightedF1', 'DVWMicroF1', 'DVWInvertedLoss', 'DVWInvertedMSE',
									 'DVWInvertedMAE', 'FedAnnealing']

	def __init__(self, fed_training_hosts, fed_evaluator_tf_host,
				 federation_controller_grpc_servicer_endpoint="localhost:50050",
				 federation_evaluator_grpc_servicer_endpoint="localhost:50051",
				 federation_evaluator_tensorflow_ps_endpoint="localhost:9898",
				 federation_rounds=1, synchronous_execution=True, community_function="FedAvg",
				 execution_time_in_mins=180):
		if not isinstance(fed_training_hosts, list):
			raise TypeError("The `fed_training_hosts` parameter must be of type list()")
		if not any(isinstance(fed_host, FedHost) for fed_host in fed_training_hosts):
			raise TypeError("All the federation hosts passed with the `fed_training_hosts` parameter must be of type {}"
							.format(FedHost))
		if community_function not in self.SUPPORTED_COMMUNITY_FUNCTIONS:
			raise RuntimeError("The provided community function: {} is not currently supported by the framework."
							   .format(community_function))
		self.fed_training_hosts = fed_training_hosts
		self.fed_evaluator_tf_host = fed_evaluator_tf_host
		self.federation_controller_grpc_servicer_endpoint = federation_controller_grpc_servicer_endpoint
		self.federation_evaluator_grpc_servicer_endpoint = federation_evaluator_grpc_servicer_endpoint
		self.federation_evaluator_tensorflow_ps_endpoint = federation_evaluator_tensorflow_ps_endpoint
		self.federation_rounds = federation_rounds
		self.host_partition_idx_catalog = {fed_host.host_identifier: fed_host_idx for fed_host_idx, fed_host in
										   enumerate(self.fed_training_hosts)}
		self.partition_idx_host_catalog = {partition_idx: fed_host_identifier for fed_host_identifier, partition_idx in
										   self.host_partition_idx_catalog.items()}
		self.synchronous_execution = synchronous_execution
		self.community_function = community_function
		self.execution_time_in_mins = execution_time_in_mins


	@classmethod
	def get_supported_community_functions(cls):
		return cls.SUPPORTED_COMMUNITY_FUNCTIONS


	@classmethod
	def init_multi_localhost_tf_clusters(cls, clusters_num=5, ps_servers_per_cluster=1, worker_servers_per_cluster=1,
										 federation_rounds=5, target_local_epochs=5, batch_size_per_worker=50,
										 starting_port=2222, validation_percentage=5, validation_cycle_tombstones=4,
										 validation_cycle_loss_percentage_threshold=2, execution_time_in_mins=3600,
										 community_function="FedAvg", synchronous_execution=True):

		# TODO FIX localhost gpu assignment!
		CUDA_DEVICES = os.environ.get('CUDA_VISIBLE_DEVICES', [])
		local_hostname = NetOpsUtil.get_hostname()
		federation_controller_grpc_servicer_endpoint = cls.ENDPOINT_SPEC.format(local_hostname, 50050)
		federation_evaluator_grpc_servicer_endpoint = cls.ENDPOINT_SPEC.format(local_hostname, 50051)
		CPUS = [x for x in range(0, multiprocessing.cpu_count())]

		last_gpu_idx = 0
		last_cpu_idx = 0
		port_idx = starting_port
		# Plus one for the evaluator service
		clusters_num += 1
		fed_training_hosts = list()
		fed_evaluator_host = ""
		workers_use_gpus = True if CUDA_DEVICES else False
		for c_id in range(0, clusters_num):
			cluster_gpus = [CUDA_DEVICES[x % len(CUDA_DEVICES)] for x in
							 range(last_gpu_idx, last_gpu_idx + worker_servers_per_cluster)]
			last_gpu_idx = last_gpu_idx + worker_servers_per_cluster
			cluster_cpus = [CPUS[x % len(CPUS)] for x in
							 range(last_cpu_idx, last_cpu_idx + ps_servers_per_cluster + worker_servers_per_cluster)]
			last_cpu_idx = last_cpu_idx + ps_servers_per_cluster + worker_servers_per_cluster
			fed_host = cls.init_single_localhost_tf_cluster(
				ps_servers_no=ps_servers_per_cluster, worker_servers_no=worker_servers_per_cluster,
				batch_size_per_worker=batch_size_per_worker, target_local_epochs=target_local_epochs,
				validation_percentage=validation_percentage, validation_cycle_tombstones=validation_cycle_tombstones,
				validation_cycle_loss_percentage_threshold=validation_cycle_loss_percentage_threshold,
				starting_port=port_idx, workers_use_gpus=workers_use_gpus, GPUS=cluster_gpus, CPUS=cluster_cpus)

			# Increase Port idx for next cluster
			port_idx = port_idx + ps_servers_per_cluster + worker_servers_per_cluster
			fed_host.name = "{}:{}".format(local_hostname, c_id)

			if c_id == clusters_num-1:
				fed_evaluator_host = fed_host
			else:
				fed_training_hosts.append(fed_host)

		return FedEnvironment(
			fed_training_hosts=fed_training_hosts, fed_evaluator_tf_host=fed_evaluator_host,
			federation_rounds=federation_rounds,
			federation_controller_grpc_servicer_endpoint=federation_controller_grpc_servicer_endpoint,
			federation_evaluator_grpc_servicer_endpoint=federation_evaluator_grpc_servicer_endpoint,
			federation_evaluator_tensorflow_ps_endpoint=fed_evaluator_host.fed_parameter_servers[0].endpoint,
			execution_time_in_mins=execution_time_in_mins, community_function=community_function,
			synchronous_execution=synchronous_execution)


	@classmethod
	def init_single_localhost_tf_cluster(cls, ps_servers_no=1, worker_servers_no=3, batch_size_per_worker=50,
										 validation_percentage=5, validation_cycle_tombstones=4,
										 validation_cycle_loss_percentage_threshold=2, target_local_epochs=5,
										 starting_port=2222, workers_use_gpus=False, GPUS=list(), CPUS=list()):

		""" Create tensorflow cluster and return Server instances. """
		GPUS = ','.join([str(gpu_id) for gpu_id in GPUS])
		os.environ['CUDA_VISIBLE_DEVICES'] = GPUS
		local_hostname = NetOpsUtil.get_hostname()
		if workers_use_gpus:
			ws_device = cls.GPU_DEVICE_SPEC
			gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1, allow_growth=True)
		else:
			ws_device = cls.CPU_DEVICE_SPEC
			gpu_options = None

		port_idx = starting_port
		ps_task = 0
		worker_task = 0
		ps_hosts, worker_hosts, ps_servers, worker_servers = ([] for _ in range(4))

		# Define Cluster Spec.
		for _ in range(0, ps_servers_no):
			ps_hosts.append(cls.ENDPOINT_SPEC.format(local_hostname, port_idx))
			port_idx += 1

		for _ in range(0, worker_servers_no):
			worker_hosts.append(cls.ENDPOINT_SPEC.format(local_hostname, port_idx))
			port_idx += 1

		cluster_spec = tf.train.ClusterSpec({'ps': ps_hosts, 'worker': worker_hosts})
		server_config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True,
									   inter_op_parallelism_threads=5,
									   intra_op_parallelism_threads=5)

		# Start Parameter Servers.
		for pid in range(0, ps_servers_no):
			cpuid = CPUS[pid]
			metis_logger.info(msg='Starting Tensorflow Parameter Server @ %s with CPU:%s' % (ps_hosts[pid], cpuid))
			ps_device_name = cls.CPU_DEVICE_SPEC.format('ps', ps_task, cpuid)
			ps_server = tf.train.Server(server_or_cluster_def=cluster_spec, job_name='ps', task_index=ps_task,
										config=server_config, start=True)
			fedps = FedTFParameterServer(endpoint=ps_hosts[pid], task_index=ps_task, device_name=ps_device_name,
										 tf_server=ps_server, is_remote_server=False)
			ps_servers.append(fedps)
			ps_task += 1

		# Start Worker Servers.
		for wid in range(0, worker_servers_no):
			w_gpuid = None
			w_cpuid = None
			if workers_use_gpus:
				w_gpuid = GPUS[wid]
				metis_logger.info('Starting Tensorflow Worker Server @ %s with GPU:%s' % (worker_hosts[wid], w_gpuid))
				worker_device_name = ws_device.format('worker', worker_task, w_gpuid)
			else:
				# Round-Robin assignment of CPUs continuing from where the Parameter Servers were left.
				CPU_IDX = (ps_servers_no + wid) % len(CPUS)
				w_cpuid = CPUS[CPU_IDX]
				metis_logger.info('Starting Tensorflow Worker Server @ %s with CPU:%s' % (worker_hosts[wid], w_cpuid))
				worker_device_name = ws_device.format('worker', worker_task, w_cpuid)

			worker_server = tf.train.Server(server_or_cluster_def=cluster_spec, job_name='worker',
											task_index=worker_task, config=server_config, start=True)
			w_batch_size = batch_size_per_worker
			fedworker = FedWorkerServer(
				endpoint=worker_hosts[wid], task_index=worker_task, device_name=worker_device_name,
				tf_server=worker_server, is_remote_server=False, validation_percentage=validation_percentage,
				validation_cycle_tombstones=validation_cycle_tombstones,
				validation_cycle_loss_percentage_threshold=validation_cycle_loss_percentage_threshold,
				is_leader= worker_task == 0, fed_db=FedDB(), local_batch_size=w_batch_size, model_filepath=None,
				local_epochs_target_update=target_local_epochs, gpu_id=w_gpuid, cpu_id=w_cpuid)
			worker_servers.append(fedworker)
			worker_task += 1

		master_worker= worker_servers[0]
		metis_logger.info(msg='The elected master worker of this session is: %s' % master_worker.grpc_endpoint)

		# Create a federation host object.
		fed_host = FedHost(name="localhost", fed_parameter_servers=ps_servers, fed_worker_servers=worker_servers,
						   fed_master=master_worker, cluster_spec=cluster_spec)

		return fed_host


	@classmethod
	def tf_federated_cluster_from_yaml(cls, cluster_specs_file, init_cluster=False):

		# Read YAML Configs.
		fstream = open(cluster_specs_file).read()
		loaded_stream = yaml.load(fstream, Loader=yaml.SafeLoader)

		federation_environment = loaded_stream.get('FederationRuntimeEnvironment')
		federation_rounds = federation_environment.get('Rounds')
		synchronous_execution = federation_environment.get('SynchronousExecution')
		community_function = federation_environment.get('CommunityFunction')
		execution_time_in_secs = federation_environment.get('ExecutionTimeMins')

		federation_controller = federation_environment.get('FederationController')
		federation_controller_servicer = federation_controller.get('GRPCServicer')
		federation_controller_hostname = federation_controller_servicer.get('hostname/ip')
		federation_controller_grpc_port = federation_controller_servicer.get('port')
		federation_controller_grpc_endpoint = cls.ENDPOINT_SPEC.format(federation_controller_hostname,
																   federation_controller_grpc_port)

		federation_evaluator = federation_environment.get('FederationEvaluator')
		federation_evaluator_servicer = federation_evaluator.get('GRPCServicer')
		federation_evaluator_hostname = federation_evaluator_servicer.get('hostname/ip')
		federation_evaluator_grpc_port = federation_evaluator_servicer.get('port')
		federation_evaluator_grpc_endpoint = cls.ENDPOINT_SPEC.format(federation_evaluator_hostname,
																  federation_evaluator_grpc_port)
		federation_evaluator_tf_host = federation_evaluator.get('TFHosts')

		tf_cluster = federation_environment.get('TensorflowCluster')
		tf_hosts = tf_cluster.get('TFHosts')
		tf_hosts.extend(federation_evaluator_tf_host)

		# Define TF Parameter and Worker Servers.
		fed_training_tf_hosts = []
		for tf_host in tf_hosts:

			tf_host_id = tf_host.get('HostID')
			tf_host_parameter_servers = tf_host.get('ParameterServers')
			tf_host_worker_servers = tf_host.get('WorkerServers')
			tf_host_remote_configs = tf_host.get('RemoteConfigs')
			tf_host_dataset_configs = tf_host.get('DatasetConfigs')
			tf_host_fed_pss, tf_host_fed_wss = ([] for _ in range(2))
			master_fedworker = None

			ps_hosts = [cls.ENDPOINT_SPEC.format(ps['hostname/ip'], ps['port']) for ps in tf_host_parameter_servers]
			ws_hosts = [cls.ENDPOINT_SPEC.format(ws['hostname/ip'], ws['port']) for ws in tf_host_worker_servers]
			host_cluster_spec = tf.train.ClusterSpec({'ps': ps_hosts, 'worker': ws_hosts})
			host_cluster_spec_as_dict = host_cluster_spec.as_dict()
			host_cluster_spec_as_dict_pickled_string = str(pickle.dumps(host_cluster_spec_as_dict))

			remote_server_configs = RemoteServerConfigs(
				hostname=tf_host_remote_configs.get("hostname/ip"),
				login_username=tf_host_remote_configs.get("username"),
				ld_library_path=tf_host_remote_configs.get("ld_library_path"),
				cuda_home=tf_host_remote_configs.get("cuda_home"),
				python_interpreter=tf_host_remote_configs.get("python_interpreter"))

			dataset_configs = FedHostDatasetConfigs(None, None, None)
			if tf_host_dataset_configs is not None:
				dataset_configs = FedHostDatasetConfigs(
					train_dataset_mappings=tf_host_dataset_configs.get('train_dataset_mappings'),
					validation_dataset_mappings=tf_host_dataset_configs.get('validation_dataset_mappings'),
					test_dataset_mappings=tf_host_dataset_configs.get('test_dataset_mappings'))

			# Parameter Servers Setup.
			for pid, ps in enumerate(tf_host_parameter_servers):
				ps_hostname = ps['hostname/ip']
				ps_device_name = cls.CPU_DEVICE_SPEC.format('ps', ps['taskid'], ps['cpuid'])
				ps_taskid = ps['taskid']
				ps_port = ps['port']
				ps_endpoint = cls.ENDPOINT_SPEC.format(ps_hostname, ps_port)

				if init_cluster:
					metis_logger.info(msg='Starting Tensorflow Parameter Server: %s' % ps_endpoint)
					ps_remote_server_configs = copy.deepcopy(remote_server_configs)
					fedps = FedTFParameterServer(endpoint=ps_endpoint, task_index=ps_taskid, device_name=ps_device_name,
												 tf_server=None, is_remote_server=True,
												 remote_server_configs=ps_remote_server_configs)
					remote_pid_filepath = RemoteTFServerUtils.init_remote_tf_server_process(
						fed_tf_server=fedps, cluster_spec=host_cluster_spec_as_dict_pickled_string,
						is_worker_node=False)
					fedps.remote_server_configs.remote_pid_filepath = remote_pid_filepath
					metis_logger.info(msg='Started Tensorflow Parameter Server: %s' % ps_endpoint)
				else:
					metis_logger.info(msg='Checking Remote Connection to TF Parameter Server: %s, Device Name: %s' %
										  (ps_endpoint, ps_device_name))
					is_host_listening = NetOpsUtil.is_endpoint_listening(host=ps_hostname, port=ps_port)
					if not is_host_listening:
						raise RuntimeError("Endpoint %s:%s is down. Resurrect it from the dead." %
										   (ps_hostname, ps_port))
					fedps = FedTFParameterServer(endpoint=ps_endpoint, task_index=ps_taskid, device_name=ps_device_name,
												 tf_server=None, is_remote_server=True,
												 remote_server_configs=remote_server_configs)
				tf_host_fed_pss.append(fedps)

			# Worker Servers Setup
			for wid, ws in enumerate(tf_host_worker_servers):
				if ws['cpuid'] is not None:
					ws_device_name = cls.CPU_DEVICE_SPEC.format('worker', ws['taskid'], ws['cpuid'])
				else:
					ws_device_name = cls.GPU_DEVICE_SPEC.format('worker', ws['taskid'], ws['gpuid'])

				w_hostname = ws['hostname/ip']
				w_task_id = ws['taskid']
				w_port = ws['port']
				w_gpu_id = ws['gpuid'] if len(str(ws['gpuid'])) > 0 else None
				w_cpu_id = ws['cpuid'] if len(str(ws['cpuid'])) > 0 else None
				w_endpoint = cls.ENDPOINT_SPEC.format(w_hostname, w_port)
				w_is_leader = ws['is_host_training_leader']
				w_batch_size = ws['local_batch_size'] if 'local_batch_size' in ws else 32
				w_target_update_epochs = ws['local_epochs_target_update'] if 'local_epochs_target_update' in ws else 5
				w_validation_percentage = ws['validation_percentage'] if 'validation_percentage' in ws else 0
				w_validation_cycle_tombstones = ws['validation_cycle_tombstones'] \
					if 'validation_cycle_tombstones' in ws else 0
				w_validation_cycle_loss_threshold = ws['validation_cycle_loss_percentage_threshold'] \
					if 'validation_cycle_loss_percentage_threshold' in ws else 0

				if init_cluster:
					metis_logger.info('Starting Tensorflow Worker Server: %s' % w_endpoint)
					w_remote_server_configs = copy.deepcopy(remote_server_configs)
					fedworker = FedWorkerServer(
						endpoint=w_endpoint, task_index=w_task_id, gpu_id=w_gpu_id, cpu_id=w_cpu_id,
						device_name=ws_device_name, tf_server=None, is_remote_server=True, is_leader=w_is_leader,
						fed_db=FedDB(), local_batch_size=w_batch_size,
						local_epochs_target_update=w_target_update_epochs, model_filepath=None,
						remote_server_configs=w_remote_server_configs, validation_percentage=w_validation_percentage,
						validation_cycle_tombstones=w_validation_cycle_tombstones,
						validation_cycle_loss_percentage_threshold=w_validation_cycle_loss_threshold)
					remote_pid_filepath = RemoteTFServerUtils.init_remote_tf_server_process(
						fed_tf_server=fedworker, cluster_spec=host_cluster_spec_as_dict_pickled_string,
						is_worker_node=True)
					fedworker.remote_server_configs.remote_pid_filepath = remote_pid_filepath
					metis_logger.info('Started Tensorflow Worker Server: %s' % w_endpoint)
				else:
					metis_logger.info(msg='Checking Remote Connection to TF Worker Server: %s, Device Name: %s' %
										  (w_endpoint, ws_device_name))
					is_host_listening = NetOpsUtil.is_endpoint_listening(host=w_hostname, port=w_port)
					if not is_host_listening:
						raise RuntimeError("Endpoint %s:%s is down. Raise it from the dead." % (w_hostname, w_port))
					fedworker = FedWorkerServer(
						endpoint=w_endpoint, task_index=w_task_id, gpu_id=w_gpu_id, cpu_id=w_cpu_id,
						device_name=ws_device_name, tf_server=None, is_remote_server=True, is_leader=w_is_leader,
						fed_db=FedDB(), local_batch_size=w_batch_size,
						local_epochs_target_update=w_target_update_epochs, model_filepath=None,
						remote_server_configs=remote_server_configs, validation_percentage=w_validation_percentage,
						validation_cycle_tombstones=w_validation_cycle_tombstones,
						validation_cycle_loss_percentage_threshold=w_validation_cycle_loss_threshold)
				tf_host_fed_wss.append(fedworker)

				if w_is_leader:
					master_fedworker = fedworker

			if master_fedworker is None:
				metis_logger.info(msg='Since no master worker was defined for host: %s, the elected master worker is: '
									  '%s' % (tf_host_id, tf_host_fed_wss[0].grpc_endpoint))
				master_fedworker = tf_host_fed_wss[0]

			metis_logger.info(msg=host_cluster_spec)
			fed_training_tf_hosts.append(FedHost(name=tf_host_id, fed_parameter_servers=tf_host_fed_pss,
												 fed_worker_servers=tf_host_fed_wss, fed_master=master_fedworker,
												 cluster_spec=host_cluster_spec, dataset_configs=dataset_configs))

		# Remove Evaluator from Training Hosts
		fed_evaluator_tf_host = fed_training_tf_hosts[-1]
		fed_training_tf_hosts = fed_training_tf_hosts[:-1]

		return FedEnvironment(
			fed_training_hosts=fed_training_tf_hosts, fed_evaluator_tf_host=fed_evaluator_tf_host,
			federation_controller_grpc_servicer_endpoint=federation_controller_grpc_endpoint,
			federation_evaluator_grpc_servicer_endpoint=federation_evaluator_grpc_endpoint,
			federation_evaluator_tensorflow_ps_endpoint=fed_evaluator_tf_host.fed_parameter_servers[0].endpoint,
			federation_rounds=federation_rounds, synchronous_execution=synchronous_execution,
			community_function=community_function, execution_time_in_mins=execution_time_in_secs)


	def shutdown_yaml_created_tf_federated_cluster(self):

		hosts_to_shutdown = list()
		# FedTraining Hosts is a list of FedTFServer objects
		hosts_to_shutdown.extend(self.fed_training_hosts)
		# FedEvaluator Host is a single FedTFServer object
		hosts_to_shutdown.append(self.fed_evaluator_tf_host)

		# Shutdown All Hosts (Evaluator, Training)
		for fed_host in hosts_to_shutdown:
			for fed_ps_host in fed_host.fed_parameter_servers:
				RemoteTFServerUtils.shutdown_remote_tf_server_process(fed_tf_server=fed_ps_host)
				metis_logger.info(msg="Tensorflow Server:{} Shutdown".format(fed_ps_host.endpoint))
			for fed_worker_host in fed_host.fed_worker_servers:
				RemoteTFServerUtils.shutdown_remote_tf_server_process(fed_tf_server=fed_worker_host)
				metis_logger.info(msg="Tensorflow Server:{} Shutdown".format(fed_worker_host.endpoint))
