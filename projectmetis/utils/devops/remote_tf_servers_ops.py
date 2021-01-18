import os
import shutil

import federation.fed_cluster_env as fed_cluster_env

from fabric import Connection


class RemoteTFServerUtils(object):

	current_directory = os.path.dirname(os.path.realpath(__file__))

	@classmethod
	def get_remote_server_connection(cls,
									 remote_server_hostname,
									 remote_server_username):

		connection = Connection(host=remote_server_hostname,
								user=remote_server_username,
								port=22,
								connect_kwargs={
									"key_filename": "/nas/home/stripeli/.ssh/id_rsa",
									"passphrase": "stripeli"
								},
								inline_ssh_env=True)
		return connection


	@classmethod
	def init_remote_tf_server_process(cls,
									  fed_tf_server,
									  cluster_spec,
									  remote_working_dir="/tmp/metis_project",
									  is_worker_node=False):

		assert isinstance(fed_tf_server, fed_cluster_env.FedTFServer)

		remote_tf_server_job_name = fed_tf_server.job_name
		remote_tf_server_task_index = fed_tf_server.task_index
		remote_tf_server_port = fed_tf_server.port
		logging_filename = "{}{}.out".format(remote_tf_server_job_name, remote_tf_server_port)
		pid_filename = "{}{}.pid".format(remote_tf_server_job_name, remote_tf_server_port)

		remote_tf_servers_logs_dir = os.path.join(remote_working_dir, "tf_servers_logs")
		remote_tf_servers_pids_dir = os.path.join(remote_working_dir, "tf_servers_pids")
		remote_server_hostname = fed_tf_server.remote_server_configs.hostname
		remote_server_username = fed_tf_server.remote_server_configs.login_username
		connection = cls.get_remote_server_connection(remote_server_hostname, remote_server_username)

		remote_server_ld_library_path = fed_tf_server.remote_server_configs.ld_library_path
		remote_server_cuda_home = fed_tf_server.remote_server_configs.cuda_home
		remote_server_python_interpreter = fed_tf_server.remote_server_configs.python_interpreter

		is_remote_server_gpu_worker = False
		remote_worker_gpu_id = -1
		gpu_memory_fraction = 0
		if is_worker_node:
			assert isinstance(fed_tf_server, fed_cluster_env.FedWorkerServer)
			if fed_tf_server.gpu_id is not None:
				is_remote_server_gpu_worker = True
				remote_worker_gpu_id = fed_tf_server.gpu_id
				gpu_memory_fraction = 1

		# Python init script.
		local_tf_servers_python_init_script_name = "init_tf_server.py"
		local_tf_servers_python_init_script = os.path.join(cls.current_directory,
														   local_tf_servers_python_init_script_name)
		tf_server_init_python_script_name = "init_tf_server_{}_port{}.py".format(
			str(remote_server_hostname), str(remote_tf_server_port))
		local_python_tf_server_script = os.path.join(cls.current_directory, tf_server_init_python_script_name)
		shutil.copy(local_tf_servers_python_init_script, local_python_tf_server_script)
		remote_tf_server_python_init_script = os.path.join(remote_working_dir,
														   tf_server_init_python_script_name)
		remote_tf_server_log_file = os.path.join(remote_tf_servers_logs_dir, logging_filename)
		remote_tf_server_pid_file = os.path.join(remote_tf_servers_pids_dir, pid_filename)

		# Bash Init Script. Create an init script for current server in the local scripts directory
		bash_init_script_name = "tf_server_{}_{}.sh".format(remote_server_hostname, fed_tf_server.port)
		local_tf_servers_bash_init_script = os.path.join(cls.current_directory, "../../scripts/{}")\
			.format(bash_init_script_name)
		remote_tf_server_bash_init_script = os.path.join(remote_working_dir, bash_init_script_name)

		with open(local_tf_servers_bash_init_script, "w+") as tf_server_fout:
			print("#! /bin/bash -l", file=tf_server_fout)
			print("export GRPC_VERBOSITY=DEBUG", file=tf_server_fout)
			print("export CUDA_VISIBLE_DEVICES=\"{}\"".format(remote_worker_gpu_id), file=tf_server_fout)

			if is_remote_server_gpu_worker:
				print("export LD_LIBRARY_PATH={}".format(remote_server_ld_library_path), file=tf_server_fout)
				print("export CUDA_HOME={}".format(remote_server_cuda_home), file=tf_server_fout)

			print("export TF_SERVER_JOB_NAME=\"{}\"".format(str(remote_tf_server_job_name)), file=tf_server_fout)
			print("export TF_SERVER_TASK_INDEX={}".format(remote_tf_server_task_index), file=tf_server_fout)
			print("export TF_SERVER_CLUSTER_SPEC=\"{}\"".format(str(cluster_spec)), file=tf_server_fout)
			if is_remote_server_gpu_worker:
				print("export TF_SERVER_INTRA_OP_THREADS={}".format(3), file=tf_server_fout)
				print("export TF_SERVER_INTER_OP_THREADS={}".format(3), file=tf_server_fout)
			else:
				print("export TF_SERVER_INTRA_OP_THREADS={}".format(2), file=tf_server_fout)
				print("export TF_SERVER_INTER_OP_THREADS={}".format(2), file=tf_server_fout)

			print("export TF_SERVER_IS_GPU={}".format(str(is_remote_server_gpu_worker)), file=tf_server_fout)
			print("export TF_SERVER_GPU_MEMORY_FRACTION={}".format(gpu_memory_fraction), file=tf_server_fout)

			tf_server_exec_command = "nohup %s %s " \
									 "--job_name ${TF_SERVER_JOB_NAME} " \
									 "--task_index ${TF_SERVER_TASK_INDEX} " \
									 "--cluster_spec ${TF_SERVER_CLUSTER_SPEC} " \
									 "--intra_op_threads ${TF_SERVER_INTRA_OP_THREADS} " \
									 "--inter_op_threads ${TF_SERVER_INTER_OP_THREADS} " \
									 "--is_gpu ${TF_SERVER_IS_GPU} " \
									 "--gpu_memory_fraction ${TF_SERVER_GPU_MEMORY_FRACTION} > %s 2>&1 &" % \
									 (remote_server_python_interpreter, remote_tf_server_python_init_script,
									  remote_tf_server_log_file)
			print(tf_server_exec_command, file=tf_server_fout)
			print("echo $! > {}".format(remote_tf_server_pid_file), file=tf_server_fout)

		# Create remote directories if they do not exist.
		connection.run('mkdir -p {}'.format(remote_working_dir), hide=True)
		connection.run('mkdir -p {}'.format(remote_tf_servers_logs_dir), hide=True)
		connection.run('mkdir -p {}'.format(remote_tf_servers_pids_dir), hide=True)

		# Copy local scripts to remote server.
		connection.put(local=local_python_tf_server_script,
					   remote=remote_working_dir)
		connection.put(local=local_tf_servers_bash_init_script,
					   remote=remote_working_dir)

		# Change to executable mode and run the script.
		connection.run('chmod +x {}'.format(remote_tf_server_bash_init_script), hide=True)
		connection.run('{}'.format(remote_tf_server_bash_init_script), hide=True)

		# Add some delay to let processes to be spawned.
		connection.run('sleep 1s'.format(remote_tf_server_bash_init_script), hide=True)

		# Delete remote files after initialization.
		connection.run('rm {}'.format(remote_tf_server_bash_init_script), hide=True)
		connection.run('rm {}'.format(remote_tf_server_python_init_script), hide=True)

		# Delete local files after initialization.
		os.remove(local_tf_servers_bash_init_script)
		os.remove(local_python_tf_server_script)

		print("Finish Spawning TF Server @ {}:{}".format(
			remote_server_hostname, remote_tf_server_port
		))
		connection.close()

		return remote_tf_server_pid_file


	@classmethod
	def shutdown_remote_tf_server_process(cls, fed_tf_server):

		assert isinstance(fed_tf_server, fed_cluster_env.FedTFServer)

		remote_server_hostname = fed_tf_server.remote_server_configs.hostname
		remote_server_username = fed_tf_server.remote_server_configs.login_username
		remote_server_pid_file = fed_tf_server.remote_server_configs.remote_pid_filepath

		connection = cls.get_remote_server_connection(remote_server_hostname, remote_server_username)
		connection.run('kill -9 `cat {}`'.format(remote_server_pid_file), hide=True)
		connection.run('rm {}'.format(remote_server_pid_file), hide=True)

		# Add some delay to allow processes to be killed
		# TODO Following might fail if remote server does not support decimal points in sleep.
		connection.run('sleep 0.1s')

		connection.close()
