
from metisfl.driver.driver_session_base import DriverSessionBase

from fabric import Connection
from metisfl.utils.metis_logger import MetisLogger


class DriverSession(DriverSessionBase):

    def __init__(self,
                 fed_env,
                 model,
                 train_dataset_recipe_fn,
                 validation_dataset_recipe_fn=None,
                 test_dataset_recipe_fn=None):
        super(DriverSession, self).__init__(
            fed_env,
            model,
            train_dataset_recipe_fn,
            validation_dataset_recipe_fn,
            test_dataset_recipe_fn)

    def _init_controller(self):
        fabric_connection_config = self.federation_environment.controller \
            .connection_configs.get_fabric_connection_config()
        connection = Connection(**fabric_connection_config)
        # We do not use asynchronous or disown, since we want the remote subprocess to return standard (error) output.
        # see also, https://github.com/pyinvoke/invoke/blob/master/invoke/runners.py#L109
        remote_metis_path = "/tmp/metis/controller"
        # Delete existing directory if it exists, then recreate it.
        connection.run("rm -rf {}".format(remote_metis_path))
        connection.run("mkdir -p {}".format(remote_metis_path))
        remote_on_login = self.federation_environment.controller.connection_configs.on_login
        if len(remote_on_login) > 0 and remote_on_login[-1] == ";":
            remote_on_login = remote_on_login[:-1]

        init_cmd = "{} && cd {} && {}".format(
            remote_on_login,
            self.federation_environment.controller.project_home,
            self._init_controller_cmd())
        MetisLogger.info("Running init cmd to controller host: {}".format(init_cmd))
        connection.run(init_cmd)
        connection.close()
        return

    def _init_learner(self, learner_instance, controller_instance):
        fabric_connection_config = \
            learner_instance.connection_configs.get_fabric_connection_config()
        connection = Connection(**fabric_connection_config)
        # We do not use asynchronous or disown, since we want the remote subprocess to return standard (error) output.
        remote_metis_path = "/tmp/metis/workdir_learner_{}".format(learner_instance.grpc_servicer.port)
        # Delete existing directory if it exists, then recreate it.
        connection.run("rm -rf {}".format(remote_metis_path))
        connection.run("mkdir -p {}".format(remote_metis_path))
        # Place/Copy model definition and dataset recipe files from the driver to the remote host.
        # Model definition ship .gz file and decompress it.
        MetisLogger.info("Copying model definition and dataset recipe files at learner: {}"
                         .format(learner_instance.learner_id))

        if self.model_definition_tar_fp:
            connection.put(self.model_definition_tar_fp, remote_metis_path)

        if self.train_dataset_recipe_fp:
            connection.put(self.train_dataset_recipe_fp, remote_metis_path)

        if self.validation_dataset_recipe_fp:
            connection.put(self.validation_dataset_recipe_fp, remote_metis_path)

        if self.test_dataset_recipe_fp:
            connection.put(self.test_dataset_recipe_fp, remote_metis_path)

        # Fabric runs every command on a non-interactive mode and therefore the $PATH that might be set for a
        # running user might not be visible while running the command. A workaround is to always
        # source the respective bash_environment files.
        cuda_devices_str = ""
        # Exporting this environmental variable works for both Tensorflow/Keras and PyTorch.
        if learner_instance.cuda_devices is not None and len(learner_instance.cuda_devices) > 0:
            cuda_devices_str = "export CUDA_VISIBLE_DEVICES=\"{}\" " \
                .format(",".join([str(c) for c in learner_instance.cuda_devices]))
        remote_on_login = learner_instance.connection_configs.on_login
        if len(remote_on_login) > 0 and remote_on_login[-1] == ";":
            remote_on_login = remote_on_login[:-1]

        # Un-taring model definition zipped file.
        MetisLogger.info("Un-taring model definition files at learner: {}"
                         .format(learner_instance.learner_id))
        connection.run("cd {}; tar -xvzf {}".format(
            remote_metis_path,
            self.model_definition_tar_fp))

        init_cmd = "{} && {} && cd {} && {}".format(
            remote_on_login,
            cuda_devices_str,
            learner_instance.project_home,
            self._init_learner_cmd(learner_instance, controller_instance))
        MetisLogger.info("Running init cmd to learner host: {}".format(init_cmd))
        connection.run(init_cmd)
        connection.close()
        return

    def shutdown_federation(self):
        self._shutdown()

