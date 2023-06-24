
import os
import cloudpickle
import shutil
import tarfile

from typing import Callable, Dict
from fabric import Connection

from metisfl.driver.utils import create_server_entity
from metisfl.models.model_wrapper import MetisModel
from metisfl.utils import fedenv_parser
from metisfl.utils.metis_logger import MetisLogger

MODEL_SAVE_DIR = "model_definition"
DATASET_RECEIPE_FILENAMES = {
    "train": "model_train_dataset_ops.pkl",
    "validation": "model_validation_dataset_ops.pkl",
    "test": "model_test_dataset_ops.pkl"
}
REMOTE_METIS_CONTROLLER_PATH = "/tmp/metis/controller"
REMOTE_METIS_LEARNER_PATH = "/tmp/metis/workdir_learner_{}"

class DriverSession:

    def __init__(self,
                 fed_env: fedenv_parser.FederationEnvironment,
                 model: MetisModel,
                 working_dir: str,
                 dataset_recipe_fns: Dict[str, Callable]):
        assert "train" in dataset_recipe_fns, "Train dataset recipe function is required."

        self.federation_environment = fed_env
        self.model = model
        self.working_dir = working_dir
        
        self.prepare_working_dir(working_dir)
        self.dataset_receipe_fps= self.save_dataset_receipe(dataset_recipe_fns)
        self.model_definition_tar_fp = self.save_initial_model(model)

    def prepare_working_dir(self) -> None:
        if os.path.exists(self.working_dir):
            shutil.rmtree(self.working_dir)
        self._save_model_dir = os.path.join(self.working_dir, MODEL_SAVE_DIR)
        os.makedirs(self._save_model_dir)

    def save_dataset_receipes(self, dataset_recipe_fns) -> str:
        dataset_receipe_fps = dict()
        for key, dataset_recipe_fn in dataset_recipe_fns.items():
            if dataset_recipe_fn:
                dataset_pkl = os.path.join(self.working_dir, DATASET_RECEIPE_FILENAMES[key])
                cloudpickle.dump(obj=dataset_recipe_fn, file=open(dataset_pkl, "wb+"))
                dataset_receipe_fps[key] = dataset_pkl
        return dataset_receipe_fps
            
    def save_initial_model(self, model):        
        self._model_weights_descriptor = model.get_weights_descriptor()
        model.save(self._save_model_dir)        
        return self._make_tarfile(
            output_filename=self._save_model_dir_name,
            source_dir=self._save_model_dir
        )
        
    def _make_tarfile(self, output_filename, source_dir):
        output_dir = os.path.abspath(os.path.join(source_dir, os.pardir))
        output_filepath = os.path.join(output_dir, "{}.tar.gz".format(output_filename))
        with tarfile.open(output_filepath, "w:gz") as tar:
            tar.add(source_dir, arcname=os.path.basename(source_dir))
        return output_filepath

    def init_controller(self):
        fabric_connection_config = self.federation_environment.controller \
            .connection_configs.get_fabric_connection_config()
        connection = Connection(**fabric_connection_config)
        # We do not use asynchronous or disown, since we want the remote subprocess to return standard (error) output.
        # see also, https://github.com/pyinvoke/invoke/blob/master/invoke/runners.py#L109

        # Delete existing directory if it exists, then recreate it.
        connection.run("rm -rf {}".format(REMOTE_METIS_CONTROLLER_PATH))
        connection.run("mkdir -p {}".format(REMOTE_METIS_CONTROLLER_PATH))
        remote_on_login = self.federation_environment.controller.connection_configs.on_login
        if len(remote_on_login) > 0 and remote_on_login[-1] == ";":
            remote_on_login = remote_on_login[:-1]

        # @stripeli this assumes that metisfl is installed in the remote host; make it more robust
        init_cmd = "{} && cd {} && {}".format(
            remote_on_login,
            self.federation_environment.controller.project_home,
            self._init_controller_cmd())
        MetisLogger.info("Running init cmd to controller host: {}".format(init_cmd))
        connection.run(init_cmd)
        connection.close()
        return

    def init_learner(self, learner_instance, controller_instance):
        assert self.controller_server_entity_pb is not None, "Controller server entity is not initialized. Musrt call init_controller() first."
        
        fabric_connection_config = \
            learner_instance.connection_configs.get_fabric_connection_config()
        connection = Connection(**fabric_connection_config)
        
        # We do not use asynchronous or disown, since we want the remote subprocess to return standard (error) output.
        remote_metis_path = REMOTE_METIS_LEARNER_PATH.format(learner_instance.grpc_servicer.port)
        
        # Delete existing directory if it exists, then recreate it.
        connection.run("rm -rf {}".format(remote_metis_path))
        connection.run("mkdir -p {}".format(remote_metis_path))

        MetisLogger.info("Copying model definition and dataset recipe files at learner: {}"
                         .format(learner_instance.learner_id))
        for _, filepath in self.dataset_receipe_fps.items():
            connection.put(filepath, remote_metis_path) if filepath else None
        if self.model_definition_tar_fp:
            connection.put(self.model_definition_tar_fp, remote_metis_path)

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
    
    def _init_controller_cmd(self):
        # To create the controller grpc server entity, we need the hostname to which the server
        # will bind to and the port of the grpc servicer defined in the initial configuration file.
        # Controller is a subtype of RemoteHost instance, hence we pass it as is.        
        config_attrs = [
            "controller_server_entity",
            "global_model_config", 
            "communication_protocol", 
            "local_model_config", 
            "model_store_config"
        ]
        self.controller_server_entity_pb = create_server_entity(
            remote_host_instance=self.federation_environment.controller,
            initialization_entity=True)
        config = {}
        config["controller_server_entity"] = self.controller_server_entity_pb
        for attr in config_attrs:
            config[attr] = getattr(self.federation_environment, attr).to_proto()
        return self._init_cmd("controller", config)

    def _init_learner_cmd(self, learner_instance):
        remote_metis_path = REMOTE_METIS_LEARNER_PATH.format(learner_instance.grpc_servicer.port)
        remote_metis_model_path = os.path.join(remote_metis_path, self._save_model_dir_name)
        remote_dataset_recipe_fps = dict()
        for filename, filepath in self.dataset_receipe_fps.items():
            remote_dataset_recipe_fps[filename] = os.path.join(remote_metis_path, filename) if filepath else None

        learner_server_entity_pb = create_server_entity(
            remote_host_instance=learner_instance,
            initialization_entity=True)     
        
        config = {
            "learner_server_entity_protobuff_serialized_hexadecimal": learner_server_entity_pb.SerializeToString().hex(),
            "controller_server_entity_protobuff_serialized_hexadecimal": self.controller_server_entity_pb.SerializeToString().hex(),
            "he_scheme_protobuff_serialized_hexadecimal": self.federation_environment.homomorphic_encryption.to_proto().SerializeToString().hex(),
            "model_dir": remote_metis_model_path,
            "train_dataset": learner_instance.dataset_configs.train_dataset_path,
            "validation_dataset": learner_instance.dataset_configs.validation_dataset_path,
            "test_dataset": learner_instance.dataset_configs.test_dataset_path,
            "train_dataset_recipe": remote_dataset_recipe_fps["train_dataset_recipe"],
            "validation_dataset_recipe": remote_dataset_recipe_fps["validation_dataset_recipe"],
            "test_dataset_recipe": remote_dataset_recipe_fps["test_dataset_recipe"],
            "neural_engine": self.model.get_neural_engine()
        }
        return self._init_cmd("learner", config)
    
    def _get_cmd(self, entity, config):
        cmd = "python3 -m metisfl.{} ".format(entity)
        for key, value in config.items():
            cmd += "--{}={} ".format(key, value)
        return cmd

