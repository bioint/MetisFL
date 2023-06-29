
import os
import shutil
import tarfile
from typing import Callable

import cloudpickle
from fabric import Connection

from metisfl.models.model_wrapper import MetisModel
from metisfl.proto.metis_pb2 import ServerEntity
from metisfl.utils import fedenv_parser
from metisfl.utils.metis_logger import MetisLogger

from . import constants


class DriverInitializer:

    def __init__(self,
                 dataset_recipe_fns: dict[str, Callable],
                 fed_env: fedenv_parser.FederationEnvironment,
                 controller_server_entity_pb: ServerEntity,
                 learner_server_entities_pb: list[ServerEntity],
                 model: MetisModel,
                 working_dir: str):
        assert "train" in dataset_recipe_fns, "Train dataset recipe function is required."

        self._federation_environment = fed_env
        self._model = model
        self._working_dir = working_dir
        
        self._controller_server_entity_pb = controller_server_entity_pb
        self._learner_server_entities_pb = learner_server_entities_pb
        
        self._prepare_working_dir()
        self._dataset_receipe_fps= self._save_dataset_receipes(dataset_recipe_fns)
        self._model_definition_tar_fp = self._save_initial_model(model)
        

    def _prepare_working_dir(self) -> None:
        if os.path.exists(self._working_dir):
            shutil.rmtree(self._working_dir)
        self._save_model_dir = os.path.join(self._working_dir, constants.MODEL_SAVE_DIR_NAME)
        os.makedirs(self._save_model_dir)

    def _save_dataset_receipes(self, dataset_recipe_fns) -> str:
        dataset_receipe_fps = dict()
        for key, dataset_recipe_fn in dataset_recipe_fns.items():
            if dataset_recipe_fn:
                dataset_pkl = os.path.join(self._working_dir, constants.DATASET_RECEIPE_FILENAMES[key])
                cloudpickle.dump(obj=dataset_recipe_fn, file=open(dataset_pkl, "wb+"))
                dataset_receipe_fps[key] = dataset_pkl
        return dataset_receipe_fps
            
    def _save_initial_model(self, model):        
        self._model_weights_descriptor = model.get_weights_descriptor()
        model.save(self._save_model_dir)        
        return self._make_tarfile(
            output_filename=constants.MODEL_SAVE_DIR_NAME,
            source_dir=self._save_model_dir
        )
        
    def _make_tarfile(self, output_filename, source_dir):
        output_dir = os.path.abspath(os.path.join(source_dir, os.pardir))
        output_filepath = os.path.join(output_dir, "{}.tar.gz".format(output_filename))
        with tarfile.open(output_filepath, "w:gz") as tar:
            tar.add(source_dir, arcname=os.path.basename(source_dir))
        return output_filepath

    def init_controller(self):
        fabric_connection_config = self._federation_environment.controller \
            .connection_configs.get_fabric_connection_config()
        connection = Connection(**fabric_connection_config)
        # We do not use asynchronous or disown, since we want the remote subprocess to return standard (error) output.
        # see also, https://github.com/pyinvoke/invoke/blob/master/invoke/runners.py#L109

        connection.run("rm -rf {}".format(constants.REMOTE_METIS_CONTROLLER_PATH))
        connection.run("mkdir -p {}".format(constants.REMOTE_METIS_CONTROLLER_PATH))
        remote_on_login = self._federation_environment.controller.connection_configs.on_login
        if len(remote_on_login) > 0 and remote_on_login[-1] == ";":
            remote_on_login = remote_on_login[:-1]            
        MetisLogger.info("Copying model definition and dataset recipe files at controller.")
        
        # @stripeli this assumes that metisfl is installed in the remote host; make it more robust
        init_cmd = "{} && cd {} && {}".format(
            remote_on_login,
            self._federation_environment.controller.project_home,
            self._init_controller_cmd())
        MetisLogger.info("Running init cmd to controller host: {}".format(init_cmd))
        connection.run(init_cmd)
        connection.close()
        return

    def init_learner(self, index: int):
        # assert self.controller_server_entity_pb is not None, "Controller server entity is not initialized. Must call init_controller() first."
        learner_instance = self._federation_environment.learners.learners[index]
        fabric_connection_config = \
            learner_instance.connection_configs.get_fabric_connection_config()
        connection = Connection(**fabric_connection_config)
        
        # We do not use asynchronous or disown, since we want the remote subprocess to return standard (error) output.
        remote_metis_path = constants.REMOTE_METIS_LEARNER_PATH.format(learner_instance.grpc_servicer.port)
        
        # Delete existing directory if it exists, then recreate it.
        connection.run("rm -rf {}".format(remote_metis_path))
        connection.run("mkdir -p {}".format(remote_metis_path))

        MetisLogger.info("Copying model definition and dataset recipe files at learner: {}"
                         .format(learner_instance.learner_id))
        for _, filepath in self._dataset_receipe_fps.items():
            connection.put(filepath, remote_metis_path) if filepath else None
        if self._model_definition_tar_fp:
            connection.put(self._model_definition_tar_fp, remote_metis_path)

        # Fabric runs every command on a non-interactive mode and therefore the $PATH that might be set for a
        # running user might not be visible while running the command. A workaround is to always
        # source the respective bash_environment files.
        cuda_devices_str = ""
        # Exporting this environmental variable works for both Tensorflow/Keras and PyTorch.
        if learner_instance.cuda_devices and len(learner_instance.cuda_devices) > 0:
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
            self._model_definition_tar_fp))

        init_cmd = "{} && {} && cd {} && {}".format(
            remote_on_login,
            cuda_devices_str,
            learner_instance.project_home,
            self._init_learner_cmd(index))
        MetisLogger.info("Running init cmd to learner host: {}".format(init_cmd))
        connection.run(init_cmd)
        connection.close()
        return
    
    def _init_controller_cmd(self):
        # To create the controller grpc server entity, we need the hostname to which the server
        # will bind to and the port of the grpc servicer defined in the initial configuration file.
        # Controller is a subtype of RemoteHost instance, hence we pass it as is.        
        args = {}
        args["e"] = self._controller_server_entity_pb.SerializeToString().hex()
        config_attrs = {
            "g": "global_model_config",
            "c": "communication_protocol",
            "m": "local_model_config",
            "s": "model_store_config",
        }        
        for key, attr in config_attrs.items():
            args[key] = getattr(self._federation_environment, attr).to_proto().SerializeToString().hex()
        return self._get_cmd("controller", args)

    def _init_learner_cmd(self, index):
        learner_instance = self._federation_environment.learners.learners[index]
        
        remote_metis_path = constants.REMOTE_METIS_LEARNER_PATH.format(learner_instance.grpc_servicer.port)
        remote_metis_model_path = os.path.join(remote_metis_path, constants.MODEL_SAVE_DIR_NAME)
        
        remote_dataset_recipe_fps = dict()
        for filename, filepath in self._dataset_receipe_fps.items():
            remote_dataset_recipe_fps[filename] = os.path.join(remote_metis_path, filename) if filepath else None   

        config = {
            "l": self._learner_server_entities_pb[index].SerializeToString().hex(),
            "c": self._controller_server_entity_pb.SerializeToString().hex(),
            "f": self._federation_environment.homomorphic_encryption.to_proto().SerializeToString().hex(),
            "m": remote_metis_model_path,
            "t": learner_instance.dataset_configs.train_dataset_path,
            "v": learner_instance.dataset_configs.validation_dataset_path,
            "s": learner_instance.dataset_configs.test_dataset_path,
            "u": remote_dataset_recipe_fps[constants.TRAIN], 
            "w": remote_dataset_recipe_fps[constants.VALIDATION],
            "z": remote_dataset_recipe_fps[constants.TEST],
            "e": self._model.get_neural_engine()
        }
        return self._get_cmd("learner", config)
    
    def _get_cmd(self, entity, config):
        cmd = "python3 -m metisfl.{} ".format(entity)
        for key, value in config.items():
            cmd += "-{}={} ".format(key, value)
        return cmd

