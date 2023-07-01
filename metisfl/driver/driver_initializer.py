
import os
import tarfile
from typing import Callable

import cloudpickle
import inspect
from fabric import Connection

from metisfl import config
from metisfl.models.model_wrapper import MetisModel
from metisfl.proto.metis_pb2 import ServerEntity
from metisfl.utils import fedenv_parser
from metisfl.utils.metis_logger import MetisLogger


class DriverInitializer:

    def __init__(self,
                 controller_server_entity_pb: ServerEntity,
                 dataset_recipe_fns: dict[str, Callable],
                 dataset_fps: dict[str, list[str]],
                 fed_env: fedenv_parser.FederationEnvironment,
                 learner_server_entities_pb: list[ServerEntity],
                 model: MetisModel):
        assert config.TRAIN in dataset_recipe_fns, "Train dataset recipe function is required."
        
        self._controller_server_entity_pb = controller_server_entity_pb
        self._dataset_fps = dataset_fps
        self._federation_environment = fed_env
        self._learner_server_entities_pb = learner_server_entities_pb
        self._model = model
        
        self._model_save_dir = config.get_driver_model_save_dir()
        self._driver_dir = config.get_driver_path()

        self._dataset_recipe_fps = self._save_dataset_recipes(
            dataset_recipe_fns)
        self._model_definition_tar_fp = self._save_initial_model(model)

    def _save_dataset_recipes(self, dataset_recipe_fns) -> str:
        dataset_recipe_fps = dict()
        for key, dataset_recipe_fn in dataset_recipe_fns.items():
            if dataset_recipe_fn:
                dataset_pkl = os.path.join(
                    self._driver_dir, config.DATASET_recipe_FILENAMES[key])
                cloudpickle.register_pickle_by_value(
                    inspect.getmodule(dataset_recipe_fn))
                cloudpickle.dump(obj=dataset_recipe_fn,
                                 file=open(dataset_pkl, "wb+"))
                dataset_recipe_fps[key] = dataset_pkl
        return dataset_recipe_fps

    def _save_initial_model(self, model):
        self._model_weights_descriptor = model.get_weights_descriptor()
        model.save(self._model_save_dir)
        return self._make_tarfile(
            source_dir=self._model_save_dir,
            output_filename=os.path.basename(self._model_save_dir)
        )

    def _make_tarfile(self, output_filename, source_dir):
        output_dir = os.path.abspath(os.path.join(source_dir, os.pardir))
        output_filepath = os.path.join(
            output_dir, "{}.tar.gz".format(output_filename))
        with tarfile.open(output_filepath, "w:gz") as tar:
            tar.add(source_dir, arcname=os.path.basename(source_dir))
        return output_filepath

    def init_controller(self):
        fabric_connection_config = self._federation_environment.controller \
            .connection_configs.get_fabric_connection_config()
        connection = Connection(**fabric_connection_config)
        connection.run("rm -rf {}".format(config.get_controller_path()))
        connection.run("mkdir -p {}".format(config.get_controller_path()))
        
        remote_on_login = self._federation_environment.controller.connection_configs.on_login
        if len(remote_on_login) > 0 and remote_on_login[-1] == ";":
            remote_on_login = remote_on_login[:-1]

        # @stripeli this assumes that metisfl is installed in the remote host; make it more robust
        init_cmd = "{} && cd {} && {}".format(
            remote_on_login,
            self._federation_environment.controller.project_home,
            self._init_controller_cmd())
        MetisLogger.info(
            "Running init cmd to controller host: {}".format(init_cmd))
        connection.run(init_cmd)
        connection.close()
        return

    def init_learner(self, index: int):
        learner_instance, connection = self._get_learner_connection(index)

        # We do not use asynchronous or disown, since we want the remote subprocess to return standard (error) output.
        learner_path = config.get_learner_path(
            learner_instance.grpc_servicer.port)
        MetisLogger.info("Copying model definition, datasets and and dataset recipe files to learner: {}"
                         .format(learner_instance.learner_id))
        self._copy_assets_to_learner(index, connection, learner_path)

        # Fabric runs every command on a non-interactive mode and therefore the $PATH that might be set for a
        # running user might not be visible while running the command. A workaround is to always
        # source the respective bash_environment files.
        cuda_devices_str = ""
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
            learner_path,
            self._model_definition_tar_fp))

        init_cmd = "{} && {} && cd {} && {}".format(
            remote_on_login,
            cuda_devices_str,
            learner_instance.project_home,
            self._init_learner_cmd(index))
        MetisLogger.info(
            "Running init cmd to learner host: {}".format(init_cmd))
        connection.run(init_cmd)
        connection.close()

    def _copy_assets_to_learner(self, index, connection, learner_path):
        connection.run("rm -rf {}".format(learner_path))
        connection.run("mkdir -p {}".format(learner_path))
        
        for _, filepath in self._dataset_recipe_fps.items():
            connection.put(filepath, learner_path) if filepath else None
        for _, dataset_fps in self._dataset_fps.items():
            dataset_fp = dataset_fps[index]
            MetisLogger.info("Copying dataset file {} to learner: {}".format(
                dataset_fp, learner_path))
            connection.put(dataset_fp, learner_path) if dataset_fp else None
        if self._model_definition_tar_fp:
            connection.put(self._model_definition_tar_fp, learner_path)

    def _get_learner_connection(self, index):
        learner_instance = self._federation_environment.learners.learners[index]
        fabric_connection_config = \
            learner_instance.connection_configs.get_fabric_connection_config()
        connection = Connection(**fabric_connection_config)
        return learner_instance, connection

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
            args[key] = getattr(self._federation_environment,
                                attr).to_proto().SerializeToString().hex()
        return self._get_cmd("controller", args)

    def _init_learner_cmd(self, index):
        learner_instance = self._federation_environment.learners.learners[index]

        remote_metis_path = config.get_learner_path(
            learner_instance.grpc_servicer.port)
        remote_metis_model_path = os.path.join(
            remote_metis_path, config.MODEL_SAVE_DIR_NAME)
        remote_dataset_recipe_fps = dict()
        for key, filename in self._dataset_recipe_fps.items():
            remote_dataset_recipe_fps[key] = os.path.join(
                remote_metis_path, filename) if filename else None

        args = {
            "l": self._learner_server_entities_pb[index].SerializeToString().hex(),
            "c": self._controller_server_entity_pb.SerializeToString().hex(),
            "f": self._federation_environment.homomorphic_encryption.to_proto().SerializeToString().hex(),
            "m": remote_metis_model_path,
            "t": self._dataset_fps[config.TRAIN][index],
            "v": self._dataset_fps[config.VALIDATION][index] if config.VALIDATION in self._dataset_fps else None,
            "s": self._dataset_fps[config.TEST][index] if config.TEST in self._dataset_fps else None,
            "u": remote_dataset_recipe_fps[config.TRAIN],
            "w": remote_dataset_recipe_fps[config.VALIDATION] if config.VALIDATION in remote_dataset_recipe_fps else None,
            "z": remote_dataset_recipe_fps[config.TEST] if config.TEST in remote_dataset_recipe_fps else None,
            "e": self._model.get_neural_engine()
        }
        return self._get_cmd("learner", args)

    def _get_cmd(self, entity, config):
        cmd = "python3 -m metisfl.{} ".format(entity)
        for key, value in config.items():
            if value:
                cmd += "-{}={} ".format(key, value)
        return cmd
