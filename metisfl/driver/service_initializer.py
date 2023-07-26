
import cloudpickle
import inspect
import os
import tarfile

from fabric import Connection
from typing import Callable, Dict, List, Tuple

from metisfl import config
from metisfl.models.metis_model import MetisModel
from metisfl.utils.fedenv import FederationEnvironment, RemoteHost
from metisfl.utils.logger import MetisLogger


class ServiceInitializer:

    def __init__(self,
                 federation_environment: FederationEnvironment,
                 dataset_recipe_fns: Dict[str, Callable],
                 dataset_fps: Dict[str, List[str]],
                 model: MetisModel):
        assert config.TRAIN in dataset_recipe_fns, "Train dataset recipe function is required."
        # Setup configuration variables.
        self._model_save_dir = config.get_driver_model_save_dir()
        self._driver_dir = config.get_driver_path()

        self._federation_environment = federation_environment
        self._dataset_recipe_fps = self._save_dataset_recipes(
            dataset_recipe_fns)
        self._dataset_fps = dataset_fps
        self._model = model
        self._model_definition_tar_fp = self._save_initial_model(model)

    def init_controller(self):
        connection = self._get_controller_connection()
        connection.run("rm -rf {}".format(config.get_controller_path()))
        connection.run("mkdir -p {}".format(config.get_controller_path()))

        remote_on_login = self._federation_environment.controller.on_login_command
        if len(remote_on_login) > 0 and remote_on_login[-1] == ";":
            remote_on_login = remote_on_login[:-1] + " &&"

        # @stripeli this assumes that metisfl is installed in the remote host; make it more robust
        init_cmd = "{} cd {} && {}".format(
            remote_on_login,
            self._federation_environment.controller.project_home,
            self._init_controller_cmd())
        MetisLogger.info(
            "Running init cmd to controller host: {}".format(init_cmd))
        connection.run(init_cmd)
        connection.close()
        return

    def init_learner(self, index: int):
        learner, connection = self._get_learner_connection(index)

        # We do not use asynchronous or disown, since we want the remote subprocess to return standard (error) output.
        learner_path = config.get_learner_path(learner.grpc_port)
        MetisLogger.info(
            "Copying model definition, datasets and and dataset recipe files to learner: {}"
            .format(learner.identifier))
        self._copy_assets_to_learner(index, connection, learner_path)

        # Fabric runs every command on a non-interactive mode and therefore the $PATH that might be set for a
        # running user might not be visible while running the command. A workaround is to always
        # source the respective bash_environment files.
        cuda_devices_str = ""
        if learner.cuda_devices and len(learner.cuda_devices) > 0:
            cuda_devices_str = "export CUDA_VISIBLE_DEVICES=\"{}\"" \
                .format(",".join([str(c) for c in learner.cuda_devices]))
        remote_on_login = learner.on_login_command
        if remote_on_login:
            remote_on_login = + " &&"

        # Un-taring model definition zipped file.
        MetisLogger.info("Un-taring model definition files at learner: {}"
                         .format(learner.identifier))
        connection.run("cd {}; tar -xvzf {}".format(
            learner_path,
            self._model_definition_tar_fp))

        init_cmd = "{} cd {} && {} && {}".format(
            remote_on_login,
            learner.project_home,
            cuda_devices_str,
            self._init_learner_cmd(index))
        MetisLogger.info(
            "Running init cmd to learner host: {}".format(init_cmd))
        connection.run(init_cmd)
        connection.close()
        return

    def _save_dataset_recipes(self, dataset_recipe_fns) -> str:
        dataset_recipe_fps = dict()
        for dataset_id, dataset_recipe_fn in dataset_recipe_fns.items():
            if dataset_recipe_fn:
                dataset_pkl = os.path.join(
                    self._driver_dir, config.DATASET_RECIPE_FILENAMES[dataset_id])
                cloudpickle.register_pickle_by_value(
                    inspect.getmodule(dataset_recipe_fn))
                cloudpickle.dump(obj=dataset_recipe_fn,
                                 file=open(dataset_pkl, "wb+"))
                dataset_recipe_fps[dataset_id] = dataset_pkl
        return dataset_recipe_fps

    def _save_initial_model(self, model: MetisModel) -> str:
        self._model_weights_descriptor = model.get_weights_descriptor()
        model.save(self._model_save_dir, initialize=True)
        return self._make_tarfile(
            source_dir=self._model_save_dir,
            output_filename=os.path.basename(self._model_save_dir))

    def _make_tarfile(self, output_filename, source_dir) -> str:
        output_dir = os.path.abspath(os.path.join(source_dir, os.pardir))
        output_filepath = os.path.join(
            output_dir, "{}.tar.gz".format(output_filename))
        with tarfile.open(output_filepath, "w:gz") as tar:
            tar.add(source_dir, arcname=os.path.basename(source_dir))
        return output_filepath

    def _copy_assets_to_learner(self, index, connection, learner_path):
        connection.run("rm -rf {}".format(learner_path))
        connection.run("mkdir -p {}".format(learner_path))

        for recipe_fp in self._dataset_recipe_fps.values():
            connection.put(recipe_fp, learner_path) if recipe_fp else None
        for dataset_fp in self._dataset_fps.values():
            dataset_fp = dataset_fp[index]
            MetisLogger.info("Copying dataset file {} to learner: {}".format(
                dataset_fp, learner_path))
            connection.put(dataset_fp, learner_path) if dataset_fp else None
        if self._model_definition_tar_fp:
            connection.put(self._model_definition_tar_fp, learner_path)

    def _get_controller_connection(self) -> Connection:
        fabric_connection_config = self._federation_environment.controller \
            .get_fabric_connection_config()
        connection = Connection(**fabric_connection_config)
        return connection

    def _get_learner_connection(self, index) -> Tuple[RemoteHost, Connection]:
        learner = self._federation_environment.learners[index]
        fabric_connection_config = \
            learner.get_fabric_connection_config()
        connection = Connection(**fabric_connection_config)
        return learner, connection

    def _init_controller_cmd(self):
        args = {
            "e": self._federation_environment.controller.get_server_entity_pb().SerializeToString().hex(),
            "g": self._federation_environment.get_global_model_config_pb().SerializeToString().hex(),
            "c": self._federation_environment.get_communication_protocol_pb().SerializeToString().hex(),
            "m": self._federation_environment.get_local_model_config_pb().SerializeToString().hex(),
            "s": self._federation_environment.get_model_store_config_pb().SerializeToString().hex(),
        }
        return self._get_cmd("controller", args)

    def _init_learner_cmd(self, index):
        learner = self._federation_environment.learners[index]

        remote_metis_path = config.get_learner_path(
            learner.grpc_port)
        remote_metis_model_path = os.path.join(
            remote_metis_path, config.MODEL_SAVE_DIR_NAME)
        remote_dataset_recipe_fps = dict()
        for key, filename in self._dataset_recipe_fps.items():
            remote_dataset_recipe_fps[key] = os.path.join(
                remote_metis_path, filename) if filename else None

        args = {
            "l": self._federation_environment.learners[index].get_server_entity_pb().SerializeToString().hex(),
            "c": self._federation_environment.controller.get_server_entity_pb(gen_connection_entity=True).SerializeToString().hex(),
            "f": self._federation_environment.get_encryption_scheme_pb().SerializeToString().hex(),
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

    def _get_cmd(self, entity, config) -> str:
        cmd = "python3 -m metisfl.{} ".format(entity)
        MetisLogger.info("Running: {}".format(config))
        for key, value in config.items():
            if value:
                cmd += "-{}={} ".format(key, value)
        return cmd
