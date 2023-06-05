

class BazelMetisServicesCmdFactory(object):

    @classmethod
    def bazel_init_controller_target(cls,
                                     hostname,
                                     port,
                                     aggregation_rule,
                                     participation_ratio,
                                     communication_specs_pb,
                                     model_hyperparameters_pb,
                                     fhe_scheme_protobuff):
        bazel_cmd = \
            "bazel " \
            "run --incompatible_strict_action_env=true " \
            "//projectmetis/python/driver:initialize_controller " \
            "-- " \
            "--controller_hostname=\"{hostname}\" " \
            "--controller_port={port} " \
            "--aggregation_rule=\"{aggregation_rule}\" " \
            "--learners_participation_ratio={participation_ratio} " \
            "--communication_specs_protobuff=\"{communication_specs_pb}\" " \
            "--model_hyperparameters_protobuff=\"{model_hyperparameters_pb}\" " \
            "--fhe_scheme_protobuff=\"{fhe_scheme_protobuff}\" ".format(
                hostname=hostname, port=port, aggregation_rule=aggregation_rule,
                participation_ratio=participation_ratio, communication_specs_pb=communication_specs_pb,
                model_hyperparameters_pb=model_hyperparameters_pb, fhe_scheme_protobuff=fhe_scheme_protobuff)
        return bazel_cmd

    @classmethod
    def bazel_init_learner_target(cls,
                                  learner_hostname,
                                  learner_port,
                                  controller_hostname,
                                  controller_port,
                                  fhe_scheme_protobuff,
                                  model_definition,
                                  train_dataset,
                                  train_dataset_recipe,
                                  validation_dataset="",
                                  test_dataset="",
                                  validation_dataset_recipe="",
                                  test_dataset_recipe="",
                                  neural_engine="keras"):
        bazel_cmd = \
            "bazel " \
            "run --incompatible_strict_action_env=true " \
            "//projectmetis/python/driver:initialize_learner " \
            "-- " \
            "--neural_engine=\"{neural_engine}\" " \
            "--learner_hostname=\"{learner_hostname}\" " \
            "--learner_port={learner_port} " \
            "--controller_hostname=\"{controller_hostname}\" " \
            "--controller_port={controller_port} " \
            "--model_definition=\"{model_definition}\" " \
            "--train_dataset=\"{train_dataset}\" " \
            "--validation_dataset=\"{validation_dataset}\" " \
            "--test_dataset=\"{test_dataset}\" " \
            "--train_dataset_recipe=\"{train_dataset_recipe}\" " \
            "--validation_dataset_recipe=\"{validation_dataset_recipe}\" " \
            "--test_dataset_recipe=\"{test_dataset_recipe}\" " \
            "--fhe_scheme_protobuff=\"{fhe_scheme_protobuff}\" ".format(
                neural_engine=neural_engine,
                learner_hostname=learner_hostname,
                learner_port=learner_port,
                controller_hostname=controller_hostname,
                controller_port=controller_port,
                model_definition=model_definition,
                train_dataset=train_dataset,
                validation_dataset=validation_dataset,
                test_dataset=test_dataset,
                train_dataset_recipe=train_dataset_recipe,
                validation_dataset_recipe=validation_dataset_recipe,
                test_dataset_recipe=test_dataset_recipe,
                fhe_scheme_protobuff=fhe_scheme_protobuff)
        return bazel_cmd
