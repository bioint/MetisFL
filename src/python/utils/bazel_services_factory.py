class BazelMetisServicesCmdFactory(object):

    @classmethod
    def bazel_init_controller_target(cls,
                                     hostname,
                                     port,
                                     global_model_specs_pb_ser,
                                     communication_specs_pb_ser,
                                     model_hyperparameters_pb_ser,
                                     model_store_config_pb_ser):

        # We convert the serialized protobuf version to hexadecimal in order
        # to send it over the wire. The reason is that the serialized protobuff
        # may contain unexpected order of single and double quotes,
        # e.g.,"b"\x08\x04\x10\x04\x1a\x0c*\n\r\xac\xc5'7\x15\x17\xb7\xd18"
        # In that case simply casting the byte representation to string fails.
        # Simply casting the byte representation to string will remove the
        # backslashes, therefore it is more convenient to compute the hexadecimal
        # representation and thereafter decode the hex to bytes.
        global_model_specs_pb_ser_hex = global_model_specs_pb_ser.hex()
        communication_specs_pb_ser_hex = communication_specs_pb_ser.hex()
        model_hyperparameters_pb_ser_hex = model_hyperparameters_pb_ser.hex()
        model_store_config_pb_ser_hex = model_store_config_pb_ser.hex()

        # CAUTION: For the hexadecimal to be valid we need to leave
        # a space to the right of the string replacing placeholder.

        # NOTE: Needs to go; simply run `python initialize_controller.py`
        bazel_cmd = \
            "bazel " \
            "run --incompatible_strict_action_env=true " \
            "//src/python/driver:initialize_controller " \
            "-- " \
            "--controller_hostname=\"{hostname}\" " \
            "--controller_port={port} " \
            "--global_model_specs_protobuff_serialized_hexadecimal={global_model_specs_pb_ser_hex} " \
            "--communication_specs_protobuff_serialized_hexadecimal={communication_specs_pb_ser_hex} " \
            "--model_hyperparameters_protobuff_serialized_hexadecimal={model_hyperparameters_pb_ser_hex} " \
            "--model_store_config_protobuff_serialized_hexadecimal={model_store_config_pb_ser_hex} ".format(
                hostname=hostname,
                port=port,
                global_model_specs_pb_ser_hex=global_model_specs_pb_ser_hex,
                communication_specs_pb_ser_hex=communication_specs_pb_ser_hex,
                model_hyperparameters_pb_ser_hex=model_hyperparameters_pb_ser_hex,
                model_store_config_pb_ser_hex=model_store_config_pb_ser_hex)
        return bazel_cmd

    @classmethod
    def bazel_init_learner_target(cls,
                                  learner_hostname,
                                  learner_port,
                                  controller_hostname,
                                  controller_port,
                                  he_scheme_pb_ser,
                                  model_definition,
                                  train_dataset,
                                  train_dataset_recipe,
                                  validation_dataset="",
                                  test_dataset="",
                                  validation_dataset_recipe="",
                                  test_dataset_recipe="",
                                  neural_engine="keras"):

        # Similarly as in the controller process invocation, we need to
        # convert the serialized protobuff to its hexadecimal representation.
        he_scheme_pb_ser_hex = he_scheme_pb_ser.hex()

        bazel_cmd = \
            "bazel " \
            "run --incompatible_strict_action_env=true " \
            "//src/python/driver:initialize_learner " \
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
            "--he_scheme_protobuff_serialized_hexadecimal={he_scheme_pb_ser_hex} ".format(
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
                he_scheme_pb_ser_hex=he_scheme_pb_ser_hex)
        return bazel_cmd
