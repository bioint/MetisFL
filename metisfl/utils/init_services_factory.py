class MetisInitServicesCmdFactory(object):

    @classmethod
    def init_controller_target(cls,
                               controller_server_entity_pb_ser,
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
        controller_server_entity_pb_ser_hex = controller_server_entity_pb_ser.hex()
        global_model_specs_pb_ser_hex = global_model_specs_pb_ser.hex()
        communication_specs_pb_ser_hex = communication_specs_pb_ser.hex()
        model_hyperparameters_pb_ser_hex = model_hyperparameters_pb_ser.hex()
        model_store_config_pb_ser_hex = model_store_config_pb_ser.hex()

        # CAUTION: For the hexadecimal to be valid we need to leave
        # a space to the right of the string replacing placeholder.
        bazel_cmd = \
            "python " \
            "-m metisfl.controller " \
            "--controller_server_entity_protobuff_serialized_hexadecimal={controller_server_entity_pb_ser_hex} " \
            "--global_model_specs_protobuff_serialized_hexadecimal={global_model_specs_pb_ser_hex} " \
            "--communication_specs_protobuff_serialized_hexadecimal={communication_specs_pb_ser_hex} " \
            "--model_hyperparameters_protobuff_serialized_hexadecimal={model_hyperparameters_pb_ser_hex} " \
            "--model_store_config_protobuff_serialized_hexadecimal={model_store_config_pb_ser_hex} ".format(
                controller_server_entity_pb_ser_hex=controller_server_entity_pb_ser_hex,
                global_model_specs_pb_ser_hex=global_model_specs_pb_ser_hex,
                communication_specs_pb_ser_hex=communication_specs_pb_ser_hex,
                model_hyperparameters_pb_ser_hex=model_hyperparameters_pb_ser_hex,
                model_store_config_pb_ser_hex=model_store_config_pb_ser_hex)
        return bazel_cmd

    @classmethod
    def init_learner_target(cls,
                            learner_server_entity_pb_ser,
                            controller_server_entity_pb_ser,
                            he_scheme_pb_ser,
                            model_dir,
                            train_dataset,
                            train_dataset_recipe,
                            validation_dataset="",
                            test_dataset="",
                            validation_dataset_recipe="",
                            test_dataset_recipe="",
                            neural_engine="keras"):
        # Similarly as in the controller process invocation, we need to
        # convert the serialized protobuff to its hexadecimal representation.
        learner_server_entity_pb_ser_hex = learner_server_entity_pb_ser.hex()
        controller_server_entity_pb_ser_hex = controller_server_entity_pb_ser.hex()
        he_scheme_pb_ser_hex = he_scheme_pb_ser.hex()

        bazel_cmd = \
            "python " \
            "-m metisfl.learner " \
            "--learner_server_entity_protobuff_serialized_hexadecimal={learner_server_entity_pb_ser_hex} " \
            "--controller_server_entity_protobuff_serialized_hexadecimal={controller_server_entity_pb_ser_hex} " \
            "--he_scheme_protobuff_serialized_hexadecimal={he_scheme_pb_ser_hex} " \
            "--neural_engine=\"{neural_engine}\" " \
            "--model_dir=\"{model_dir}\" " \
            "--train_dataset=\"{train_dataset}\" " \
            "--validation_dataset=\"{validation_dataset}\" " \
            "--test_dataset=\"{test_dataset}\" " \
            "--train_dataset_recipe=\"{train_dataset_recipe}\" " \
            "--validation_dataset_recipe=\"{validation_dataset_recipe}\" " \
            "--test_dataset_recipe=\"{test_dataset_recipe}\" ".format(
                learner_server_entity_pb_ser_hex=learner_server_entity_pb_ser_hex,
                controller_server_entity_pb_ser_hex=controller_server_entity_pb_ser_hex,
                he_scheme_pb_ser_hex=he_scheme_pb_ser_hex,
                neural_engine=neural_engine,
                model_dir=model_dir,
                train_dataset=train_dataset,
                validation_dataset=validation_dataset,
                test_dataset=test_dataset,
                train_dataset_recipe=train_dataset_recipe,
                validation_dataset_recipe=validation_dataset_recipe,
                test_dataset_recipe=test_dataset_recipe)
        return bazel_cmd
