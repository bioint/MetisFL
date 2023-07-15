import numpy.lib.format
import sys

import numpy as np

from metisfl.proto import controller_pb2, learner_pb2, model_pb2, metis_pb2, service_common_pb2


class ControllerServiceProtoMessages(object):

    @classmethod
    def construct_get_community_model_evaluation_lineage_request_pb(cls, num_backtracks):
        return controller_pb2.GetCommunityModelEvaluationLineageRequest(num_backtracks=num_backtracks)

    @classmethod
    def construct_get_local_task_lineage_request_pb(cls, num_backtracks, learner_ids):
        return controller_pb2.GetLocalTaskLineageRequest(num_backtracks=num_backtracks,
                                                         learner_ids=learner_ids)

    @classmethod
    def construct_get_runtime_metadata_lineage_request_pb(cls, num_backtracks):
        return controller_pb2.GetRuntimeMetadataLineageRequest(num_backtracks=num_backtracks)

    @classmethod
    def construct_get_participating_learners_request_pb(cls):
        return controller_pb2.GetParticipatingLearnersRequest()

    @classmethod
    def construct_join_federation_request_pb(cls, server_entity_pb, local_dataset_spec_pb):
        return controller_pb2.JoinFederationRequest(server_entity=server_entity_pb,
                                                    local_dataset_spec=local_dataset_spec_pb)

    @classmethod
    def construct_leave_federation_request_pb(cls, learner_id, auth_token):
        return controller_pb2.LeaveFederationRequest(learner_id=learner_id, auth_token=auth_token)

    @classmethod
    def construct_mark_task_completed_request_pb(cls, learner_id, auth_token, completed_learning_task_pb):
        return controller_pb2.MarkTaskCompletedRequest(learner_id=learner_id,
                                                       auth_token=auth_token,
                                                       task=completed_learning_task_pb)

    @classmethod
    def construct_replace_community_model_request_pb(cls, federated_model_pb):
        assert isinstance(federated_model_pb, model_pb2.FederatedModel)
        return controller_pb2.ReplaceCommunityModelRequest(model=federated_model_pb)


class LearnerServiceProtoMessages(object):

    @classmethod
    def construct_evaluate_model_request_pb(cls, model=None, batch_size=None, eval_train=None,
                                            eval_test=None, eval_valid=None, metrics_pb=None):
        assert isinstance(metrics_pb, metis_pb2.EvaluationMetrics)
        evaluation_dataset = []
        if eval_train:
            evaluation_dataset.append(
                learner_pb2.EvaluateModelRequest.dataset_to_eval.TRAINING)
        if eval_test:
            evaluation_dataset.append(
                learner_pb2.EvaluateModelRequest.dataset_to_eval.TEST)
        if eval_valid:
            evaluation_dataset.append(
                learner_pb2.EvaluateModelRequest.dataset_to_eval.VALIDATION)
        return learner_pb2.EvaluateModelRequest(
            model=model, batch_size=batch_size, evaluation_dataset=evaluation_dataset, metrics=metrics_pb)

    @classmethod
    def construct_evaluate_model_response_pb(cls, evaluation_pb=None):
        assert isinstance(evaluation_pb, metis_pb2.ModelEvaluations)
        return learner_pb2.EvaluateModelResponse(evaluations=evaluation_pb)

    @classmethod
    def construct_run_task_request_pb(cls, federated_model_pb=None, learning_task_pb=None, hyperparameters_pb=None):
        assert isinstance(federated_model_pb, model_pb2.FederatedModel) \
            and isinstance(learning_task_pb, metis_pb2.LearningTask) \
            and isinstance(hyperparameters_pb, metis_pb2.Hyperparameters)
        return learner_pb2.RunTaskRequest(
            federated_model=federated_model_pb,
            task=learning_task_pb,
            hyperparameters=hyperparameters_pb)

    @classmethod
    def construct_run_task_response_pb(cls, ack_pb=None):
        assert isinstance(ack_pb, service_common_pb2.Ack)
        return learner_pb2.RunTaskResponse(ack=ack_pb)


class MetisProtoMessages(object):

    @classmethod
    def construct_server_entity_pb(cls, hostname, port, public_certificate_file=None, private_key_file=None):
        return metis_pb2.ServerEntity(hostname=hostname,
                                      port=port,
                                      public_certificate_file=public_certificate_file,
                                      private_key_file=private_key_file)

    @classmethod
    def construct_ssl_config_pb(cls, enable_ssl=False, config_pb=None):
        if enable_ssl and config_pb:
            if isinstance(config_pb, metis_pb2.SSLConfigFiles):
                return metis_pb2.SSLConfig(enable_ssl=True, ssl_config_files=config_pb)
            elif isinstance(config_pb, metis_pb2.SSLConfigStream):
                return metis_pb2.SSLConfig(enable_ssl=True, ssl_config_stream=config_pb)
        else:
            # Default is TLS/SSL disabled.
            return metis_pb2.SSLConfig(enable_ssl=False)

    @classmethod
    def construct_ssl_config_files_pb(cls, public_certificate_file=None, private_key_file=None):
        return metis_pb2.SSLConfigFiles(
            public_certificate_file=public_certificate_file,
            private_key_file=private_key_file)

    @classmethod
    def construct_ssl_config_stream_pb(cls, public_certificate_stream=None, private_key_stream=None):
        return metis_pb2.SSLConfigStream(
            public_certificate_stream=public_certificate_stream,
            private_key_stream=private_key_stream)

    @classmethod
    def construct_he_scheme_config_pb(cls, enabled=False, crypto_context_file=None,
                                      public_key_file=None, private_key_file=None,
                                      empty_scheme_config_pb=None, ckks_scheme_config_pb=None):
        if empty_scheme_config_pb is not None:
            return metis_pb2.HESchemeConfig(enabled=enabled,
                                            crypto_context_file=crypto_context_file,
                                            public_key_file=public_key_file,
                                            private_key_file=private_key_file,
                                            empty_scheme_config=empty_scheme_config_pb)
        if ckks_scheme_config_pb is not None:
            return metis_pb2.HESchemeConfig(enabled=enabled,
                                            crypto_context_file=crypto_context_file,
                                            public_key_file=public_key_file,
                                            private_key_file=private_key_file,
                                            ckks_scheme_config=ckks_scheme_config_pb)

    @classmethod
    def construct_empty_scheme_config_pb(cls):
        return metis_pb2.EmptySchemeConfig()

    @classmethod
    def construct_ckks_scheme_config_pb(cls, batch_size, scaling_factor_bits):
        return metis_pb2.CKKSSchemeConfig(batch_size=batch_size, scaling_factor_bits=scaling_factor_bits)

    @classmethod
    def construct_dataset_spec_pb(cls, num_training_examples, num_validation_examples, num_test_examples,
                                  training_spec, validation_spec, test_spec,
                                  is_classification=False, is_regression=False):
        if is_classification is True:
            training_spec = cls.construct_classification_dataset_spec_pb(
                training_spec)
            validation_spec = cls.construct_classification_dataset_spec_pb(
                validation_spec)
            test_spec = cls.construct_classification_dataset_spec_pb(test_spec)
            return metis_pb2.DatasetSpec(num_training_examples=num_training_examples,
                                         num_validation_examples=num_validation_examples,
                                         num_test_examples=num_test_examples,
                                         training_classification_spec=training_spec,
                                         validation_classification_spec=validation_spec,
                                         test_classification_spec=test_spec)
        elif is_regression:
            training_spec = cls.construct_regression_dataset_spec_pb(
                training_spec)
            validation_spec = cls.construct_regression_dataset_spec_pb(
                validation_spec)
            test_spec = cls.construct_regression_dataset_spec_pb(test_spec)
            return metis_pb2.DatasetSpec(num_training_examples=num_training_examples,
                                         num_validation_examples=num_validation_examples,
                                         num_test_examples=num_test_examples,
                                         training_regression_spec=training_spec,
                                         validation_regression_spec=validation_spec,
                                         test_regression_spec=test_spec)
        else:
            raise RuntimeError(
                "Need to specify whether incoming dataset spec is regression or classification.")

    @classmethod
    def construct_classification_dataset_spec_pb(cls, class_distribution_specs=None):
        if class_distribution_specs is None:
            class_distribution_specs = {}
        return metis_pb2.DatasetSpec.ClassificationDatasetSpec(
            class_examples_num=class_distribution_specs)

    @classmethod
    def construct_regression_dataset_spec_pb(cls, regression_specs=None):
        if regression_specs is None:
            regression_specs = metis_pb2.DatasetSpec.RegressionDatasetSpec()
        else:
            regression_specs = metis_pb2.DatasetSpec.RegressionDatasetSpec(
                min=regression_specs["min"], max=regression_specs["max"],
                mean=regression_specs["mean"], median=regression_specs["median"],
                mode=regression_specs["mode"], stddev=regression_specs["stddev"]
            )
        return regression_specs

    @classmethod
    def construct_learning_task_pb(cls, num_local_updates, validation_dataset_pct, metrics=None):
        return metis_pb2.LearningTask(
            num_local_updates=num_local_updates,
            training_dataset_percentage_for_stratified_validation=validation_dataset_pct,
            metrics=metrics)

    @classmethod
    def construct_completed_learning_task_pb(cls, model_pb, task_execution_metadata_pb, aux_metadata):
        return metis_pb2.CompletedLearningTask(model=model_pb,
                                               execution_metadata=task_execution_metadata_pb,
                                               aux_metadata=aux_metadata)

    @classmethod
    def construct_task_execution_metadata_pb(cls,
                                             global_iteration,
                                             task_evaluation_pb,
                                             completed_epochs,
                                             completed_batches,
                                             batch_size,
                                             processing_ms_per_epoch,
                                             processing_ms_per_batch):
        return metis_pb2.TaskExecutionMetadata(global_iteration=global_iteration,
                                               task_evaluation=task_evaluation_pb,
                                               completed_epochs=completed_epochs,
                                               completed_batches=completed_batches,
                                               batch_size=batch_size,
                                               processing_ms_per_epoch=processing_ms_per_epoch,
                                               processing_ms_per_batch=processing_ms_per_batch)

    @classmethod
    def construct_task_evaluation_pb(cls, epoch_training_evaluations_pbs,
                                     epoch_validation_evaluations_pbs=None,
                                     epoch_test_evaluations_pbs=None):
        return metis_pb2.TaskEvaluation(training_evaluation=epoch_training_evaluations_pbs,
                                        validation_evaluation=epoch_validation_evaluations_pbs,
                                        test_evaluation=epoch_test_evaluations_pbs)

    @classmethod
    def construct_epoch_evaluation_pb(cls, epoch_id, model_evaluation_pb):
        return metis_pb2.EpochEvaluation(epoch_id=epoch_id,
                                         model_evaluation=model_evaluation_pb)

    @classmethod
    def construct_evaluation_metrics_pb(cls, metrics=None):
        if metrics is None:
            metrics = [""]
        if not isinstance(metrics, list):
            metrics = [metrics]
        return metis_pb2.EvaluationMetrics(metric=metrics)

    @classmethod
    def construct_model_evaluation_pb(cls, metric_values=None):
        if metric_values is None:
            metric_values = dict()
        return metis_pb2.ModelEvaluation(metric_values=metric_values)

    @classmethod
    def construct_model_evaluations_pb(cls, training_evaluation_pb, validation_evaluation_pb, test_evaluation_pb):
        return metis_pb2.ModelEvaluations(training_evaluation=training_evaluation_pb,
                                          validation_evaluation=validation_evaluation_pb,
                                          test_evaluation=test_evaluation_pb)

    @classmethod
    def construct_hyperparameters_pb(cls, batch_size, optimizer_config_pb):
        assert isinstance(batch_size, int)
        return metis_pb2.Hyperparameters(batch_size=batch_size, optimizer=optimizer_config_pb)

    @classmethod
    def construct_controller_params_pb(cls, server_entity_pb, global_model_specs_pb,
                                       communication_specs_pb, model_store_config_pb,
                                       model_hyperparams_pb):
        return metis_pb2.ControllerParams(server_entity=server_entity_pb,
                                          global_model_specs=global_model_specs_pb,
                                          communication_specs=communication_specs_pb,
                                          model_store_config=model_store_config_pb,
                                          model_hyperparams=model_hyperparams_pb)

    @classmethod
    def construct_controller_modelhyperparams_pb(cls, batch_size, epochs, optimizer_pb, percent_validation):
        return metis_pb2.ControllerParams.ModelHyperparams(batch_size=batch_size,
                                                           epochs=epochs,
                                                           optimizer=optimizer_pb,
                                                           percent_validation=percent_validation)

    @classmethod
    def construct_no_eviction_pb(cls):
        return metis_pb2.NoEviction()

    @classmethod
    def construct_lineage_length_eviction_pb(cls, lineage_length):
        assert lineage_length > 0, "Lineage length value needs to be positive!"
        return metis_pb2.LineageLengthEviction(lineage_length=lineage_length)

    @classmethod
    def construct_eviction_policy_pb(cls, policy_name, lineage_length):
        if policy_name.upper() == "NOEVICTION":
            return MetisProtoMessages.construct_no_eviction_pb()
        elif policy_name.upper() == "LINEAGELENGTHEVICTION":
            return MetisProtoMessages.construct_lineage_length_eviction_pb(lineage_length)

    @classmethod
    def construct_model_store_specs_pb(cls, eviction_policy_pb):
        if isinstance(eviction_policy_pb, metis_pb2.NoEviction):
            return metis_pb2.ModelStoreSpecs(no_eviction=eviction_policy_pb)
        elif isinstance(eviction_policy_pb, metis_pb2.LineageLengthEviction):
            return metis_pb2.ModelStoreSpecs(lineage_length_eviction=eviction_policy_pb)
        else:
            raise RuntimeError("Not a supported protobuff eviction policy.")

    @classmethod
    def construct_model_store_config_pb(cls, name, eviction_policy,
                                        lineage_length=None, store_hostname=None, store_port=None):
        eviction_policy_pb = MetisProtoMessages.construct_eviction_policy_pb(
            eviction_policy, lineage_length)
        model_store_specs_pb = MetisProtoMessages.construct_model_store_specs_pb(
            eviction_policy_pb)
        if name.upper() == "INMEMORY":
            model_store_pb = MetisProtoMessages.construct_in_memory_store_pb(
                model_store_specs_pb)
            return metis_pb2.ModelStoreConfig(in_memory_store=model_store_pb)
        elif name.upper() == "REDIS":
            model_store_pb = MetisProtoMessages.construct_redis_store_pb(
                model_store_specs_pb, store_hostname, store_port)
            return metis_pb2.ModelStoreConfig(redis_db_store=model_store_pb)
        else:
            raise RuntimeError("Not a supported model store.")

    @classmethod
    def construct_in_memory_store_pb(cls, model_store_specs_pb):
        return metis_pb2.InMemoryStore(model_store_specs=model_store_specs_pb)

    @classmethod
    def construct_redis_store_pb(cls, model_store_specs_pb, hostname, port):
        server_entity_pb = MetisProtoMessages.construct_server_entity_pb(
            hostname=hostname, port=port)
        return metis_pb2.RedisDBStore(model_store_specs=model_store_specs_pb,
                                      server_entity=server_entity_pb)

    @classmethod
    def construct_fed_avg_pb(cls):
        return metis_pb2.FedAvg()

    @classmethod
    def construct_fed_stride_pb(cls, stride_length):
        return metis_pb2.FedStride(stride_length=stride_length)

    @classmethod
    def construct_fed_rec_pb(cls):
        return metis_pb2.FedRec()

    @classmethod
    def construct_pwa_pb(cls, he_scheme_config_pb):
        return metis_pb2.PWA(he_scheme_config=he_scheme_config_pb)

    @classmethod
    def construct_aggregation_rule_specs_pb(cls, scaling_factor):
        if scaling_factor.upper() == "NUMCOMPLETEDBATCHES":
            scaling_factor_pb = metis_pb2.AggregationRuleSpecs.ScalingFactor.NUM_COMPLETED_BATCHES
        elif scaling_factor.upper() == "NUMPARTICIPANTS":
            scaling_factor_pb = metis_pb2.AggregationRuleSpecs.ScalingFactor.NUM_PARTICIPANTS
        elif scaling_factor.upper() == "NUMTRAININGEXAMPLES":
            scaling_factor_pb = metis_pb2.AggregationRuleSpecs.ScalingFactor.NUM_TRAINING_EXAMPLES
        else:
            scaling_factor_pb = metis_pb2.AggregationRuleSpecs.ScalingFactor.UNKNOWN
            raise RuntimeError("Unsupported scaling factor.")

        return metis_pb2.AggregationRuleSpecs(scaling_factor=scaling_factor_pb)

    @classmethod
    def construct_aggregation_rule_pb(cls, rule_name, scaling_factor, stride_length, he_scheme_config_pb):
        aggregation_rule_specs_pb = MetisProtoMessages.construct_aggregation_rule_specs_pb(
            scaling_factor)
        if rule_name.upper() == "FEDAVG":
            return metis_pb2.AggregationRule(fed_avg=MetisProtoMessages.construct_fed_avg_pb(),
                                             aggregation_rule_specs=aggregation_rule_specs_pb)
        elif rule_name.upper() == "FEDSTRIDE":
            return metis_pb2.AggregationRule(fed_stride=MetisProtoMessages.construct_fed_stride_pb(stride_length),
                                             aggregation_rule_specs=aggregation_rule_specs_pb)
        elif rule_name.upper() == "FEDREC":
            return metis_pb2.AggregationRule(fed_rec=MetisProtoMessages.construct_fed_rec_pb(),
                                             aggregation_rule_specs=aggregation_rule_specs_pb)
        elif rule_name.upper() == "PWA":
            return metis_pb2.AggregationRule(
                pwa=MetisProtoMessages.construct_pwa_pb(
                    he_scheme_config_pb=he_scheme_config_pb),
                aggregation_rule_specs=aggregation_rule_specs_pb)
        else:
            raise RuntimeError("Unsupported rule name.")

    @classmethod
    def construct_global_model_specs(cls, aggregation_rule_pb, learners_participation_ratio):
        return metis_pb2.GlobalModelSpecs(aggregation_rule=aggregation_rule_pb,
                                          learners_participation_ratio=learners_participation_ratio)

    @classmethod
    def construct_communication_specs_pb(cls, protocol, semi_sync_lambda=None, semi_sync_recompute_num_updates=None):
        if protocol.upper() == "SYNCHRONOUS":
            protocol_pb = metis_pb2.CommunicationSpecs.Protocol.SYNCHRONOUS
        elif protocol.upper() == "ASYNCHRONOUS":
            protocol_pb = metis_pb2.CommunicationSpecs.Protocol.ASYNCHRONOUS
        elif protocol.upper() == "SEMI_SYNCHRONOUS":
            protocol_pb = metis_pb2.CommunicationSpecs.Protocol.SEMI_SYNCHRONOUS
        else:
            protocol_pb = metis_pb2.CommunicationSpecs.Protocol.UNKNOWN

        return metis_pb2.CommunicationSpecs(protocol=protocol_pb,
                                            protocol_specs=metis_pb2.ProtocolSpecs(
                                                semi_sync_lambda=semi_sync_lambda,
                                                semi_sync_recompute_num_updates=semi_sync_recompute_num_updates))


class ModelProtoMessages(object):

    class TensorSpecProto(object):

        NUMPY_DATA_TYPE_TO_PROTO_LOOKUP = {
            "i1": model_pb2.DType.Type.INT8,
            "i2": model_pb2.DType.Type.INT16,
            "i4": model_pb2.DType.Type.INT32,
            "i8": model_pb2.DType.Type.INT64,
            "u1": model_pb2.DType.Type.UINT8,
            "u2": model_pb2.DType.Type.UINT16,
            "u4": model_pb2.DType.Type.UINT32,
            "u8": model_pb2.DType.Type.UINT64,
            "f4": model_pb2.DType.Type.FLOAT32,
            "f8": model_pb2.DType.Type.FLOAT64
        }

        INV_NUMPY_DATA_TYPE_TO_PROTO_LOOKUP = {
            v: k for k, v in NUMPY_DATA_TYPE_TO_PROTO_LOOKUP.items()
        }

        @classmethod
        def numpy_array_to_proto_tensor_spec(cls, arr):

            # Examples of numpy arrays representation:
            #   "<i2" == (little-endian int8)
            #   "<u4" == (little-endian uint64)
            #   ">f4" == (big-endian float32)
            #   "=f2" == (system-default endian float8)
            # In general, the first character represents the endian type
            # and the subsequent characters the data type in the form of
            # integer(i), unsigned integer(u), float(f), complex(c) and the
            # digits the number of bytes, 4 refers to 4 bytes = 64bits.

            length = arr.size
            arr_metadata = numpy.lib.format.header_data_from_array_1_0(arr)
            shape = arr_metadata["shape"]
            dimensions = [s for s in shape]
            fortran_order = arr_metadata["fortran_order"]

            # For the byteorder representation in numpy check
            # https://numpy.org/doc/stable/reference/generated/numpy.dtype.byteorder.html
            descr = arr_metadata["descr"]
            if "<" in descr:
                endian = model_pb2.DType.ByteOrder.LITTLE_ENDIAN_ORDER
            elif ">" in descr:
                endian = model_pb2.DType.ByteOrder.BIG_ENDIAN_ORDER
            elif "=" in descr:
                endian = sys.byteorder
                if endian == "big":
                    endian = model_pb2.DType.ByteOrder.BIG_ENDIAN_ORDER
                else:
                    endian = model_pb2.DType.ByteOrder.LITTLE_ENDIAN_ORDER
            else:
                endian = model_pb2.DType.ByteOrder.NA  # case "|"

            nparray_dtype = descr[1:]
            if nparray_dtype in ModelProtoMessages.TensorSpecProto.NUMPY_DATA_TYPE_TO_PROTO_LOOKUP:
                proto_data_type = \
                    ModelProtoMessages.TensorSpecProto.NUMPY_DATA_TYPE_TO_PROTO_LOOKUP[
                        nparray_dtype]
            else:
                raise RuntimeError(
                    "Provided data type: {}, is not supported".format(nparray_dtype))

            dtype = model_pb2.DType(
                type=proto_data_type, byte_order=endian, fortran_order=fortran_order)

            flatten_array_bytes = arr.flatten().tobytes()
            tensor_spec = model_pb2.TensorSpec(
                length=length, dimensions=dimensions, type=dtype, value=flatten_array_bytes)
            return tensor_spec

        @classmethod
        def get_numpy_data_type_from_tensor_spec(cls, tensor_spec):
            if tensor_spec.type.byte_order == model_pb2.DType.ByteOrder.BIG_ENDIAN_ORDER:
                endian_char = ">"
            elif tensor_spec.type.byte_order == model_pb2.DType.ByteOrder.LITTLE_ENDIAN_ORDER:
                endian_char = "<"
            else:
                endian_char = "|"

            data_type = tensor_spec.type.type
            fortran_order = tensor_spec.type.fortran_order
            np_data_type = \
                endian_char + \
                ModelProtoMessages.TensorSpecProto.INV_NUMPY_DATA_TYPE_TO_PROTO_LOOKUP[data_type]
            return np_data_type

        @classmethod
        def proto_tensor_spec_to_numpy_array(cls, tensor_spec):
            np_data_type = \
                ModelProtoMessages.TensorSpecProto.get_numpy_data_type_from_tensor_spec(
                    tensor_spec)
            dimensions = tensor_spec.dimensions
            value = tensor_spec.value
            length = tensor_spec.length

            np_array = np.frombuffer(
                buffer=value, dtype=np_data_type, count=length)
            np_array = np_array.reshape(dimensions)

            return np_array

        @classmethod
        def proto_tensor_spec_with_list_values_to_numpy_array(cls, tensor_spec, list_of_values):
            np_data_type = \
                ModelProtoMessages.TensorSpecProto.get_numpy_data_type_from_tensor_spec(
                    tensor_spec)
            dimensions = tensor_spec.dimensions

            np_array = np.array(list_of_values, dtype=np_data_type)
            np_array = np_array.reshape(dimensions)

            return np_array

    @classmethod
    def construct_tensor_pb(cls, nparray, ciphertext=None):
        # We prioritize the ciphertext over the plaintext.
        if not isinstance(nparray, np.ndarray):
            raise TypeError(
                "Parameter {} must be of type {}.".format(nparray, np.ndarray))

        tensor_spec = \
            ModelProtoMessages.TensorSpecProto.numpy_array_to_proto_tensor_spec(
                nparray)

        if ciphertext is not None:
            # If the tensor is a ciphertext we need to set the bytes of the
            # ciphertext as the value of the tensor not the numpy array bytes.
            tensor_spec.value = ciphertext
            tensor_pb = model_pb2.CiphertextTensor(tensor_spec=tensor_spec)
        else:
            tensor_pb = model_pb2.PlaintextTensor(tensor_spec=tensor_spec)
        return tensor_pb

    @classmethod
    def construct_model_variable_pb(cls, name, trainable, tensor_pb):
        assert isinstance(name, str) and isinstance(trainable, bool)
        if isinstance(tensor_pb, model_pb2.PlaintextTensor):
            return model_pb2.Model.Variable(name=name, trainable=trainable, plaintext_tensor=tensor_pb)
        elif isinstance(tensor_pb, model_pb2.CiphertextTensor):
            return model_pb2.Model.Variable(name=name, trainable=trainable, ciphertext_tensor=tensor_pb)
        else:
            raise RuntimeError(
                "Tensor proto message refers to a non-supported tensor protobuff datatype.")

    @classmethod
    def construct_model_pb_from_vars_pb(
            cls, variables_pb):
        assert isinstance(variables_pb, list) and \
            all([isinstance(var, model_pb2.Model.Variable)
                for var in variables_pb])
        return model_pb2.Model(variables=variables_pb)

    @classmethod
    def construct_federated_model_pb(cls, num_contributors, model_pb):
        assert isinstance(model_pb, model_pb2.Model)
        return model_pb2.FederatedModel(num_contributors=num_contributors, model=model_pb)

    @classmethod
    def construct_optimizer_config_pb(cls, optimizer_name, learning_rate, kwargs):
        return model_pb2.OptimizerConfig(
            name=optimizer_name, learning_rate=learning_rate, kwargs=kwargs)


class ServiceCommonProtoMessages(object):

    @classmethod
    def construct_ack_pb(cls, status, google_timestamp, message=None):
        return service_common_pb2.Ack(status=status, timestamp=google_timestamp, message=message)

    @classmethod
    def construct_get_services_health_status_request_pb(cls):
        return service_common_pb2.GetServicesHealthStatusRequest()

    @classmethod
    def construct_get_services_health_status_response_pb(cls, services_status):
        assert isinstance(services_status, dict)
        return service_common_pb2.GetServicesHealthStatusResponse(services_status=services_status)

    @classmethod
    def construct_shutdown_request_pb(cls):
        return service_common_pb2.ShutDownRequest()

    @classmethod
    def construct_shutdown_response_pb(cls, ack_pb):
        assert isinstance(ack_pb, service_common_pb2.Ack)
        return service_common_pb2.ShutDownResponse(ack=ack_pb)
