import numpy as np

from projectmetis.proto import controller_pb2, learner_pb2, model_pb2, metis_pb2, service_common_pb2


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
    def construct_server_entity_pb(cls, hostname, port):
        return metis_pb2.ServerEntity(hostname=hostname, port=port)

    @classmethod
    def construct_fhe_scheme_pb(cls, enabled=False, name=None, batch_size=None, scaling_bits=None, cryptocontext=None,
                                public_key=None, private_key=None):
        return metis_pb2.FHEScheme(enabled=enabled, name=name, batch_size=batch_size, scaling_bits=scaling_bits,
                                   cryptocontext=cryptocontext, public_key=public_key, private_key=private_key)

    @classmethod
    def construct_dataset_spec_pb(cls, num_training_examples, num_validation_examples, num_test_examples,
                                  training_spec, validation_spec, test_spec,
                                  is_classification=False, is_regression=False):
        if is_classification is True:
            training_spec = cls.construct_classification_dataset_spec_pb(training_spec)
            validation_spec = cls.construct_classification_dataset_spec_pb(validation_spec)
            test_spec = cls.construct_classification_dataset_spec_pb(test_spec)
            return metis_pb2.DatasetSpec(num_training_examples=num_training_examples,
                                         num_validation_examples=num_validation_examples,
                                         num_test_examples=num_test_examples,
                                         training_classification_spec=training_spec,
                                         validation_classification_spec=validation_spec,
                                         test_classification_spec=test_spec)
        elif is_regression:
            training_spec = cls.construct_regression_dataset_spec_pb(training_spec)
            validation_spec = cls.construct_regression_dataset_spec_pb(validation_spec)
            test_spec = cls.construct_regression_dataset_spec_pb(test_spec)
            return metis_pb2.DatasetSpec(num_training_examples=num_training_examples,
                                         num_validation_examples=num_validation_examples,
                                         num_test_examples=num_test_examples,
                                         training_regression_spec=training_spec,
                                         validation_regression_spec=validation_spec,
                                         test_regression_spec=test_spec)
        else:
            raise RuntimeError("Need to specify whether incoming dataset spec is regression or classification.")

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
                                       communication_specs_pb, model_hyperparams_pb):
        return metis_pb2.ControllerParams(server_entity=server_entity_pb, global_model_specs=global_model_specs_pb,
                                          communication_specs=communication_specs_pb,
                                          model_hyperparams_pb=model_hyperparams_pb)
    @classmethod
    def construct_controller_modelhyperparams_pb(cls, batch_size, epochs, optimizer_pb, percent_validation):
        return metis_pb2.ControllerParams.ModelHyperparams(batch_size=batch_size,
                                                           epochs=epochs,
                                                           optimizer=optimizer_pb,
                                                           percent_validation=percent_validation)

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

    @classmethod
    def construct_tensor_pb_from_nparray(cls, nparray, ciphertext=None):
        if not isinstance(nparray, np.ndarray):
            raise TypeError("Parameter {} must be of type {}.".format(nparray, np.ndarray))
        size = int(nparray.size)
        shape = nparray.shape
        if ciphertext is not None:
            tensor_spec = model_pb2.TensorSpec(length=size, dimensions=shape, dtype=model_pb2.TensorSpec.DType.UNKNOWN)
            tensor_pb = model_pb2.CiphertextTensor(spec=tensor_spec, values=ciphertext)
        else:
            dtype = str(nparray.dtype.name)
            values = nparray.flatten()
            if "int" in dtype:
                tensor_spec = model_pb2.TensorSpec(length=size, dimensions=shape, dtype=model_pb2.TensorSpec.DType.INT)
                tensor_pb = model_pb2.IntTensor(spec=tensor_spec, values=values)
            elif "long" in dtype:
                tensor_spec = model_pb2.TensorSpec(length=size, dimensions=shape, dtype=model_pb2.TensorSpec.DType.LONG)
                tensor_pb = model_pb2.IntTensor(spec=tensor_spec, values=values)
            elif "float32" in dtype:
                # The default dtype for Tensorflow and PyTorch weights is float32.
                tensor_spec = model_pb2.TensorSpec(length=size, dimensions=shape, dtype=model_pb2.TensorSpec.DType.FLOAT)
                tensor_pb = model_pb2.FloatTensor(spec=tensor_spec, values=values)
            elif "float" in dtype:
                # The default dtype for numpy arrays is float64, also represented as float.
                tensor_spec = model_pb2.TensorSpec(length=size, dimensions=shape, dtype=model_pb2.TensorSpec.DType.DOUBLE)
                tensor_pb = model_pb2.DoubleTensor(spec=tensor_spec, values=values)
            else:
                raise RuntimeError("Provided data type: {}, is not supported".format(dtype))

        return tensor_pb

    @classmethod
    def construct_model_variable_pb(cls, name, trainable, tensor_pb):
        assert isinstance(name, str) and isinstance(trainable, bool)
        if isinstance(tensor_pb, model_pb2.IntTensor):
            return model_pb2.Model.Variable(name=name, trainable=trainable, int_tensor=tensor_pb)
        elif isinstance(tensor_pb, model_pb2.FloatTensor):
            return model_pb2.Model.Variable(name=name, trainable=trainable, float_tensor=tensor_pb)
        elif isinstance(tensor_pb, model_pb2.DoubleTensor):
            return model_pb2.Model.Variable(name=name, trainable=trainable, double_tensor=tensor_pb)
        elif isinstance(tensor_pb, model_pb2.CiphertextTensor):
            return model_pb2.Model.Variable(name=name, trainable=trainable, ciphertext_tensor=tensor_pb)
        else:
            raise RuntimeError("Tensor proto message refers to a non-supported tensor protobuff datatype.")

    @classmethod
    def construct_model_pb(cls, variables_pb):
        assert isinstance(variables_pb, list) and \
               all([isinstance(var, model_pb2.Model.Variable) for var in variables_pb])
        return model_pb2.Model(variables=variables_pb)

    @classmethod
    def construct_federated_model_pb(cls, num_contributors, model_pb):
        assert isinstance(model_pb, model_pb2.Model)
        return model_pb2.FederatedModel(num_contributors=num_contributors, model=model_pb)

    @classmethod
    def construct_vanilla_sgd_optimizer_pb(cls, learning_rate, l1_reg=0.0, l2_reg=0.0):
        return model_pb2.VanillaSGD(learning_rate=learning_rate, L1_reg=l1_reg, L2_reg=l2_reg)

    @classmethod
    def construct_momentum_sgd_optimizer_pb(cls, learning_rate, momentum_factor=0.0):
        return model_pb2.MomentumSGD(learning_rate=learning_rate, momentum_factor=momentum_factor)

    @classmethod
    def construct_fed_prox_optimizer_pb(cls, learning_rate, proximal_term=0.0):
        return model_pb2.FedProx(learning_rate=learning_rate, proximal_term=proximal_term)

    @classmethod
    def construct_adam_optimizer_pb(cls, learning_rate, beta_1=0.0, beta_2=0.0, epsilon=0.0):
        return model_pb2.Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon)

    @classmethod
    def construct_optimizer_config_pb(cls, optimizer_pb):
        if isinstance(optimizer_pb, model_pb2.VanillaSGD):
            return model_pb2.OptimizerConfig(vanilla_sgd=optimizer_pb)
        elif isinstance(optimizer_pb, model_pb2.MomentumSGD):
            return model_pb2.OptimizerConfig(momentum_sgd=optimizer_pb)
        elif isinstance(optimizer_pb, model_pb2.FedProx):
            return model_pb2.OptimizerConfig(fed_prox=optimizer_pb)
        elif isinstance(optimizer_pb, model_pb2.Adam):
            return model_pb2.OptimizerConfig(adam=optimizer_pb)
        else:
            raise RuntimeError("Optimizer proto message refers to a non-supported optimizer.")


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
