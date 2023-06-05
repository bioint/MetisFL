import time

from experiments.keras.models.brainage_3dcnn import BrainAge3DCNN
from experiments.keras.models.cifar_cnn import CifarCNN
from experiments.keras.models.fashion_mnist_fc import FashionMnistModel
from projectmetis.proto import learner_pb2_grpc
from projectmetis.python.utils.grpc_services import GRPCChannelMaxMsgLength
from projectmetis.python.utils.proto_messages_factory import \
    MetisProtoMessages, ModelProtoMessages, LearnerServiceProtoMessages, ServiceCommonProtoMessages


def check_health_status(channel):
    get_services_health_status_request_pb = \
        ServiceCommonProtoMessages.construct_get_services_health_status_request_pb()
    print("Checking learner health status.", flush=True)
    stub = learner_pb2_grpc.LearnerServiceStub(channel)
    response = stub.GetServicesHealthStatus(get_services_health_status_request_pb)
    print("Health status response: {}".format(response), flush=True)
    return response


def evaluate_model(channel):
    # model_weights = BrainAge3DCNN().get_model().get_weights()
    # batch_size = 1
    model_weights = FashionMnistModel().get_model().get_weights()
    batch_size = 100
    # model_weights = CifarCNN().get_model().get_weights()
    # batch_size = 100

    model_vars = []
    for widx, weight in enumerate(model_weights):
        tensor_pb = ModelProtoMessages.construct_tensor_pb_from_nparray(weight)
        model_var = ModelProtoMessages.construct_model_variable_pb(name="arr_{}".format(widx),
                                                                   trainable=True,
                                                                   tensor_pb=tensor_pb)
        model_vars.append(model_var)

    model_pb = ModelProtoMessages.construct_model_pb(model_vars)
    metrics_pb = MetisProtoMessages.construct_evaluation_metrics_pb()
    evaluate_model_request_pb = LearnerServiceProtoMessages.construct_evaluate_model_request_pb(
        model=model_pb, batch_size=batch_size, eval_train=True,
        eval_test=True, eval_valid=True, metrics_pb=metrics_pb)
    stub = learner_pb2_grpc.LearnerServiceStub(channel)
    response = stub.EvaluateModel(evaluate_model_request_pb)
    print(response)


def run_task(channel, epochs=1):
    # BrainAge weights, samples, batch_size
    # model_weights = BrainAge3DCNN().get_model().get_weights()
    # total_examples = 21
    # batch_size = 100

    # FashionMNIST weights, samples, batch_size
    model_weights = FashionMnistModel().get_model().get_weights()
    total_examples = 54000
    batch_size = 100

    # Cifar10 weights, samples, batch_size
    # model_weights = CifarCNN().get_model().get_weights()

    num_epochs = epochs
    model_vars = []
    for widx, weight in enumerate(model_weights):
        # tensor_pb = ModelProtoMessages.construct_tensor_pb_from_nparray(np.array([1.0, 3.0, 4.0]))
        tensor_pb = ModelProtoMessages.construct_tensor_pb_from_nparray(weight)
        # model_var = ModelProtoMessages.construct_model_variable_pb(name="arr1", trainable=True, tensor_pb=tensor_pb)
        model_var = ModelProtoMessages.construct_model_variable_pb(name="arr_{}".format(widx),
                                                                   trainable=True,
                                                                   tensor_pb=tensor_pb)
        model_vars.append(model_var)

    model_pb = ModelProtoMessages.construct_model_pb(model_vars)
    federated_model_pb = ModelProtoMessages.construct_federated_model_pb(10, model_pb)
    optimizer_pb = ModelProtoMessages.construct_vanilla_sgd_optimizer_pb(learning_rate=0.001)
    optimizer_config_pb = ModelProtoMessages.construct_optimizer_config_pb(optimizer_pb)
    hyperparameters_pb = MetisProtoMessages.construct_hyperparameters_pb(
        batch_size=batch_size,
        optimizer_config_pb=optimizer_config_pb)
    learning_task_pb = MetisProtoMessages.construct_learning_task_pb(
        num_local_updates=int(num_epochs*(total_examples/batch_size)),
        validation_dataset_pct=0.0)
    run_task_request_pb = LearnerServiceProtoMessages.construct_run_task_request_pb(
        federated_model_pb=federated_model_pb,
        learning_task_pb=learning_task_pb,
        hyperparameters_pb=hyperparameters_pb)

    stub = learner_pb2_grpc.LearnerServiceStub(channel)
    response = stub.RunTask(run_task_request_pb)
    print(response)


def shutdown(channel):
    stub = learner_pb2_grpc.LearnerServiceStub(channel)
    shutdown_request_pb = ServiceCommonProtoMessages.construct_shutdown_request_pb()
    response = stub.ShutDown(shutdown_request_pb)
    print(response)


if __name__ == '__main__':
    channel = GRPCChannelMaxMsgLength('localhost', 50052).channel
    # channel = GRPCChannelMaxMsgLength('axon.isi.edu', 4224).channel
    # for i in range(1):
    #     time.sleep(5)
    #     run_task(channel, epochs=i)
    # run_task(channel, epochs=1)
    # evaluate_model(channel)
    # check_health_status(channel)
    # shutdown(channel)
