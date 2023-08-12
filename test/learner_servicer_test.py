import argparse
import numpy as np
import datetime
import time

from examples.neuroimaging.brainage_cnns import BrainAge3DCNN
from examples.cifar.model import CifarCNN
from examples.fashionmnist.model import FashionMnistModel

from examples.scalability.housing_mlp import HousingMLP
from metisfl.proto import learner_pb2_grpc
from metisfl.grpc.client import GRPCChannelMaxMsgLength, GRPCServerMaxMsgLength
from metisfl.proto.proto_messages_factory import \
    MetisProtoMessages, ModelProtoMessages, LearnerServiceProtoMessages, ServiceCommonProtoMessages
from metisfl.proto import metis_pb2


def check_health_status(channel):
    get_services_health_status_request_pb = \
        ServiceCommonProtoMessages.construct_get_services_health_status_request_pb()
    print("Checking learner health status.", flush=True)
    stub = learner_pb2_grpc.LearnerServiceStub(channel)
    response = stub.GetServicesHealthStatus(get_services_health_status_request_pb)
    print("Health status response: {}".format(response), flush=True)
    return response


def generate_model_request(model_vars, batch_size):
    model_pb = ModelProtoMessages.construct_model_pb(model_vars)
    metrics_pb = MetisProtoMessages.construct_evaluation_metrics_pb()
    evaluate_model_request_pb = LearnerServiceProtoMessages.construct_evaluate_model_request_pb(
        model=model_pb, batch_size=batch_size, eval_train=True,
        eval_test=True, eval_valid=True, metrics_pb=metrics_pb)
    return evaluate_model_request_pb


def evaluate_model(stub, nn_model):
    model_weights = nn_model.get_weights()
    batch_size = 100

    print(datetime.datetime.now(), "Constructing model variables protobuf.")
    model_vars = []
    for widx, weight in enumerate(model_weights):
        tensor_pb = ModelProtoMessages.construct_tensor_pb(weight)
        model_var = ModelProtoMessages.construct_model_variable_pb(name="arr_{}".format(widx),
                                                                   trainable=True,
                                                                   tensor_pb=tensor_pb)
        model_vars.append(model_var)
    print(datetime.datetime.now(), "Constructed model variables protobuf.")

    evaluate_model_request_pb = generate_model_request(model_vars, batch_size)
    print(datetime.datetime.now(), "Sending model evaluation request.")
    response = stub.EvaluateModel(evaluate_model_request_pb)
    print(datetime.datetime.now(), "Sent model evaluation request.")
    print(response)


def run_task(stub, epochs=1):
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
        tensor_pb = ModelProtoMessages.construct_tensor_pb(weight)
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
        num_local_updates=int(num_epochs * (total_examples / batch_size)),
        validation_dataset_pct=0.0)
    run_task_request_pb = LearnerServiceProtoMessages.construct_run_task_request_pb(
        federated_model_pb=federated_model_pb,
        learning_task_pb=learning_task_pb,
        hyperparameters_pb=hyperparameters_pb)

    response = stub.RunTask(run_task_request_pb)
    print(response)


def shutdown(stub):
    shutdown_request_pb = ServiceCommonProtoMessages.construct_shutdown_request_pb()
    response = stub.ShutDown(shutdown_request_pb)
    print(response)


class LearnerServicer(learner_pb2_grpc.LearnerServiceServicer):

    def __init__(self, host, port):
        server = GRPCServerMaxMsgLength(max_workers=None).server
        learner_pb2_grpc.add_LearnerServiceServicer_to_server(self, server)
        server.add_insecure_port("{}:{}".format(host, port))
        server.start()
        print("Initialized Learner Servicer.")
        time.sleep(10000)

    def EvaluateModel(self, request, context):
        print(datetime.datetime.now(), "Learner Servicer received model evaluation task.")
        model_pb = request.model
        model_variables = [np.frombuffer(v.float_tensor.values, dtype="float32") for v in model_pb.variables]
        total_params = sum([v.size for v in model_variables])
        print(total_params)
        batch_size = request.batch_size
        metrics_pb = request.metrics
        evaluation_dataset_pb = request.evaluation_dataset
        evaluate_model_response_pb = \
            LearnerServiceProtoMessages.construct_evaluate_model_response_pb(
                metis_pb2.ModelEvaluations())
        return evaluate_model_response_pb


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--initialize_server", default="False", type=str)
    args = parser.parse_args()

    if args.initialize_server == "True":
        LearnerServicer("localhost", 50052)
    else:
        channel = GRPCChannelMaxMsgLength('127.0.0.1', 50052).channel
        # for i in range(1):
        #     time.sleep(5)
        #     run_task(channel, epochs=i)
        # run_task(channel, epochs=1)

        # nn_model = BrainAge3DCNN().get_model()
        # nn_model = CifarCNN().get_model()
        # nn_model = FashionMnistModel().get_model()
        stub = learner_pb2_grpc.LearnerServiceStub(channel)

        start_time = datetime.datetime.now()
        print("Start Time: ", start_time)
        nn_model = HousingMLP(params_per_layer=30).get_model()
        nn_model.summary()
        evaluate_model(stub, nn_model)
        end_time = datetime.datetime.now()
        print("End Time: ", end_time)
        print("Difference:", (end_time - start_time).total_seconds(), "secs")

        start_time = datetime.datetime.now()
        print("Start Time: ", start_time)
        nn_model = HousingMLP(params_per_layer=10000).get_model()
        nn_model.summary()
        evaluate_model(stub, nn_model)
        end_time = datetime.datetime.now()
        print("End Time: ", end_time)
        print("Difference:", (end_time - start_time).total_seconds(), "secs")

        # check_health_status(channel)
        # shutdown(channel)
