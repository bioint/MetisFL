// MIT License
//
// Copyright (c) 2021 Project Metis
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "projectmetis/controller/controller_servicer.h"

#include <iostream>
#include <memory>

#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>

#include "absl/strings/str_cat.h"
#include "projectmetis/controller/controller.h"
#include "projectmetis/proto/controller.grpc.pb.h"

namespace projectmetis::controller {
using ::grpc::Server;
using ::grpc::ServerBuilder;
using ::grpc::ServerContext;
using ::grpc::Status;

void RunServer(ControllerParams &params) {
  // Creates a controller.
  auto controller = Controller::New(params);
  auto service = ControllerServicer::New(controller.get());

  const auto server_address = absl::StrCat(params.server_entity().hostname(),
                                           ":", params.server_entity().port());

  grpc::EnableDefaultHealthCheckService(true);
  grpc::reflection::InitProtoReflectionServerBuilderPlugin();
  ServerBuilder builder;
  // Listens on the given address without any authentication mechanism.
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  // Registers "service" as the instance through which we'll communicate with
  // clients. In this case it corresponds to an *synchronous* service.
  builder.RegisterService(service.get());
  // Finally assemble the server.
  std::unique_ptr<Server> server(builder.BuildAndStart());
  std::cout << "Controller listening on " << server_address << std::endl;

  // Waits for the server to shutdown. Note that some other thread must be
  // responsible for shutting down the server for this call to ever return.
  server->Wait();
}

} // namespace projectmetis::controller

int main(int argc, char **argv) {
  // Initializes controller parameters proto message.
  projectmetis::ControllerParams controller_params;
  controller_params.mutable_server_entity()->set_hostname("0.0.0.0");
  controller_params.mutable_server_entity()->set_port(50051);
  controller_params.mutable_global_model_specs()
      ->set_learners_participation_ratio(1);
  controller_params.mutable_global_model_specs()->set_aggregation_rule(
      projectmetis::GlobalModelSpecs_AggregationRule_FED_AVG);
  controller_params.mutable_global_model_specs()
      ->set_learners_participation_ratio(1);
  controller_params.mutable_communication_protocol_specs()->set_protocol(
      projectmetis::CommunicationProtocolSpecs_Protocol_SYNCHRONOUS);
  controller_params.set_federated_execution_cutoff_mins(200);
  controller_params.set_federated_execution_cutoff_score(0.85);
  projectmetis::controller::RunServer(controller_params);

  return 0;
}
