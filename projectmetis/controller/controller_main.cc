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

/**
 * A standalone controller instance running at port 50051.
 */

#include <memory>
#include <csignal>

#include "projectmetis/controller/controller.h"
#include "projectmetis/controller/controller_servicer.h"

using projectmetis::controller::Controller;
using projectmetis::controller::ControllerServicer;
using projectmetis::GlobalModelSpecs;
using projectmetis::CommunicationProtocolSpecs;

std::unique_ptr<ControllerServicer> servicer;

void sigint_handler(int code) {
  std::cout << "Received SIGINT (code " << code << ")" << std::endl;
  if (servicer != nullptr) {
    servicer->StopService();
  }
  exit(code);
}

int main(int argc, char **argv) {
  // Initializes controller parameters proto message.
  projectmetis::ControllerParams params;
  params.mutable_server_entity()->set_hostname("0.0.0.0");
  params.mutable_server_entity()->set_port(50051);
  params.mutable_global_model_specs()
      ->set_learners_participation_ratio(1);
  params.mutable_global_model_specs()->set_aggregation_rule(
      projectmetis::GlobalModelSpecs::FED_AVG);
  params.mutable_communication_protocol_specs()->set_protocol(
      projectmetis::CommunicationProtocolSpecs::SYNCHRONOUS);
  params.set_federated_execution_cutoff_mins(200);
  params.set_federated_execution_cutoff_score(0.85);

  {
    struct sigaction sigIntHandler;

    sigIntHandler.sa_handler = sigint_handler;
    sigemptyset(&sigIntHandler.sa_mask);
    sigIntHandler.sa_flags = 0;

    sigaction(SIGINT, &sigIntHandler, nullptr);
  }

  auto controller = Controller::New(params);
  servicer = ControllerServicer::New(controller.get());

  servicer->StartService();
  servicer->WaitService();

  return 0;
}
