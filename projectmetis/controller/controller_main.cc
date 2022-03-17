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
//#include <csignal>
#include <signal.h>

#include "projectmetis/controller/controller.h"
#include "projectmetis/controller/controller_servicer.h"

#include "projectmetis/core/macros.h"

using ::proto::ParseTextOrDie;

using projectmetis::controller::Controller;
using projectmetis::controller::ControllerServicer;
using projectmetis::GlobalModelSpecs;
using projectmetis::CommunicationSpecs;

std::unique_ptr<ControllerServicer> servicer;

void sigint_handler(int code) {
  std::cout << "Received SIGINT (code " << code << ")" << std::endl;
  if (servicer != nullptr) {
    servicer->StopService();
  }
  //exit(code);
}

int main(int argc, char **argv) {
  // Initializes controller parameters proto message.
  auto params = ParseTextOrDie<projectmetis::ControllerParams>(R"pb2(
    server_entity {
      hostname: "0.0.0.0"
      port: 50051
      ssl_config {
            server_key: "/resources/ssl/server-key.pem"
            server_cert: "/resources/ssl/server-cert.pem"
        }
    }
    global_model_specs {
      learners_participation_ratio: 1
      aggregation_rule: FED_AVG
    }
    communication_specs {
      protocol: SYNCHRONOUS
    }
    model_hyperparams {
      batch_size: 1
      epochs: 1
      optimizer {
        vanilla_sgd {
          learning_rate: 0.05
          L2_reg: 0.001
        }
      }
      percent_validation: 0
    }
  )pb2");

  std::cout << "Starting controller with params: " << std::endl;
  std::cout << params.DebugString() << std::endl;

  signal(SIGINT, sigint_handler);

  auto controller = Controller::New(params);
  servicer = ControllerServicer::New(controller.get());

  servicer->StartService();
  servicer->WaitService();

  std::cout << "Exiting... Bye!" << std::endl;
  return 0;
}
