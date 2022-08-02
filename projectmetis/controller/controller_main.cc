
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
