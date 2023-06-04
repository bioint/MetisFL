
/**
 * A standalone controller instance running at port 50051.
 */

#include <memory>
#include <signal.h>

#include <glog/logging.h>

#include "src/cc/controller/controller.h"
#include "src/cc/controller/controller_servicer.h"
#include "src/cc/core/macros.h"

using ::proto::ParseTextOrDie;

using projectmetis::controller::Controller;
using projectmetis::controller::ControllerServicer;
using projectmetis::GlobalModelSpecs;
using projectmetis::CommunicationSpecs;

std::unique_ptr<ControllerServicer> servicer;

void sigint_handler(int code) {
  PLOG(INFO) << "Received SIGINT (code " << code << ")" << std::endl;
  if (servicer != nullptr) {
    servicer->StopService();
  }
}

int main(int argc, char **argv) {

  // Set flags picked up by glog before initialization.
  FLAGS_log_dir = "/tmp";
  FLAGS_alsologtostderr = true;
  // According to the ISO C11 standard, the first argument,
  // argv[0], represents the program name. We use this name
  // to decorate the log entries of glog.
  google::InitGoogleLogging(argv[0]);

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
      aggregation_rule {
        fed_avg {}
        aggregation_rule_specs {
          scaling_factor: NUM_TRAINING_EXAMPLES
        }
      }
    }
    model_store_config {
      in_memory_store {
        model_store_specs {
          lineage_length_eviction {
            lineage_length: 1
          }
        }
      }      
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

  PLOG(INFO) << "Starting controller with params: ";
  PLOG(INFO) << params.DebugString();

  signal(SIGINT, sigint_handler);

  auto controller = Controller::New(params);
  servicer = ControllerServicer::New(controller.get());

  servicer->StartService();
  servicer->WaitService();

  PLOG(INFO) << "Exiting... Bye!";

  return 0;
}
