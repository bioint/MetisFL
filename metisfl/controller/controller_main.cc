
/**
 * A standalone controller instance example running at port 50051.
 */

#include <memory>
#include <signal.h>
#include <filesystem>

#include <glog/logging.h>

#include "metisfl/controller/core/controller.h"
#include "metisfl/controller/core/controller_servicer.h"
#include "metisfl/controller/common/macros.h"

using ::proto::ParseTextOrDie;

using metisfl::controller::Controller;
using metisfl::controller::ControllerServicer;
using metisfl::GlobalModelSpecs;
using metisfl::CommunicationSpecs;

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
  auto params = ParseTextOrDie<metisfl::ControllerParams>(R"pb2(
    server_entity {
      hostname: "0.0.0.0"
      port: 50051
      ssl_config {
        enable: true
        ssl_config_files {
          public_certificate_file: "metisfl/resources/ssl/ca-cert.pem"
          private_key_file: "metisfl/resources/ssl/ca-key.pem"
        }
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
