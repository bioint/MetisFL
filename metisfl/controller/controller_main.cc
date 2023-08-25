
/**
 * A standalone controller instance example running at port 50051.
 */

#include <glog/logging.h>
#include <signal.h>

#include <filesystem>
#include <memory>

#include "metisfl/controller/common/macros.h"
#include "metisfl/controller/core/controller.h"
#include "metisfl/controller/core/controller_servicer.h"

using metisfl::CommunicationSpecs;
using metisfl::GlobalModelSpecs;
using metisfl::controller::Controller;
using metisfl::controller::ControllerServicer;

std::unique_ptr<ControllerServicer> servicer;

void sigint_handler(int code) {
  PLOG(INFO) << "Received SIGINT (code " << code << ")" << std::endl;
  if (servicer != nullptr) {
    servicer->StopService();
  }
}

int main(int argc, char **argv) {
  FLAGS_log_dir = "/tmp";
  FLAGS_alsologtostderr = true;
  google::InitGoogleLogging(argv[0]);

  PLOG(INFO) << "Starting controller with params: ";
  PLOG(INFO) << params.DebugString();

  signal(SIGINT, sigint_handler);

  // FIXME: Add params

  auto controller = Controller::New(params);
  servicer = ControllerServicer::New(controller.get());

  servicer->StartService();
  servicer->WaitService();

  PLOG(INFO) << "Exiting... Bye!";

  return 0;
}
