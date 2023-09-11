
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

void sigint_handler(int code) {
  LOG(INFO) << "Received SIGINT (code " << code << ")" << std::endl;
}

int main(int argc, char **argv) {
  FLAGS_log_dir = "/tmp";
  FLAGS_alsologtostderr = true;
  google::InitGoogleLogging(argv[0]);

  LOG(INFO) << "Starting controller with params: ";

  signal(SIGINT, sigint_handler);

  // FIXME: Add params

  LOG(INFO) << "Exiting... Bye!";

  return 0;
}
