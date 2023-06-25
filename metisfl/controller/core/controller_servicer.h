
#ifndef METISFL_METISFL_CONTROLLER_CORE_CONTROLLER_SERVICER_H_
#define METISFL_METISFL_CONTROLLER_CORE_CONTROLLER_SERVICER_H_

#include <memory>
#include <sstream>
#include <fstream>
#include <filesystem>

#include "metisfl/controller/core/controller.h"
#include "metisfl/proto/controller.grpc.pb.h"

namespace metisfl::controller {

class ControllerServicer : public ControllerService::Service {
public:
  ABSL_MUST_USE_RESULT
  virtual const Controller *GetController() const = 0;

  // Starts the gRPC service.
  virtual void StartService() = 0;

  // Waits for the gRPC service to shut down.
  virtual void WaitService() = 0;

  // Stops the gRPC service.
  virtual void StopService() = 0;

public:
  static std::unique_ptr<ControllerServicer> New(Controller *controller);
};

} // namespace metisfl::controller

#endif //METISFL_METISFL_CONTROLLER_CORE_CONTROLLER_SERVICER_H_
