
#ifndef PROJECTMETIS_RC_PROJECTMETIS_CONTROLLER_CONTROLLER_SERVICER_H_
#define PROJECTMETIS_RC_PROJECTMETIS_CONTROLLER_CONTROLLER_SERVICER_H_

#include <memory>
#include <sstream>
#include <fstream>
#include <filesystem>

#include "metisfl/controller/core/controller.h"
#include "metisfl/proto/controller.grpc.pb.h"

namespace projectmetis::controller {

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

} // namespace projectmetis::controller

#endif // PROJECTMETIS_RC_PROJECTMETIS_CONTROLLER_CONTROLLER_SERVICER_H_
