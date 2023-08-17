
#ifndef METISFL_METISFL_CONTROLLER_CORE_CONTROLLER_SERVICER_H_
#define METISFL_METISFL_CONTROLLER_CORE_CONTROLLER_SERVICER_H_

#include <memory>
#include <sstream>
#include <fstream>
#include <filesystem>

#include "metisfl/controller/core/controller.h"
#include "metisfl/proto/controller.grpc.pb.h"

namespace metisfl::controller
{
  class ControllerServicer : public ControllerService::Service
  {
  public:
    ABSL_MUST_USE_RESULT
    virtual const Controller *GetController() const = 0;

    virtual void StartService() = 0;

    virtual void WaitService() = 0;

    virtual void StopService() = 0;

    virtual bool ShutdownRequestReceived() = 0;

  public:
    static std::unique_ptr<ControllerServicer> New(Controller *controller);
  };

} // namespace metisfl::controller

#endif // METISFL_METISFL_CONTROLLER_CORE_CONTROLLER_SERVICER_H_
