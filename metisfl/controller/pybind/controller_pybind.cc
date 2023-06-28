
#include <iostream>
#include <pybind11/pybind11.h>
#include <string>

#include "metisfl/controller/core/controller.h"
#include "metisfl/controller/core/controller_servicer.h"

namespace py = pybind11;

namespace metisfl::controller {

using metisfl::controller::Controller;
using metisfl::controller::ControllerServicer;

class ControllerWrapper {

 public:
  ~ControllerWrapper() = default;

  void Start(std::string params_serialized) {
    metisfl::ControllerParams params;
    params.ParseFromString(params_serialized);
    controller_ = Controller::New(params);
    servicer_ = ControllerServicer::New(controller_.get());
    servicer_->StartService();
  }

  void Shutdown() {
    PLOG(INFO) << "Wrapping up resources and shutting down..";
    servicer_->StopService();
    servicer_->WaitService();
  }

  bool ShutdownRequestReceived() {
    return servicer_->ShutdownRequestReceived();
  }

  void Wait() {
    servicer_->WaitService();
  }

 private:
  std::unique_ptr<Controller> controller_;
  std::unique_ptr<ControllerServicer> servicer_;

};

} // namespace metisfl::controller

PYBIND11_MODULE(controller, m) {
  m.doc() = "Federation controller python soft wrapper.";

  py::class_<metisfl::controller::ControllerWrapper>(m, "ControllerWrapper")
    .def(py::init<>())
    .def("start",
        &metisfl::controller::ControllerWrapper::Start,
        "Initializes and starts the controller.")
    .def("shutdown",
        &metisfl::controller::ControllerWrapper::Shutdown,
        "Shuts down the controller.")
    .def("shutdown_request_received",
        &metisfl::controller::ControllerWrapper::ShutdownRequestReceived,
        "Check if controller has already received a shutdown request.")
    .def("wait",
        &metisfl::controller::ControllerWrapper::Wait,
        "Wait for controller main thread.");
}
