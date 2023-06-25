
#include <csignal>
#include <iostream>
#include <pybind11/pybind11.h>
#include <string>

#include "metisfl/controller/core/controller.h"
#include "metisfl/controller/core/controller_servicer.h"

namespace py = pybind11;

namespace metisfl::controller {

using metisfl::controller::Controller;
using metisfl::controller::ControllerServicer;

class ServicerWrapper {

public:
    ~ServicerWrapper() = default;

    void BuildAndStart(std::string params_serialized) {
      metisfl::ControllerParams params;
      params.ParseFromString(params_serialized);
      controller_ = Controller::New(params);
      servicer_ = ControllerServicer::New(controller_.get());
      servicer_->StartService();
    };

    void Wait() {
      servicer_->WaitService();
    }

    void Shutdown() {
      PLOG(INFO) << "Shutting down..";
      servicer_->StopService();
    };

private:
  std::unique_ptr<Controller> controller_;
  std::unique_ptr<ControllerServicer> servicer_;

};

} // namespace metisfl::controller

PYBIND11_MODULE(controller, m) {
  m.doc() = "Federation controller python soft wrapper.";

  py::class_<metisfl::controller::ServicerWrapper>(m, "ServicerWrapper")
    .def(py::init<>())
    .def("BuildAndStart",
        &metisfl::controller::ServicerWrapper::BuildAndStart,
        "Initializes and starts the controller.")
    .def("Wait",
        &metisfl::controller::ServicerWrapper::Wait,
        "Blocks until the service has shut down.")
    .def("Shutdown",
        &metisfl::controller::ServicerWrapper::Shutdown,
        "Shuts down the controller.");
}