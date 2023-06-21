
#include <csignal>
#include <iostream>
#include <pybind11/pybind11.h>
#include <string>

#include "metisfl/controller/core/controller.h"
#include "metisfl/controller/core/controller_servicer.h"

namespace py = pybind11;

namespace projectmetis::controller {

using projectmetis::controller::Controller;
using projectmetis::controller::ControllerServicer;

class ServicerWrapper {

public:
    ~ServicerWrapper() = default;

    void BuildAndStart(std::string params_serialized) {
      projectmetis::ControllerParams params;
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

} // namespace projectmetis::controller

PYBIND11_MODULE(controller, m) {
  m.doc() = "Federation controller python soft wrapper.";

  py::class_<projectmetis::controller::ServicerWrapper>(m, "ServicerWrapper")
    .def(py::init<>())
    .def("BuildAndStart",
        &projectmetis::controller::ServicerWrapper::BuildAndStart,
        "Initializes and starts the controller.")
    .def("Wait",
        &projectmetis::controller::ServicerWrapper::Wait,
        "Blocks until the service has shut down.")
    .def("Shutdown",
        &projectmetis::controller::ServicerWrapper::Shutdown,
        "Shuts down the controller.");
}