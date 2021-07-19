//
// MIT License
//
// Copyright (c) 2021 Project Metis
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <pybind11/pybind11.h>
#include <string>
#include <iostream>

#include "projectmetis/controller/controller.h"
#include "projectmetis/controller/controller_servicer.h"

namespace py = pybind11;

namespace projectmetis::controller {
using projectmetis::controller::Controller;
using projectmetis::controller::ControllerServicer;

struct ServicerWrapper {
  std::unique_ptr<Controller> controller;
  std::unique_ptr<ControllerServicer> servicer;
};

ServicerWrapper BuildAndStart(std::string &params_serialized) {
  projectmetis::ControllerParams params;
  params.ParseFromString(params_serialized);
  struct ServicerWrapper wrapper;

  wrapper.controller = Controller::New(params);
  wrapper.servicer = ControllerServicer::New(wrapper.controller.get());

  wrapper.servicer->StartService();

  return wrapper;
}

void Wait(ServicerWrapper &wrapper) { wrapper.servicer->WaitService(); }

void Shutdown(ServicerWrapper &wrapper) {
  std::cout << "received shutdown signal.." << std::endl;
  wrapper.servicer->StopService();
  wrapper.servicer = nullptr;
  wrapper.controller = nullptr;
}

} // namespace projectmetis::controller

PYBIND11_MODULE(controller, m) {
  m.doc() = "Federation controller python soft wrapper.";

  py::class_<projectmetis::controller::ServicerWrapper>(m, "ServicerWrapper");
  m.def("BuildAndStart",
        &projectmetis::controller::BuildAndStart,
        "Initializes and starts the controller.");
  m.def("Wait",
        &projectmetis::controller::Wait,
        "Blocks until the service has shut down.");
  m.def("Shutdown",
        &projectmetis::controller::Shutdown,
        "Shuts down the controller.");
}