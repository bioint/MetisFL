
#include <glog/logging.h>
#include <pybind11/pybind11.h>

#include <iostream>
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

  void StartWrapper(const py::kwargs& params) {
    ServerParams server_params = {};
    server_params.hostname = params["hostname"].cast<std::string>();
    server_params.port = params["port"].cast<int>();
    server_params.root_certificate =
        params["root_certificate"].cast<std::string>();
    server_params.public_certificate =
        params["server_certificate"].cast<std::string>();
    server_params.private_key = params["private_key"].cast<std::string>();

    GlobalTrainParams global_train_params = {};
    global_train_params.aggregation_rule =
        params["aggregation_rule"].cast<std::string>();
    global_train_params.scheduler =
        params["scheduler"].cast<std::string>();
    global_train_params.scaling_factor =
        params["scaling_factor"].cast<std::string>();
    global_train_params.participation_ratio =
        params["participation_ratio"].cast<float>();
    global_train_params.stride_length = params["stride_length"].cast<int>();
    global_train_params.he_batch_size = params["he_batch_size"].cast<int>();
    global_train_params.he_scaling_factor_bits =
        params["he_scaling_factor_bits"].cast<int>();
    global_train_params.he_crypto_context_file =
        params["he_crypto_context_file"].cast<std::string>();
    global_train_params.semi_sync_lambda =
        params["semi_sync_lambda"].cast<float>();
    global_train_params.semi_sync_recompute_num_updates =
        params["semi_sync_recompute_num_updates"].cast<int>();

    ModelStoreParams model_store_params = {};
    model_store_params.model_store = params["model_store"].cast<std::string>();
    model_store_params.lineage_length = params["lineage_length"].cast<int>();
    model_store_params.hostname =
        params["model_store_hostname"].cast<std::string>();
    model_store_params.port = params["model_store_port"].cast<int>();

    Start(server_params, global_train_params, model_store_params);
  }

  void Start(const ServerParams& server_params,
             const GlobalTrainParams& global_train_params,
             const ModelStoreParams& model_store_params) {
    controller = new Controller(global_train_params, model_store_params);
    servicer = new ControllerServicer(server_params, controller);
    servicer->StartService();
  }

  void Shutdown() {
    LOG(INFO) << "Wrapping up resources and shutting down...";
    servicer->StopService();
    servicer->WaitService();
  }

  bool ShutdownRequestReceived() { return servicer->ShutdownRequestReceived(); }

  void Wait() { servicer->WaitService(); }

 private:
  Controller* controller;
  ControllerServicer* servicer;
};

}  // namespace metisfl::controller

PYBIND11_MODULE(controller, m) {
  m.doc() = "Federation controller python soft wrapper.";

  py::class_<metisfl::controller::ControllerWrapper>(m, "ControllerWrapper")
      .def(py::init<>())
      .def("start", &metisfl::controller::ControllerWrapper::StartWrapper,
           "Initializes and starts the controller.")
      .def("shutdown", &metisfl::controller::ControllerWrapper::Shutdown,
           "Shuts down the controller.")
      .def("shutdown_request_received",
           &metisfl::controller::ControllerWrapper::ShutdownRequestReceived,
           "Check if controller has already received a shutdown request.")
      .def("wait", &metisfl::controller::ControllerWrapper::Wait,
           "Wait for controller main thread.");
}
