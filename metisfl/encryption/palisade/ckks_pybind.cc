#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

#include "metisfl/encryption/palisade/ckks_scheme.h"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)
namespace py = pybind11;

class CKKSWrapper {
  CKKS* ckks_;

 public:
  static void PyGenCryptoParamsFiles(uint32_t batch_size,
                                     uint32_t scaling_factor_bits,
                                     std::string crypto_context_file,
                                     std::string public_key_file,
                                     std::string private_key_file) {
    GenCryptoParamsFiles(batch_size, scaling_factor_bits, crypto_context_file,
                         public_key_file, private_key_file);
  }

  CKKSWrapper(uint32_t batch_size, uint32_t scaling_factor_bits,
              std::string crypto_context_file) {
    ckks_ = new CKKS(batch_size, scaling_factor_bits, crypto_context_file);
  }

  CKKSWrapper(uint32_t batch_size, uint32_t scaling_factor_bits,
              std::string crypto_context_file, std::string public_key_file,
              std::string private_key_file) {
    ckks_ = new CKKS(batch_size, scaling_factor_bits, crypto_context_file,
                     public_key_file, private_key_file);
  }

  py::bytes PyAggregate(py::list learners_data, py::list scaling_factors) {
    if (learners_data.size() != scaling_factors.size()) {
      LOG(FATAL)
          << "Error: learner_data and scaling_factors size need to match";
    }
    auto learners_data_vec = learners_data.cast<std::vector<std::string>>();
    auto scaling_factors_vec = scaling_factors.cast<std::vector<double>>();

    auto weighted_avg_str =
        ckks_->Aggregate(learners_data_vec, scaling_factors_vec);
    py::bytes py_bytes_avg(weighted_avg_str);

    return py_bytes_avg;
  }

  py::array_t<double> PyDecrypt(std::string data,
                                unsigned long int num_elements) {
    auto data_decrypted = ckks_->Decrypt(data, num_elements);
    auto py_array_decrypted =
        py::array_t<double>(py::cast(std::move(data_decrypted)));
    return py_array_decrypted;
  }

  py::bytes PyEncrypt(py::array_t<double> data_array) {
    auto data_vec = std::vector<double>(data_array.data(),
                                        data_array.data() + data_array.size());
    auto data_encrypted_str = ckks_->Encrypt(data_vec);
    py::bytes py_bytes(data_encrypted_str);
    return py_bytes;
  }
};

PYBIND11_MODULE(fhe, m) {
  m.doc() = "CKKS soft python wrapper.";
  py::class_<CKKSWrapper>(m, "CKKS")
      .def(py::init<int, int, std::string>(), py::arg("batch_size"),
           py::arg("scaling_factor_bits"), py::arg("crypto_context_file"))
      .def(py::init<int, int, std::string, std::string, std::string>(),
           py::arg("batch_size"), py::arg("scaling_factor_bits"),
           py::arg("crypto_context_file"), py::arg("public_key_file"),
           py::arg("private_key_file"))
      .def_static("gen_crypto_params_files",
                  &CKKSWrapper::PyGenCryptoParamsFiles)
      .def("aggregate", &CKKSWrapper::PyAggregate)
      .def("encrypt", &CKKSWrapper::PyEncrypt)
      .def("decrypt", &CKKSWrapper::PyDecrypt);

  m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------
        .. currentmodule:: cmake_example
        .. autosummary::
           :toctree: _generate
    )pbdoc";

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif
}
