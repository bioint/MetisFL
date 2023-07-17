
#ifndef METISFL_METISFL_ENCRYPTION_PALISADE_HE_SCHEME_H_
#define METISFL_METISFL_ENCRYPTION_PALISADE_HE_SCHEME_H_

#include <iomanip>
#include <omp.h>
#include <random>
#include <string>

#include "ciphertext-ser.h"
#include "cryptocontext-ser.h"
#include "palisade.h"
#include "pubkeylp-ser.h"
#include "utils/serialize-binary.h"
#include "scheme/ckks/ckks-ser.h"

using namespace lbcrypto;
using namespace std::chrono;

class EncryptionScheme {

 public:
  virtual ~EncryptionScheme() = default;
  EncryptionScheme(std::string name) : name(name) {};

  std::string Name() {
    return name;
  }

  virtual std::string Aggregate(std::vector<std::string> data_array,
                                std::vector<float> scaling_factors) {
    return "";
  };

  virtual std::vector<float> Aggregate(std::vector<std::vector<float>> data_array,
                                       std::vector<float> scaling_factors) {
    return {0};
  };

 private:
  std::string name;

};

#endif //METISFL_METISFL_ENCRYPTION_PALISADE_HE_SCHEME_H_
