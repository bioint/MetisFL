
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

class HEScheme {

 public:
  virtual ~HEScheme() = default;
  HEScheme(std::string name) : name(name) {};

  std::string Name() {
    return name;
  }

  virtual void LoadCryptoContext() = 0;
  virtual void LoadPrivateKey() = 0;
  virtual void LoadPublicKey() = 0;
  virtual int GenCryptoContextAndKeys() = 0;
  virtual std::string Encrypt(vector<double> data_array) = 0;
  virtual std::string ComputeWeightedAverage(vector<std::string> learners_Data,
                                             vector<float> scalingFactors) = 0;
  virtual vector<double> Decrypt(std::string learner_Data,
                                 unsigned long int data_dimensions) = 0;

 private:
  std::string name;

};

#endif //METISFL_METISFL_ENCRYPTION_PALISADE_HE_SCHEME_H_
