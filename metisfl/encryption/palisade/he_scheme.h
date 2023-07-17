
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
#include "metisfl/encryption/encryption_scheme.h"

using namespace lbcrypto;
using namespace std::chrono;

class HEScheme : EncryptionScheme {

 public:
  virtual ~HEScheme() = default;
  HEScheme(std::string name) : name(name) {};

  virtual void GenCryptoContextAndKeys(std::string crypto_dir) = 0;
  virtual void LoadCryptoContextFromFile(std::string crypto_context_key_file) = 0;
  virtual void LoadPrivateKeyFromFile(std::string private_key_file) = 0;
  virtual void LoadPublicKeyFromFile(std::string public_key_file) = 0;
  virtual std::string Encrypt(std::vector<double> data_array) = 0;
  virtual std::string Aggregate(std::vector<std::string> learners_data,
                                std::vector<float> scalingFactors) = 0;
  virtual std::vector<double> Decrypt(std::string learner_Data,
                                      unsigned long int data_dimensions) = 0;

};

#endif //METISFL_METISFL_ENCRYPTION_PALISADE_HE_SCHEME_H_