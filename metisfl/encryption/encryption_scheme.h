
#ifndef METISFL_METISFL_ENCRYPTION_ENCRYPTION_SCHEME_H_
#define METISFL_METISFL_ENCRYPTION_ENCRYPTION_SCHEME_H_

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

struct CryptoParamsFiles {
  std::string crypto_context_file;
  std::string public_key_file;
  std::string private_key_file;
  std::string eval_mult_key_file;
};

class EncryptionScheme {

 public:
  virtual ~EncryptionScheme() = default;
  EncryptionScheme(std::string name) : name(name) {};

  std::string Name() {
    return name;
  }

  virtual void GenCryptoParams(CryptoParamsFiles crypto_params_files) = 0;
  virtual CryptoParamsFiles GetCryptoParams() = 0;
  virtual void LoadCryptoParams(CryptoParamsFiles crypto_params_files) = 0;
  virtual void LoadCryptoContextFromFile(std::string crypto_context_key_file) = 0;
  virtual void LoadPublicKeyFromFile(std::string public_key_file) = 0;
  virtual void LoadPrivateKeyFromFile(std::string private_key_file) = 0;
  virtual void LoadEvalMultiKeyFromFile(std::string eval_mult_key_file) = 0;  
  virtual std::string Aggregate(std::vector<std::string> data_array,
                                std::vector<float> scaling_factors) = 0;
  virtual std::string Encrypt(std::vector<double> data_array) = 0;
  virtual std::vector<double> Decrypt(std::string data,
                                      unsigned long int data_dimensions) = 0;

 private:
  std::string name;

};

#endif //METISFL_METISFL_ENCRYPTION_ENCRYPTION_H_
