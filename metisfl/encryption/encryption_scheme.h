
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
};

struct CryptoParams {
  std::string crypto_context;
  std::string public_key;
  std::string private_key;
};

class EncryptionScheme {

 public:
  virtual ~EncryptionScheme() = default;
  EncryptionScheme(std::string name) : name(name) {};

  std::string Name() {
    return name;
  }

  // File-based API.
  virtual void GenCryptoParamsFiles(CryptoParamsFiles crypto_params_files) = 0;
  virtual CryptoParamsFiles GetCryptoParamsFiles() = 0;
  virtual void LoadCryptoParamsFromFiles(CryptoParamsFiles crypto_params_files) = 0;
  virtual void LoadCryptoContextFromFile(std::string crypto_context_file) = 0;
  virtual void LoadPublicKeyFromFile(std::string public_key_file) = 0;
  virtual void LoadPrivateKeyFromFile(std::string private_key_file) = 0;
  
  // String-based API.
  virtual CryptoParams GenCryptoParams() = 0;
  virtual CryptoParams GetCryptoParams() = 0;
  virtual void LoadCryptoParams(CryptoParams crypto_params) = 0;
  virtual void LoadCryptoContext(std::string crypto_context) = 0;
  virtual void LoadPublicKey(std::string public_key) = 0;
  virtual void LoadPrivateKey(std::string private_key) = 0;  
  
  // Encryption Scheme functionality: 
  //    Aggregate Ciphertext - Encrypt Plaintext - Decrypt Ciphertext.
  virtual std::string Aggregate(std::vector<std::string> data_array,
                                std::vector<float> scaling_factors) = 0;
  virtual std::string Encrypt(std::vector<double> data_array) = 0;
  virtual std::vector<double> Decrypt(std::string data,
                                      unsigned long int data_dimensions) = 0;

 private:
  std::string name;

};

#endif //METISFL_METISFL_ENCRYPTION_ENCRYPTION_SCHEME_H_
