
#ifndef METISFL_METISFL_ENCRYPTION_PALISADE_CKKS_SCHEME_H_
#define METISFL_METISFL_ENCRYPTION_PALISADE_CKKS_SCHEME_H_

#include <glog/logging.h>
#include <omp.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <sstream>

#include "ciphertext-ser.h"
#include "cryptocontext-ser.h"
#include "metisfl/encryption/encryption_scheme.h"
#include "palisade.h"
#include "pubkeylp-ser.h"
#include "scheme/ckks/ckks-ser.h"

using namespace lbcrypto;
using namespace std::chrono;

class CKKS : public EncryptionScheme {

 public:
  ~CKKS() = default;
  CKKS();
  CKKS(uint32_t batch_size, uint32_t scaling_factor_bits);

  // File-based API.
  void GenCryptoParamsFiles(CryptoParamsFiles crypto_params_files) override;  
  CryptoParamsFiles GetCryptoParamsFiles() override;
  void LoadCryptoParamsFromFiles(CryptoParamsFiles crypto_params_files) override;
  void LoadCryptoContextFromFile(std::string crypto_context_file) override;
  void LoadPublicKeyFromFile(std::string public_key_file) override;
  void LoadPrivateKeyFromFile(std::string private_key_file) override;

  // String-based API.
  CryptoParams GenCryptoParams() override;
  CryptoParams GetCryptoParams() override;
  void LoadCryptoParams(CryptoParams crypto_params) override;
  void LoadCryptoContext(std::string crypto_context) override;
  void LoadPublicKey(std::string public_key) override;
  void LoadPrivateKey(std::string private_key) override;

  // CKKS functionality.
  std::string Aggregate(vector<std::string> data_array,
                        vector<float> scaling_factors) override;
  std::string Encrypt(vector<double> data_array) override;
  std::vector<double> Decrypt(std::string data,
                              unsigned long int data_dimensions) override;
  void Print();

 private:
  uint32_t batch_size;
  uint32_t scaling_factor_bits;
  CryptoParamsFiles crypto_params_files_;
  CryptoParams crypto_params_;

  // The double-CRT (DCRT) ciphertext representation is
  // an extension of the Chinese Remainder Transform.
  CryptoContext<DCRTPoly> cc;
  LPPublicKey<DCRTPoly> pk;
  LPPrivateKey<DCRTPoly> sk;

  template<typename T>
  bool DeserializeFromFile(std::string filepath, T &obj);

  template<typename T>
  void Deserialize(std::string s, T &obj);

};

#endif //METISFL_METISFL_ENCRYPTION_PALISADE_CKKS_SCHEME_H_
