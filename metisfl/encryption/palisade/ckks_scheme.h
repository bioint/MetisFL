
#ifndef METISFL_METISFL_ENCRYPTION_PALISADE_CKKS_SCHEME_H
#define METISFL_METISFL_ENCRYPTION_PALISADE_CKKS_SCHEME_H

#include <glog/logging.h>
#include <omp.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <sstream>

#include "ciphertext-ser.h"
#include "cryptocontext-ser.h"
#include "palisade.h"
#include "pubkeylp-ser.h"
#include "scheme/ckks/ckks-ser.h"
#include "he_scheme.h"

using namespace lbcrypto;
using namespace std::chrono;

struct CryptoParamsFiles {
  std::string crypto_context_file;
  std::string public_key_file;
  std::string private_key_file;
  std::string eval_mult_key_file;
};

class CKKS : public HEScheme {

 public:
  ~CKKS() = default;
  CKKS();
  CKKS(uint32_t batch_size, uint32_t scaling_factor_bits);

  void GenCryptoContextAndKeys(std::string crypto_dir) override;
  CryptoParamsFiles GetCryptoParamsFiles();
  void LoadCryptoContextFromFile(std::string crypto_context_file) override;
  void LoadPrivateKeyFromFile(std::string private_key_file) override;
  void LoadPublicKeyFromFile(std::string public_key_file) override;
  void LoadContextAndKeysFromFiles(std::string crypto_context_file,
                                         std::string public_key_file,
                                         std::string private_key_file);
  std::string Encrypt(vector<double> data_array) override;
  std::string ComputeWeightedAverage(vector<std::string> data_array,
                                     vector<float> scaling_factors) override;
  std::vector<double> Decrypt(std::string data,
                              unsigned long int data_dimensions) override;
  void Print();

 private:
  uint32_t batch_size;
  uint32_t scaling_factor_bits;
  CryptoParamsFiles crypto_params_files_;

  // The double-CRT (DCRT) ciphertext representation is
  // an extension of the Chinese Remainder Transform.
  CryptoContext<DCRTPoly> cc;
  LPPublicKey<DCRTPoly> pk;
  LPPrivateKey<DCRTPoly> sk;

  template<typename T>
  void DeserializeFromFile(std::string filepath, T &obj);

};

#endif //METISFL_METISFL_ENCRYPTION_PALISADE_CKKS_SCHEME_H
