
#ifndef METISFL_METISFL_ENCRYPTION_PALISADE_CKKS_SCHEME_H_
#define METISFL_METISFL_ENCRYPTION_PALISADE_CKKS_SCHEME_H_

#include <glog/logging.h>
#include <math.h>
#include <omp.h>

#include <fstream>
#include <iostream>
#include <sstream>

#include "ciphertext-ser.h"
#include "cryptocontext-ser.h"
#include "palisade.h"
#include "pubkeylp-ser.h"
#include "scheme/ckks/ckks-ser.h"

using namespace lbcrypto;
using namespace std::chrono;

void GenCryptoParamsFiles(uint32_t batch_size, uint32_t scaling_factor_bits,
                          std::string crypto_context_file,
                          std::string public_key_file,
                          std::string private_key_file);
class CKKS {
 public:
  ~CKKS() = default;

  CKKS(uint32_t batch_size, uint32_t scaling_factor_bits,
       std::string crypto_context_file);

  CKKS(uint32_t batch_size, uint32_t scaling_factor_bits,
       std::string crypto_context_file, std::string public_key_file,
       std::string private_key_file);

  std::string Encrypt(vector<double> data_array);

  std::string Aggregate(vector<std::string> data_array,
                        vector<double> scaling_factors);

  std::vector<double> Decrypt(std::string data,
                              unsigned long int num_elements);
  void Print();

 private:
  uint32_t batch_size;
  uint32_t scaling_factor_bits;

  CryptoContext<DCRTPoly> cc;
  LPPublicKey<DCRTPoly> pk;
  LPPrivateKey<DCRTPoly> sk;

  template <typename T>
  void DeserializeFromFile(std::string filepath, T &obj);
};

#endif  // METISFL_METISFL_ENCRYPTION_PALISADE_CKKS_SCHEME_H_
