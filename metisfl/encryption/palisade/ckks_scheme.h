
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

class CKKS : public HEScheme {

 public:
  ~CKKS() = default;
  CKKS();
  CKKS(uint32_t batch_size, uint32_t scaling_factor_bits, std::string crypto_dir);

  int GenCryptoContextAndKeys() override;
  void LoadCryptoContext() override;
  void LoadPrivateKey() override;
  void LoadPublicKey() override;
  void LoadContextAndKeys();
  std::string Encrypt(vector<double> data_array) override;
  std::string ComputeWeightedAverage(vector<std::string> data_array,
                                     vector<float> scaling_factors) override;
  vector<double> Decrypt(std::string data,
                         unsigned long int data_dimensions) override;
  void Print();

 private:
  uint32_t batch_size;
  uint32_t scaling_factor_bits;
  std::string crypto_dir;

  // The double-CRT (DCRT) ciphertext representation is
  // an extension of the Chinese Remainder Transform.
  CryptoContext<DCRTPoly> cc;
  LPPublicKey<DCRTPoly> pk;
  LPPrivateKey<DCRTPoly> sk;

};

#endif //METISFL_METISFL_ENCRYPTION_PALISADE_CKKS_SCHEME_H
