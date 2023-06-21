#include "palisade.h"
#include <random>
#include <string>

#include "utils/serialize-binary.h"
#include <iomanip>
#include "ciphertext-ser.h"
#include "cryptocontext-ser.h"
#include "scheme/ckks/ckks-ser.h"
#include "pubkeylp-ser.h"

#include <omp.h>

using namespace lbcrypto;
using namespace std::chrono;

class FHE_Helper {

 private:

  std::string scheme;
  uint batchSize;
  uint scaleFactorBits;
  std::string cryptodir = "resources/fheparams/cryptoparams/";

  CryptoContext <DCRTPoly> cc;
  LPPublicKey <DCRTPoly> pk;
  LPPrivateKey <DCRTPoly> sk;

 public:

  FHE_Helper();
  FHE_Helper(std::string scheme,
             uint batchSize,
             uint scaleFactorBits);
  FHE_Helper(std::string scheme,
             uint batchSize,
             uint scaleFactorBits,
             std::string cryptodir);

  int genCryptoContextAndKeys();
  void load_crypto_params();

  std::string encrypt(vector<double> data_array);
  std::string computeWeightedAverage(
      vector<std::string> learners_Data,
      vector<float> scalingFactors);
  vector<double> decrypt(std::string learner_Data,
                         unsigned long int data_dimesions);

};
