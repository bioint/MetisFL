
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include "ckks_scheme.h"

void generateRandomData(std::vector<double> &learner_Data, int rows, bool ceil_numbers = false) {

  double lower_bound = 0;
  double upper_bound = 100;

  std::uniform_real_distribution<double> unif(lower_bound, upper_bound);
  std::default_random_engine re;

  for (int i = 0; i < rows; i++) {

    if (ceil_numbers) {
      learner_Data.push_back(ceil(unif(re)));
    } else {
      learner_Data.push_back(unif(re));
    }

  }

}


int main() {

  std::filesystem::path cwd = std::filesystem::current_path();
  std::cout << ":: WORKING PATH ::" << std::endl;
  std::cout << cwd << std::endl;
  // We use the current working directory, when running demo with bazel,
  // since the cryptoparams are part of the demo target as data dependency,
  // and therefore they will be copied during runtime to the path.
  std::string cryptodir = cwd / "metisfl/resources/fheparams/cryptoparams";
  std::cout << cryptodir << std::endl;
  uint32_t batchsize = 4096;
  uint32_t scalingfactorbits = 52;

  CKKS ckks(batchsize, scalingfactorbits);
  ckks.Print();

  // Whenever we change the `batchsize` and the `scalingfactorbits`
  // params we always need to invoke the GenCryptoContextAndKeys() function.
  ckks.GenCryptoContextAndKeys(cryptodir);
  auto crypto_params_files = ckks.GetCryptoParamsFiles();
  PLOG(INFO) << crypto_params_files.crypto_context_file;
  ckks.LoadCryptoContextFromFile(crypto_params_files.crypto_context_file);
  ckks.LoadPublicKeyFromFile(crypto_params_files.public_key_file);
  ckks.LoadPrivateKeyFromFile(crypto_params_files.private_key_file);

  //generating random data for testing.
  vector<double> learner_Data;
  generateRandomData(learner_Data, 10, true);

  std::cout << "Learner Data: " << std::endl;
  std::cout << learner_Data << std::endl << std::endl << std::endl;

  std::cout << "Encrypting" << std::endl;

  std::string enc_result = ckks.Encrypt(learner_Data);
//  std::ofstream enc_result_fout("/tmp/metis/encrypted_random_numbers.out");
//  enc_result_fout << enc_result;
//  enc_result_fout.close();

  vector <std::string> learners_Data;

  learners_Data.push_back(enc_result);
  learners_Data.push_back(enc_result);
  learners_Data.push_back(enc_result);

  vector<float> scalingFactors;

  scalingFactors.push_back(0.5);
  scalingFactors.push_back(0.3);
  scalingFactors.push_back(0.5);

  std::cout << "Computing 0.5*L + 0.3*L + 0.5*L" << std::endl;

  std::string pwa_result =
      ckks.ComputeWeightedAverage(learners_Data, scalingFactors);

  unsigned long int data_dimensions = learner_Data.size();

  std::cout << "Decrypting" << std::endl;

  vector<double> pwa_res_pt = ckks.Decrypt(pwa_result, data_dimensions);

  std::cout << "Result:" << std::endl;

  std::cout << pwa_res_pt << std::endl << std::endl << std::endl << std::endl;

}
