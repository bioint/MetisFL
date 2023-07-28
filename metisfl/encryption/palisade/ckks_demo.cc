
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

  uint32_t batchsize = 4096;
  uint32_t scalingfactorbits = 52;

  CKKS ckks(batchsize, scalingfactorbits);
  ckks.Print();

  // Whenever we change the `batchsize` and the `scalingfactorbits`
  // params we always need to invoke the GenCryptoContextAndKeys() function.
  auto tmp_dir = std::filesystem::temp_directory_path();
  CryptoParamsFiles crypto_params_files {
    tmp_dir / "cryptocontext.txt",
    tmp_dir / "key-public.txt",
    tmp_dir / "key-private.txt" };
  ckks.GenCryptoParams(crypto_params_files);
  PLOG(INFO) << crypto_params_files.crypto_context_file;
  ckks.LoadCryptoContextFromFile(crypto_params_files.crypto_context_file);
  ckks.LoadPublicKeyFromFile(crypto_params_files.public_key_file);
  ckks.LoadPrivateKeyFromFile(crypto_params_files.private_key_file);

  //generating random data for testing.
  vector<double> learner_data;
  generateRandomData(learner_data, 10, true);

  std::cout << "Learner Data: " << std::endl;
  std::cout << learner_data << std::endl << std::endl << std::endl;

  std::cout << "Encrypting" << std::endl;

  std::string enc_result = ckks.Encrypt(learner_data);
//  std::ofstream enc_result_fout("/tmp/metis/encrypted_random_numbers.out");
//  enc_result_fout << enc_result;
//  enc_result_fout.close();

  vector <std::string> learners_data;

  learners_data.push_back(enc_result);
  learners_data.push_back(enc_result);
  learners_data.push_back(enc_result);

  vector<float> scaling_factors;

  scaling_factors.push_back(0.5);
  scaling_factors.push_back(0.3);
  scaling_factors.push_back(0.5);

  std::cout << "Computing 0.5*L + 0.3*L + 0.5*L" << std::endl;

  std::string pwa_result =
      ckks.Aggregate(learners_data, scaling_factors);

  unsigned long int data_dimensions = learner_data.size();

  std::cout << "Decrypting" << std::endl;

  vector<double> pwa_res_pt = ckks.Decrypt(pwa_result, data_dimensions);

  std::cout << "Result:" << std::endl;

  std::cout << pwa_res_pt << std::endl << std::endl << std::endl << std::endl;

}
