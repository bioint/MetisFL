
#include "fhe_helper.h"
#include <fstream>
#include <string>
#include <iostream>

void generateRandomData(vector<double> &learner_Data, int rows, bool ceil_numbers = false) {

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

  std::string cryptodir = "resources/fheparams/cryptoparams/";
  std::cout << ":: WORKING PATH ::" << std::endl;
//  std::cout << std::filesystem::current_path() << std::endl;
  std::string scheme = "ckks";
  uint batchsize = 4096;
  uint scalingfactorbits = 52;

  FHE_Helper fhe_helper(scheme, batchsize, scalingfactorbits, cryptodir);
  // Generates CryptoParams for the entire session and the driver shares the
  // files with all learners and controller. Whenever we change the `batchsize`
  // and the `scalingfactorbits` params we always need to invoke the
  // genCryptoContextAndKeys() function.
//  fhe_helper.genCryptoContextAndKeys();
  fhe_helper.load_crypto_params();

  //generating random data for testing.
  vector<double> learner_Data;
  generateRandomData(learner_Data, 10, true);

  std::cout << "Learner Data: " << std::endl;
  std::cout << learner_Data << std::endl << std::endl << std::endl;

  std::cout << "Encrypting" << std::endl;

  std::string enc_result = fhe_helper.encrypt(learner_Data);
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
      fhe_helper.computeWeightedAverage(learners_Data, scalingFactors);

  unsigned long int data_dimensions = learner_Data.size();

  std::cout << "Decrypting" << std::endl;

  vector<double> pwa_res_pt = fhe_helper.decrypt(pwa_result, data_dimensions);

  std::cout << "Result:" << std::endl;

  std::cout << pwa_res_pt << std::endl << std::endl << std::endl << std::endl;

}
