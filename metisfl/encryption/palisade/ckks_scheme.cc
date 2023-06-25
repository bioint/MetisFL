
#include <glog/logging.h>

#include "ckks_scheme.h"

CKKS::CKKS() : HEScheme("CKKS"), batch_size(0), scaling_factor_bits(0), crypto_dir("") {}

CKKS::CKKS(uint32_t batch_size, uint32_t scaling_factor_bits, std::string crypto_dir) : HEScheme("CKKS") {
  this->batch_size = batch_size;
  this->scaling_factor_bits = scaling_factor_bits;
  this->crypto_dir = crypto_dir;
}

void CKKS::LoadCryptoParams() {

  if (!Serial::DeserializeFromFile(crypto_dir + "/cryptocontext.txt",
                                   cc,
                                   SerType::BINARY)) {
    PLOG(ERROR) << crypto_dir + "/cryptocontext.txt";
    PLOG(ERROR) << "Could not read cryptocontext!";
  }

  if (!Serial::DeserializeFromFile(crypto_dir + "/key-public.txt",
                                   pk,
                                   SerType::BINARY)) {
    PLOG(ERROR) << "Could not read public key!";
  }

  if (!Serial::DeserializeFromFile(crypto_dir + "/key-private.txt",
                                   sk,
                                   SerType::BINARY)) {
    PLOG(ERROR) << "Could not read secret key!";
  }

}

int CKKS::GenCryptoContextAndKeys() {

  usint multDepth = 2;
  CryptoContext <DCRTPoly> cryptoContext;
  cryptoContext = CryptoContextFactory<DCRTPoly>::genCryptoContextCKKS(
      multDepth, scaling_factor_bits, batch_size);
  cryptoContext->Enable(ENCRYPTION);
  cryptoContext->Enable(SHE);
  // cryptoContext->Enable(LEVELEDSHE);

  if (!Serial::SerializeToFile(crypto_dir + "/cryptocontext.txt",
                               cryptoContext, SerType::BINARY)) {
    PLOG(ERROR) << "Error writing serialization of the crypto context";
    return 0;
  }

  LPKeyPair<DCRTPoly> keyPair;
  keyPair = cryptoContext->KeyGen();

  if (!Serial::SerializeToFile(crypto_dir + "/key-public.txt",
                               keyPair.publicKey, SerType::BINARY)) {
    PLOG(ERROR) << "Error writing serialization of public key";
    return 0;
  }

  if (!Serial::SerializeToFile(crypto_dir + "/key-private.txt",
                               keyPair.secretKey, SerType::BINARY)) {
    PLOG(ERROR) << "Error writing serialization of private key";
    return 0;
  }

  // Generate the relinearization key
  cryptoContext->EvalMultKeyGen(keyPair.secretKey);

  std::ofstream emkeyfile(crypto_dir + "/" + "key-eval-mult.txt",
                          std::ios::out | std::ios::binary);
  if (emkeyfile.is_open()) {
    if (cryptoContext->SerializeEvalMultKey(emkeyfile, SerType::BINARY)
        == false) {
      PLOG(ERROR) << "Error writing serialization of the eval mult keys";
      return 0;
    }

    emkeyfile.close();

  } else {
    PLOG(ERROR) << "Error serializing eval mult keys";
    return 0;
  }

  return 1;

}

std::string CKKS::Encrypt(vector<double> data_array) {

  unsigned long int size = data_array.size();
  vector<Ciphertext<DCRTPoly>>
      ciphertext_data((int) ((size + batch_size) / batch_size));

  if (size > (unsigned long int) batch_size) {

#pragma omp parallel for
    for (unsigned long int i = 0; i < size; i += batch_size) {

      unsigned long int last = std::min((long) size, (long) i + batch_size);
      vector<double> batch;
      batch.reserve(last - i + 1);

      for (unsigned long int j = i; j < last; j++) {
        batch.push_back(data_array[j]);
      }
      Plaintext plaintext_data = cc->MakeCKKSPackedPlaintext(batch);
      ciphertext_data[(int) (i / batch_size)] =
          cc->Encrypt(pk, plaintext_data);
    }

  } else {

    vector<double> batch;
    batch.reserve(size);

    for (unsigned long int i = 0; i < size; i++) {
      batch.push_back(data_array[i]);
    }
    Plaintext plaintext_data = cc->MakeCKKSPackedPlaintext(batch);
    ciphertext_data[0] = cc->Encrypt(pk, plaintext_data);
  }

  std::stringstream s;
  const SerType::SERBINARY st;
  Serial::Serialize(ciphertext_data, s, st);

  return s.str();

}

std::string CKKS::ComputeWeightedAverage(vector<std::string> data_array,
                                         vector<float> scaling_factors) {

  if (data_array.size() != scaling_factors.size()) {
    PLOG(ERROR) << "Error: data_array and scaling_factors size mismatch";
    return "";
  }

  const SerType::SERBINARY st;
  vector<Ciphertext<DCRTPoly>> result_ciphertext;

  for (unsigned long int i = 0; i < data_array.size(); i++) {
    std::stringstream ss(data_array[i]);
    vector<Ciphertext<DCRTPoly>> data_ciphertext;
    Serial::Deserialize(data_ciphertext, ss, st);

    for (unsigned long int j = 0; j < data_ciphertext.size(); j++) {
      float sc = scaling_factors[i];
      data_ciphertext[j] = cc->EvalMult(data_ciphertext[j], sc);
    }

    if (result_ciphertext.size() == 0) {

      result_ciphertext = data_ciphertext;
    } else {
      for (unsigned long int j = 0; j < data_ciphertext.size(); j++) {
        result_ciphertext[j] =
            cc->EvalAdd(result_ciphertext[j], data_ciphertext[j]);
      }
    }

  }

  std::stringstream ss;
  Serial::Serialize(result_ciphertext, ss, st);
  return ss.str();

}

vector<double> CKKS::Decrypt(std::string data,
                             unsigned long int data_dimensions) {

  const SerType::SERBINARY st;
  std::stringstream ss(data);

  vector<Ciphertext<DCRTPoly>> data_ciphertext;
  Serial::Deserialize(data_ciphertext, ss, st);

  vector<double> result(data_dimensions);

#pragma omp parallel for
  for (unsigned long int i = 0; i < data_ciphertext.size(); i++) {

    Plaintext pt;
    cc->Decrypt(sk, data_ciphertext[i], &pt);
    int length;

    if (i == data_ciphertext.size() - 1) {

      length = data_dimensions - (i) * batch_size;
    } else {
      length = batch_size;
    }

    pt->SetLength(length);
    vector<double> layer_data = pt->GetRealPackedValue();
    int m = i * batch_size;

    for (unsigned long int j = 0; j < layer_data.size(); j++) {
      result[m++] = layer_data[j];
    }
  }

  return result;

}
