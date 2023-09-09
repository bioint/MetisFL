
#include "ckks_scheme.h"

void GenCryptoParamsFiles(uint32_t batch_size, uint32_t scaling_factor_bits,
                          std::string crypto_context_file,
                          std::string public_key_file,
                          std::string private_key_file) {
  usint multDepth = 2;
  CryptoContext<DCRTPoly> cryptoContext;
  cryptoContext = CryptoContextFactory<DCRTPoly>::genCryptoContextCKKS(
      multDepth, scaling_factor_bits, batch_size);
  cryptoContext->Enable(ENCRYPTION);
  cryptoContext->Enable(SHE);
  LPKeyPair<DCRTPoly> keyPair;
  keyPair = cryptoContext->KeyGen();

  if (!Serial::SerializeToFile(crypto_context_file, cryptoContext,
                               SerType::BINARY)) {
    LOG(FATAL) << "Error writing serialization of crypto context";
  }
  if (!Serial::SerializeToFile(public_key_file, keyPair.publicKey,
                               SerType::BINARY)) {
    LOG(FATAL) << "Error writing serialization of public key";
  }

  if (!Serial::SerializeToFile(private_key_file, keyPair.secretKey,
                               SerType::BINARY)) {
    LOG(FATAL) << "Error writing serialization of private key";
  }
}

CKKS::CKKS(uint32_t batch_size, uint32_t scaling_factor_bits,
           std::string crypto_context_file) {
  this->batch_size = batch_size;
  this->scaling_factor_bits = scaling_factor_bits;
  CKKS::DeserializeFromFile<CryptoContext<DCRTPoly>>(crypto_context_file, cc);
}

CKKS::CKKS(uint32_t batch_size, uint32_t scaling_factor_bits,
           std::string crypto_context_file, std::string public_key_file,
           std::string private_key_file) {
  this->batch_size = batch_size;
  this->scaling_factor_bits = scaling_factor_bits;
  CKKS::DeserializeFromFile<CryptoContext<DCRTPoly>>(crypto_context_file, cc);
  CKKS::DeserializeFromFile<LPPublicKey<DCRTPoly>>(public_key_file, pk);
  CKKS::DeserializeFromFile<LPPrivateKey<DCRTPoly>>(private_key_file, sk);
}

template <typename T>
void CKKS::DeserializeFromFile(std::string filepath, T &obj) {
  if (obj == nullptr) {
    if (!Serial::DeserializeFromFile(filepath, obj, SerType::BINARY)) {
      LOG(FATAL) << "Could not deserialize from file: " << filepath;
    }
  }
}

void CKKS::Print() {
  LOG(INFO) << "CKKS scheme specifications."
            << "Batch Size: " << batch_size
            << " Scaling Factor Bits: " << scaling_factor_bits;
}

std::string CKKS::Aggregate(std::vector<std::string> data_array,
                            std::vector<double> scaling_factors) {
  if (cc == nullptr) {
    LOG(FATAL) << "Crypto context is not loaded.";
  }

  if (data_array.size() != scaling_factors.size()) {
    LOG(FATAL) << "Error: learner_data and scaling_factors size need to match";
  }

  const SerType::SERBINARY st;
  vector<Ciphertext<DCRTPoly>> result_ciphertext;

  for (unsigned long int i = 0; i < data_array.size(); i++) {
    std::stringstream ss(data_array[i]);
    vector<Ciphertext<DCRTPoly>> data_ciphertext;
    Serial::Deserialize(data_ciphertext, ss, st);

    for (unsigned long int j = 0; j < data_ciphertext.size(); j++) {
      double sc = scaling_factors[i];
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
  result_ciphertext.clear();
  return ss.str();
}

std::string CKKS::Encrypt(std::vector<double> data_array) {
  if (cc == nullptr || pk == nullptr) {
    LOG(FATAL) << "Crypto context or public key is not loaded.";
  }

  unsigned long int data_size = data_array.size();
  auto ciphertext_data_size = (data_size + batch_size) / batch_size;
  if (data_size % batch_size == 0) {
    ciphertext_data_size = data_size / batch_size;
  }
  vector<Ciphertext<DCRTPoly>> ciphertext_data((int)ciphertext_data_size);

  if (data_size > (unsigned long int)batch_size) {
#pragma omp parallel for
    for (unsigned long int i = 0; i < data_size; i += batch_size) {
      unsigned long int last = std::min((long)data_size, (long)i + batch_size);
      vector<double> batch;
      batch.reserve(last - i + 1);

      for (unsigned long int j = i; j < last; j++) {
        batch.push_back(data_array[j]);
      }
      Plaintext plaintext_data = cc->MakeCKKSPackedPlaintext(batch);
      ciphertext_data[(int)(i / batch_size)] = cc->Encrypt(pk, plaintext_data);
    }
  } else {
    vector<double> batch;
    batch.reserve(data_size);

    for (unsigned long int i = 0; i < data_size; i++) {
      batch.push_back(data_array[i]);
    }
    Plaintext plaintext_data = cc->MakeCKKSPackedPlaintext(batch);
    ciphertext_data[0] = cc->Encrypt(pk, plaintext_data);
  }

  std::stringstream ss;
  const SerType::SERBINARY st;
  Serial::Serialize(ciphertext_data, ss, st);

  return ss.str();
}

vector<double> CKKS::Decrypt(std::string data, unsigned long int num_elements) {
  if (cc == nullptr) {
    LOG(FATAL) << "Crypto context is not loaded.";
  }

  if (sk == nullptr) {
    LOG(FATAL) << "Private key is not loaded.";
  }

  const SerType::SERBINARY st;
  std::stringstream ss(data);

  vector<Ciphertext<DCRTPoly>> data_ciphertext;
  Serial::Deserialize(data_ciphertext, ss, st);

  vector<double> result(num_elements);

#pragma omp parallel for
  for (unsigned long int i = 0; i < data_ciphertext.size(); i++) {
    Plaintext pt;
    cc->Decrypt(sk, data_ciphertext[i], &pt);
    int length;

    if (i == data_ciphertext.size() - 1) {
      length = num_elements - (i)*batch_size;
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
