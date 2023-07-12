
#include <filesystem>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <cmath>

#include "metisfl/controller/aggregation/private_weighted_average.h"
#include "metisfl/controller/common/macros.h"
#include "metisfl/controller/common/proto_matchers.h"
#include "metisfl/encryption/palisade/ckks_scheme.h"
#include "metisfl/proto/metis.pb.h"
#include "metisfl/proto/model.pb.h"



namespace metisfl::controller {
namespace {

using ::proto::ParseTextOrDie;
using ::testing::proto::EqualsProto;

const char kModel_template_with_10elements[] = R"pb(
variables {
  name: "var1"
  trainable: true
  ciphertext_tensor {
    tensor_spec {
      length: 10
      dimensions: 10
      type {
        type: FLOAT32
        byte_order: LITTLE_ENDIAN_ORDER
        fortran_order: False
      }
      value: ""
    }
  }
}
)pb";

class PWATest : public ::testing::Test {};

TEST_F(PWATest, PrivateWeightedAggregationCKKS) /* NOLINT */ {

  // Step 1: Define the CKKS scheme parameters.
  uint32_t ckks_scheme_batch_size = 4096;
  uint32_t ckks_scheme_scaling_factor_bits = 52;
  auto ckks_scheme = CKKS(
    ckks_scheme_batch_size, ckks_scheme_scaling_factor_bits);
  ckks_scheme.GenCryptoContextAndKeys(std::filesystem::temp_directory_path());
  auto crypto_params_files = ckks_scheme.GetCryptoParamsFiles();
  ckks_scheme.LoadCryptoContextFromFile(crypto_params_files.crypto_context_file);
  ckks_scheme.LoadPublicKeyFromFile(crypto_params_files.public_key_file);
  ckks_scheme.LoadPrivateKeyFromFile(crypto_params_files.private_key_file);

  // Step 2: Define model's variable values and encrypt them.
  std::vector<double> model_values{1, 2, 2, 4, 4, 6, 6, 8, 8, 10};
  auto model_values_encrypted = ckks_scheme.Encrypt(model_values);

  // Step 3: Parse proto variables and create proto model instances.
  auto model1 = ParseTextOrDie<Model>(kModel_template_with_10elements);
  auto model2 = ParseTextOrDie<Model>(kModel_template_with_10elements);
  *model1.mutable_variables(0)->mutable_ciphertext_tensor()
    ->mutable_tensor_spec()->mutable_value() = model_values_encrypted;
  *model2.mutable_variables(0)->mutable_ciphertext_tensor()
    ->mutable_tensor_spec()->mutable_value() = model_values_encrypted;

  // Step 4: Create a pair from the two models.
  std::vector seq1({std::make_pair<const Model *, double>(&model1, 0.5)});
  std::vector seq2({std::make_pair<const Model *, double>(&model2, 0.5)});
  std::vector to_aggregate({seq1, seq2});

  // Step 5: Create the CKKS proto instance.
  CKKSSchemeConfig ckks_scheme_config;
  ckks_scheme_config.set_batch_size(ckks_scheme_batch_size);
  ckks_scheme_config.set_scaling_factor_bits(ckks_scheme_scaling_factor_bits);

  HESchemeConfig he_scheme_config;
  he_scheme_config.set_enabled(true);
  he_scheme_config.set_crypto_context_file(crypto_params_files.crypto_context_file);
  *he_scheme_config.mutable_ckks_scheme_config() = ckks_scheme_config;

  // Step 6: Call PWA, perform the aggregation and validate.
  auto pwa = PWA(he_scheme_config);
  auto federated_model = pwa.Aggregate(to_aggregate);
  auto aggregated_ciphertext =
    federated_model.model().variables().at(0).ciphertext_tensor().tensor_spec().value();
  auto aggregated_dec =
    ckks_scheme.Decrypt(aggregated_ciphertext, model_values.size());

  // PLOG(INFO) << model_values;
  // PLOG(INFO) << aggregated_dec;

  // // TODO(hamzahsaleem) Why the returned aggregated vector is not equal to
  // // the original vector even after casting the two vectors to int?
  // auto equal_vectors = model_values == aggregated_dec;
  // PLOG(INFO) << "Equal as doubles: " << equal_vectors;

  // std::vector<int> model_values_as_int(model_values.begin(), model_values.end());
  // std::vector<int> aggregated_dec_as_int(aggregated_dec.begin(), aggregated_dec.end());
  // auto equal_vectors = model_values_as_int == aggregated_dec_as_int;
  // PLOG(INFO) << "Equal as ints: " << equal_vectors;


  // To validate whether the returned aggregated value is correct
  // we compare the aggregated value with the original vector element-by-element.
  // Note that the comparison is based on absolute values because the encryption,
  // weighted aggregation and then the decryption operations might slightly modify
  // the decimal points of the original double values.
  bool equal_vectors = 1;
  if (model_values.size() != aggregated_dec.size()) {
    PLOG(INFO) << "Different sizes: " << model_values.size() << " " << aggregated_dec.size();
    equal_vectors = 0;
  } else {
    for (size_t i = 0; i < model_values.size(); i++) {
      double diff = aggregated_dec[i] - model_values[i];
      // Given that the encryption library (PALISADE) may introduce some numerical 
      // error (e.g., 1e-11, 2e-12) during encryption, private weighted aggregation 
      // and decryption, we first find the difference of the decrypted value from its 
      // true value, then we truncate the difference, which will return either -0 or 0,
      // and finally we get the absolute value of -0 / 0 which is equal to 0.
      // This approach should work only for this case, because we are working with integers.
      diff = abs(trunc(diff));
      if (diff != 0) {
        PLOG(INFO) << "Numbers after truncation and absolute value are not equal."
        << " Their difference is: " << diff;
        equal_vectors = 0;
        break;
      }
    }
  }

  EXPECT_TRUE(equal_vectors);

}

} // namespace
} // namespace projectmetis::controller