
#include <filesystem>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "metisfl/controller/aggregation/secure_aggregation.h"
#include "metisfl/controller/common/macros.h"
#include "metisfl/controller/common/proto_matchers.h"
#include "metisfl/encryption/palisade/ckks_scheme.h"

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
    tensor {
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

class SecAggTest : public ::testing::Test {};

TEST_F(SecAggTest, SecureAggregationCKKSwithFiles) /* NOLINT */ {

  // Step 1: Define the CKKS scheme parameters.
  uint32_t ckks_scheme_batch_size = 4096;
  uint32_t ckks_scheme_scaling_factor_bits = 52;
  auto ckks_ = CKKS(
    ckks_scheme_batch_size, ckks_scheme_scaling_factor_bits);
  auto tmp_dir = std::filesystem::temp_directory_path();
  CryptoParamsFiles crypto_params_files {
    tmp_dir / "cryptocontext.txt",
    tmp_dir / "key-public.txt",
    tmp_dir / "key-private.txt" };    
  ckks_.GenCryptoParamsFiles(crypto_params_files);
  ckks_.LoadCryptoParamsFromFiles(crypto_params_files);

  // Step 2: Define model's variable values and encrypt them.
  std::vector<double> model_values{1, 2, 2, 4, 4, 6, 6, 8, 8, 10};
  auto model_values_encrypted = ckks_.Encrypt(model_values);

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
  CKKSScheme ckks_scheme;
  ckks_scheme.set_batch_size(ckks_scheme_batch_size);
  ckks_scheme.set_scaling_factor_bits(ckks_scheme_scaling_factor_bits);

  HESchemeConfig he_scheme_config;
  he_scheme_config.set_as_files(true);
  he_scheme_config.set_crypto_context(crypto_params_files.crypto_context_file);

  HEScheme he_scheme;
  *he_scheme.mutable_he_scheme_config() = he_scheme_config;
  *he_scheme.mutable_ckks_scheme() = ckks_scheme;

  EncryptionConfig encryption_config;
  *encryption_config.mutable_he_scheme() = he_scheme;

  // Step 6: Call SecAgg, perform the aggregation and validate.
  auto sec_agg = SecAgg(encryption_config);
  auto federated_model = sec_agg.Aggregate(to_aggregate);
  auto aggregated_ciphertext =
    federated_model.model().variables().at(0).ciphertext_tensor().tensor().value();
  auto aggregated_dec =
    ckks_.Decrypt(aggregated_ciphertext, model_values.size());

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

TEST_F(SecAggTest, SecureAggregationCKKSNoFiles) /* NOLINT */ {

  // Step 1: Define the CKKS scheme parameters.
  uint32_t ckks_scheme_batch_size = 4096;
  uint32_t ckks_scheme_scaling_factor_bits = 52;
  auto ckks_ = CKKS(
    ckks_scheme_batch_size, ckks_scheme_scaling_factor_bits);
  auto crypto_params = ckks_.GenCryptoParams();
  ckks_.LoadCryptoParams(crypto_params);

  // Step 2: Define model's variable values and encrypt them.
  std::vector<double> model_values{1, 2, 2, 4, 4, 6, 6, 8, 8, 10};
  auto model_values_encrypted = ckks_.Encrypt(model_values);

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
  CKKSScheme ckks_scheme;
  ckks_scheme.set_batch_size(ckks_scheme_batch_size);
  ckks_scheme.set_scaling_factor_bits(ckks_scheme_scaling_factor_bits);

  HESchemeConfig he_scheme_config;
  he_scheme_config.set_as_files(false);
  he_scheme_config.set_crypto_context(crypto_params.crypto_context);

  HEScheme he_scheme;
  *he_scheme.mutable_he_scheme_config() = he_scheme_config;
  *he_scheme.mutable_ckks_scheme() = ckks_scheme;

  EncryptionConfig encryption_config;
  *encryption_config.mutable_he_scheme() = he_scheme;

  // Step 6: Call SecAgg, perform the aggregation and validate.
  auto pwa = SecAgg(encryption_config);
  auto federated_model = pwa.Aggregate(to_aggregate);
  auto aggregated_ciphertext =
    federated_model.model().variables().at(0).ciphertext_tensor().tensor().value();
  auto aggregated_dec =
    ckks_.Decrypt(aggregated_ciphertext, model_values.size());

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
