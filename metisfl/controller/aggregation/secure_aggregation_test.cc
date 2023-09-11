
#include "metisfl/controller/aggregation/secure_aggregation.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <filesystem>

#include "metisfl/controller/common/macros.h"
#include "metisfl/controller/common/proto_matchers.h"
#include "metisfl/controller/common/proto_tensor_serde.h"
#include "metisfl/encryption/palisade/ckks_scheme.h"
#include "metisfl/proto/model.pb.h"

using metisfl::proto::ParseTextOrDie;
using ::testing::proto::EqualsProto;

namespace metisfl::controller {
namespace {

const char kModel_template_with_10elements[] = R"pb(
  tensor {
    length: 10
    encrypted: true
    dimensions: 10
    type { type: FLOAT32 byte_order: LITTLE_ENDIAN_ORDER fortran_order: False }
    value: ""
  }
)pb";

class SecAggTest : public ::testing::Test {};

TEST_F(SecAggTest, SecureAggregationCKKSwithFiles) /* NOLINT */ {
  // Step 1: Define the CKKS scheme parameters.
  uint32_t ckks_scheme_batch_size = 4096;
  uint32_t ckks_scheme_scaling_factor_bits = 52;
  auto ckks_ = CKKS(ckks_scheme_batch_size, ckks_scheme_scaling_factor_bits);
  auto tmp_dir = std::filesystem::temp_directory_path();
  CryptoParamsFiles crypto_params_files{tmp_dir / "cryptocontext.txt",
                                        tmp_dir / "key-public.txt",
                                        tmp_dir / "key-private.txt"};
  ckks_.GenCryptoParamsFiles(crypto_params_files);
  ckks_.LoadCryptoParamsFromFiles(crypto_params_files);

  // Step 2: Define model's variable values and encrypt them.
  std::vector<double> model_values{1, 2, 2, 4, 4, 6, 6, 8, 8, 10};
  auto model_values_encrypted = ckks_.Encrypt(model_values);

  // Step 3: Parse proto variables and create proto model instances.
  auto model1 = ParseTextOrDie<Model>(kModel_template_with_10elements);
  auto model2 = ParseTextOrDie<Model>(kModel_template_with_10elements);
  *model1.mutable_tensors(0)->mutable_value() = model_values_encrypted;
  *model2.mutable_tensors(0)->mutable_value() = model_values_encrypted;

  // Step 4: Create a pair from the two models.
  std::vector seq1({std::make_pair<const Model *, double>(&model1, 0.5)});
  std::vector seq2({std::make_pair<const Model *, double>(&model2, 0.5)});
  std::vector to_aggregate({seq1, seq2});

  // Step 6: Call SecAgg, perform the aggregation and validate.
  auto sec_agg = SecAgg();
  sec_agg.InitScheme(ckks_scheme_batch_size, ckks_scheme_scaling_factor_bits,
                     crypto_params_files.crypto_context_file);
  auto federated_model = sec_agg.Aggregate(to_aggregate);
  auto aggregated_ciphertext = federated_model.model().tensors().at(0).value();
  auto aggregated_dec =
      ckks_.Decrypt(aggregated_ciphertext, model_values.size());

  // To validate whether the returned aggregated value is correct
  // we compare the aggregated value with the original vector
  // element-by-element. Note that the comparison is based on absolute values
  // because the encryption, weighted aggregation and then the decryption
  // operations might slightly modify the decimal points of the original double
  // values.
  bool equal_vectors = 1;
  if (model_values.size() != aggregated_dec.size()) {
    LOG(INFO) << "Different sizes: " << model_values.size() << " "
               << aggregated_dec.size();
    equal_vectors = 0;
  } else {
    for (size_t i = 0; i < model_values.size(); i++) {
      double diff = aggregated_dec[i] - model_values[i];
      // Given that the encryption library (PALISADE) may introduce some
      // numerical error (e.g., 1e-11, 2e-12) during encryption, private
      // weighted aggregation and decryption, we first find the difference of
      // the decrypted value from its true value, then we truncate the
      // difference, which will return either -0 or 0, and finally we get the
      // absolute value of -0 / 0 which is equal to 0. This approach should work
      // only for this case, because we are working with integers.
      diff = abs(trunc(diff));
      if (diff != 0) {
        LOG(INFO)
            << "Numbers after truncation and absolute value are not equal."
            << " Their difference is: " << diff;
        equal_vectors = 0;
        break;
      }
    }
  }

  EXPECT_TRUE(equal_vectors);
}

}  // namespace
}  // namespace metisfl::controller
