
#include "metisfl/controller/aggregation/federated_rolling_average_base.h"

namespace metisfl::controller {
namespace {

using ::proto::DeserializeTensor;
using ::proto::SerializeTensor;

// Define scaling operations on Tensors
enum TensorOperation { MULTIPLY, DIVIDE, SUBTRACTION, ADDITION };

template <typename T>
std::string MergeTensors(const Tensor &tensor_spec_left,
                         const Tensor &tensor_spec_right,
                         double scaling_factor_right, TensorOperation op) {
  /**
   * The function first deserializes both tensors based on the provided data
   * type. Then it scales the right-hand-side tensor using its given scaling
   * factor and then adds the scaled right-hand-side tensor to the
   * left-hand-side tensor. Finally, it serialized the aggregated tensor and
   * returns its string representation.
   */
  auto t1_l = DeserializeTensor<T>(tensor_spec_left);
  auto t2_r = DeserializeTensor<T>(tensor_spec_right);

  // Scale the right tensor by its scaling factor.
  // Careful here: if the data type is uint or int then there are no precision
  // bits and therefore the number will be rounded to the smallest integer.
  // For instance, if 0.5 * 3 then the result is 1, int(0.5*3) = int(1.5) = 1
  transform(t2_r.begin(), t2_r.end(), t2_r.begin(),
            std::bind(std::multiplies<double>(), std::placeholders::_1,
                      scaling_factor_right));

  // Add (plus) or subtract (minus) the scaled right tensor to the left tensor.
  // Addition occurs based on data type T.
  if (op == TensorOperation::SUBTRACTION) {
    transform(t1_l.begin(), t1_l.end(), t2_r.begin(), t1_l.begin(),
              std::minus<T>());
  } else if (op == TensorOperation::ADDITION) {
    transform(t1_l.begin(), t1_l.end(), t2_r.begin(), t1_l.begin(),
              std::plus<T>());
  }

  // Serialize aggregated result.
  auto serialized_tensor = SerializeTensor<T>(t1_l);
  // Convert serialization to string.
  std::string serialized_tensor_str(serialized_tensor.begin(),
                                    serialized_tensor.end());
  return serialized_tensor_str;
}

template <typename T>
std::string ScaleTensor(const Tensor &tensor, double scaling_factor,
                        TensorOperation op) {
  /**
   * The function first deserializes a tensors based on the provided data type.
   * Then it scales using its given scaling factor.
   */
  auto ts = DeserializeTensor<T>(tensor);

  // Scale the tensor by its scaling factor.
  // Careful here: if the data type is uint or int then there are no precision
  // bits and therefore the number will be rounded to the smallest integer.
  // For instance, if 0.5 * 3 then the result is 1, int(0.5*3) = int(1.5) = 1
  if (op == TensorOperation::DIVIDE) {
    transform(ts.begin(), ts.end(), ts.begin(),
              std::bind(std::divides<double>(), std::placeholders::_1,
                        scaling_factor));
  } else if (op == TensorOperation::MULTIPLY) {
    transform(ts.begin(), ts.end(), ts.begin(),
              std::bind(std::multiplies<double>(), std::placeholders::_1,
                        scaling_factor));
  }

  // Serialize aggregated result.
  auto serialized_tensor = SerializeTensor<T>(ts);
  // Convert serialization to string.
  std::string serialized_tensor_str(serialized_tensor.begin(),
                                    serialized_tensor.end());
  return serialized_tensor_str;
}

}  // namespace

template <typename T>
void FederatedRollingAverageBase::InitializeModel(const Model *init_model,
                                                  double init_contrib_value) {
  /*
    Each Model defined in the model.proto has multiple variables. Each variable
    represents a PlaintextTensor which encapsulates a Tensor type. The
    Tensor represents a serialized object of bytes (and other members).

    * Model - has -> Multiple Variables
    * Variable - has -> PlaintextTensor - has -> Tensor
    * Tensor - has -> serialized stream of byte values.

    This function iterates through all the Model_Variables of a Model
    `init_model`. This function is used to initialize the values of the initial
    variables.
  */

  // Initialize the 'scaled' and 'community model'
  wc_scaled_model = *init_model;

  community_score_z = init_contrib_value;

  // Iterate Model_Variables of init_model and scale with community_score_z
  for (auto index = 0; index < init_model->tensors_size(); index++) {
    const auto &init_tensor = init_model->tensors(index);
    auto scaled_tensor = wc_scaled_model.mutable_tensors(index);
    if (!scaled_tensor->encrypted()) {
      auto aggregated_result = ScaleTensor<T>(init_tensor, init_contrib_value,
                                              TensorOperation::MULTIPLY);

      *scaled_tensor->mutable_value() = aggregated_result;

    }  // End If

    // TODO(stripeli): Place CipherText logic here.

  }  // End For

  *community_model.mutable_model() = wc_scaled_model;
  community_model.set_num_contributors(1);
}

template <typename T>
void FederatedRollingAverageBase::UpdateScaledModel(
    const Model *existing_model, const Model *new_model,
    double existing_contrib_value, double new_contrib_value) {
  for (int index = 0; index < wc_scaled_model.tensors_size(); index++) {
    auto scaled_tensor = wc_scaled_model.mutable_tensors(index);

    if (!scaled_tensor->encrypted()) {
      std::string aggregated_result;

      /* If existing_model is present then subtract Existing Model from Scaled
        Model. (1) Scale the Tensor of Existing Model using
        existing_contrib_value. (2) Subtract the existing tensor from the Scaled
        Model Variable.
      */
      if (existing_model->tensors_size() > 0) {
        const auto &existing_mdl_tensor = existing_model->tensors(index);
        aggregated_result = MergeTensors<T>(*scaled_tensor, existing_mdl_tensor,
                                            existing_contrib_value,
                                            TensorOperation::SUBTRACTION);
      }

      // (3) assign the updated scaled Tensor Value to Scaled Model Tensor
      *scaled_tensor->mutable_value() = aggregated_result;

      // (4) Update the Scaled Model with the values of the New Model
      aggregated_result.clear();
      auto &new_mdl_tensor = new_model->tensors(index);
      aggregated_result =
          MergeTensors(*scaled_tensor, new_mdl_tensor, new_contrib_value,
                       TensorOperation::ADDITION);

      *(scaled_tensor->mutable_value()) = aggregated_result;

    }  // End If

    // TODO(stripeli): Place CipherText logic here.

  }  // End For
}

template <typename T>
void FederatedRollingAverageBase::UpdateCommunityModel() {
  /*
    This function iterates through all the Model_Variables of a Model
    `wc_scaled_model`.
  */

  // (1) Reset the previous community model.
  community_model.clear_model();

  // (2) Using the `wc_scaled_model` we iterate through all the Model_Variables
  for (const auto &scaled_mdl_variable : wc_scaled_model.tensors()) {
    auto cm_variable = community_model.mutable_model()->add_tensors();

    // (2.a) Initialize the scaled_mdl_variable as a cm_variable.
    *cm_variable = scaled_mdl_variable;

    if (!scaled_mdl_variable.encrypted()) {
      std::string scaled_result;

      /* (3) The Model_Variables of Tensor are de-scaled
         (4) The new updated scaled values are serialize back and saved in the
         Model_Variable tensor.
      */
      scaled_result = ScaleTensor<T>(scaled_mdl_variable, community_score_z,
                                     TensorOperation::DIVIDE);
      *(cm_variable->mutable_value()) = scaled_result;

    }  // End If

    // TODO(stripeli): Place CipherText logic here.

  }  // End For
}

}  // namespace metisfl::controller
