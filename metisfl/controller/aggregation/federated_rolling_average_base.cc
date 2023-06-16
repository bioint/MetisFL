
#include "metisfl/controller/aggregation/federated_rolling_average_base.h"

namespace projectmetis::controller {
namespace {

using ::proto::DeserializeTensor;
using ::proto::SerializeTensor;

// Define scaling operations on Tensors 
enum TensorOperation {
  MULTIPLY,
  DIVIDE,
  SUBTRACTION,
  ADDITION
};

template<typename T>
std::string MergeTensors(const TensorSpec &tensor_spec_left,
                       const TensorSpec &tensor_spec_right,
                       double scaling_factor_right,
                       TensorOperation op) {

  /**
   * The function first deserializes both tensors based on the provided data type. Then it
   * scales the right-hand-side tensor using its given scaling factor and then adds the
   * scaled right-hand-side tensor to the left-hand-side tensor. Finally, it serialized
   * the aggregated tensor and returns its string representation.
   */
  auto t1_l = DeserializeTensor<T>(tensor_spec_left);
  auto t2_r = DeserializeTensor<T>(tensor_spec_right);

  // Scale the right tensor by its scaling factor.
  // Careful here: if the data type is uint or int then there are no precision
  // bits and therefore the number will be rounded to the smallest integer.
  // For instance, if 0.5 * 3 then the result is 1, int(0.5*3) = int(1.5) = 1
  transform(t2_r.begin(), t2_r.end(), t2_r.begin(),
            std::bind(std::multiplies<double>(), std::placeholders::_1, scaling_factor_right));

  // Add (plus) or subtract (minus) the scaled right tensor to the left tensor. Addition occurs based on data type T.
  if (op == TensorOperation::SUBTRACTION) {
    transform(t1_l.begin(), t1_l.end(), t2_r.begin(), t1_l.begin(), std::minus<T>());
  } else if (op == TensorOperation::ADDITION) {
    transform(t1_l.begin(), t1_l.end(), t2_r.begin(), t1_l.begin(), std::plus<T>());
  }
  
  // Serialize aggregated result.
  auto serialized_tensor = SerializeTensor<T>(t1_l);
  // Convert serialization to string.
  std::string serialized_tensor_str(serialized_tensor.begin(), serialized_tensor.end());
  return serialized_tensor_str;

}

std::string MergeTensors(const TensorSpec &tensor_spec_left,
                       const TensorSpec &tensor_spec_right,
                       double scaling_factor_right,
                       TensorOperation op) {

  /**
   * This is basically a wrapper over the MergeTensors function. It calls the MergeTensors
   * function by first casting it to the given data type. Then returns the aggregated tensor.
   */
  auto num_values_left = tensor_spec_left.length();
  auto num_values_right = tensor_spec_right.length();

  auto data_type_left = tensor_spec_left.type().type();
  auto data_type_right = tensor_spec_right.type().type();

  if (num_values_left != num_values_right) throw std::runtime_error("Left and right tensors have different sizes");
  if (data_type_left != data_type_right) throw std::runtime_error("Left and right tensors have different data types");

  std::string aggregated_result;
  if (data_type_left == DType_Type_UINT8) {
    aggregated_result = MergeTensors<unsigned char>(tensor_spec_left, tensor_spec_right, scaling_factor_right, op);
  } else if (data_type_left == DType_Type_UINT16) {
    aggregated_result = MergeTensors<unsigned short>(tensor_spec_left, tensor_spec_right, scaling_factor_right, op);
  } else if (data_type_left == DType_Type_UINT32) {
    aggregated_result = MergeTensors<unsigned int>(tensor_spec_left, tensor_spec_right, scaling_factor_right, op);
  } else if (data_type_left == DType_Type_UINT64) {
    aggregated_result = MergeTensors<unsigned long>(tensor_spec_left, tensor_spec_right, scaling_factor_right, op);
  } else if (data_type_left == DType_Type_INT8) {
    aggregated_result = MergeTensors<signed char>(tensor_spec_left, tensor_spec_right, scaling_factor_right, op);
  } else if (data_type_left == DType_Type_INT16) {
    aggregated_result = MergeTensors<signed short>(tensor_spec_left, tensor_spec_right, scaling_factor_right, op);
  } else if (data_type_left == DType_Type_INT32) {
    aggregated_result = MergeTensors<signed int>(tensor_spec_left, tensor_spec_right, scaling_factor_right, op);
  } else if (data_type_left == DType_Type_INT64) {
    aggregated_result = MergeTensors<signed long>(tensor_spec_left, tensor_spec_right, scaling_factor_right, op);
  } else if (data_type_left == DType_Type_FLOAT32) {
    aggregated_result = MergeTensors<float>(tensor_spec_left, tensor_spec_right, scaling_factor_right, op);
  } else if (data_type_left == DType_Type_FLOAT64) {
    aggregated_result = MergeTensors<double>(tensor_spec_left, tensor_spec_right, scaling_factor_right, op);
  } else {
    throw std::runtime_error("Unsupported tensor data type.");
  }

  return aggregated_result;

}

template<typename T>
std::string ScaleTensor(const TensorSpec &tensor_spec,
                       double scaling_factor, TensorOperation op) {

  /**
   * The function first deserializes a tensors based on the provided data type. Then it
   * scales using its given scaling factor.
   */
  auto ts = DeserializeTensor<T>(tensor_spec);
  
  // Scale the tensor by its scaling factor.
  // Careful here: if the data type is uint or int then there are no precision
  // bits and therefore the number will be rounded to the smallest integer.
  // For instance, if 0.5 * 3 then the result is 1, int(0.5*3) = int(1.5) = 1
  if (op == TensorOperation::DIVIDE) {
    transform(ts.begin(), ts.end(), ts.begin(),
            std::bind(std::divides<double>(), std::placeholders::_1, scaling_factor));
  } else if (op == TensorOperation::MULTIPLY) {
    transform(ts.begin(), ts.end(), ts.begin(),
            std::bind(std::multiplies<double>(), std::placeholders::_1, scaling_factor));
  }

  // Serialize aggregated result.
  auto serialized_tensor = SerializeTensor<T>(ts);
  // Convert serialization to string.
  std::string serialized_tensor_str(serialized_tensor.begin(), serialized_tensor.end());
  return serialized_tensor_str;

}

std::string ScaleTensors(const TensorSpec &tensor_spec,
                       double scaling_factor, TensorOperation op) {

  /**
   * This is basically a wrapper over the SubtractTensors function. It calls the SubtractTensors
   * function by first casting it to the given data type. Then returns the aggregated tensor.
   */
  auto num_values = tensor_spec.length();
  auto data_type = tensor_spec.type().type();

  if (num_values <= 0) throw std::runtime_error("tensor has no values.");

  std::string aggregated_result;
  if (data_type == DType_Type_UINT8) {
    aggregated_result = ScaleTensor<unsigned char>(tensor_spec, scaling_factor, op);
  } else if (data_type == DType_Type_UINT16) {
    aggregated_result = ScaleTensor<unsigned short>(tensor_spec, scaling_factor, op);
  } else if (data_type == DType_Type_UINT32) {
    aggregated_result = ScaleTensor<unsigned int>(tensor_spec, scaling_factor, op);
  } else if (data_type == DType_Type_UINT64) {
    aggregated_result = ScaleTensor<unsigned long>(tensor_spec, scaling_factor, op);
  } else if (data_type == DType_Type_INT8) {
    aggregated_result = ScaleTensor<signed char>(tensor_spec, scaling_factor, op);
  } else if (data_type == DType_Type_INT16) {
    aggregated_result = ScaleTensor<signed short>(tensor_spec, scaling_factor, op);
  } else if (data_type == DType_Type_INT32) {
    aggregated_result = ScaleTensor<signed int>(tensor_spec, scaling_factor, op);
  } else if (data_type == DType_Type_INT64) {
    aggregated_result = ScaleTensor<signed long>(tensor_spec, scaling_factor, op);
  } else if (data_type == DType_Type_FLOAT32) {
    aggregated_result = ScaleTensor<float>(tensor_spec, scaling_factor, op);
  } else if (data_type == DType_Type_FLOAT64) {
    aggregated_result = ScaleTensor<double>(tensor_spec, scaling_factor, op);
  } else {
    throw std::runtime_error("Unsupported tensor data type.");
  }

  return aggregated_result;

}

}

void FederatedRollingAverageBase::InitializeModel(const Model *init_model, double init_contrib_value) {

  /*
    Each Model defined in the model.proto has multiple variables. Each variable 
    represents a PlaintextTensor which encapsulates a TensorSpec type. The TensorSpec
    represents a serialized object of bytes (and other members).

    * Model - has -> Multiple Variables
    * Variable - has -> PlaintextTensor - has -> TensorSpec
    * TensorSpec - has -> serialized stream of byte values.

    This function iterates through all the Model_Variables of a Model `init_model`. This
    function is used to initialize the values of the initial variables. 
  */

  // Initialize the 'scaled' and 'community model'
  wc_scaled_model = *init_model;
 
  community_score_z = init_contrib_value;

  // Iterate Model_Variables of init_model and scale with community_score_z
  for (auto index = 0; index < init_model->variables_size(); index++) {

   const auto &init_variable = init_model->variables(index);
   auto scaled_variable = wc_scaled_model.mutable_variables(index);
   if (init_variable.has_plaintext_tensor()) {

    auto aggregated_result = ScaleTensors(init_variable.plaintext_tensor().tensor_spec(),
                     init_contrib_value, TensorOperation::MULTIPLY);

    *(scaled_variable->mutable_plaintext_tensor()->mutable_tensor_spec()->mutable_value()) = aggregated_result;

    } // End If

    //TODO: Place CipherText logic here. 

  } // End For

  *community_model.mutable_model() = wc_scaled_model;
  community_model.set_num_contributors(1);
}

void FederatedRollingAverageBase::UpdateScaledModel(const Model *existing_model, const Model *new_model,
                                                    double existing_contrib_value, double new_contrib_value) {

  // Iterate every Model_Variable of Model
  for (int index = 0; index < wc_scaled_model.variables_size(); index++) {

    auto scaled_variable = wc_scaled_model.mutable_variables(index);

    if (scaled_variable->has_plaintext_tensor()) {

      std::string aggregated_result;
    
      const auto &scaled_mdl_tensorSpec = scaled_variable->plaintext_tensor().tensor_spec();
      
      /* If existing_model is present then subtract Existing Model from Scaled Model. 
        (1) Scale the Tensor of Existing Model using existing_contrib_value.
        (2) Subtract the existing tensor from the Scaled Model Variable.
      */
      if (existing_model->variables_size() > 0) {
        const auto &existing_mdl_tensorSpec = existing_model->variables(index).plaintext_tensor().tensor_spec();
        aggregated_result = MergeTensors(scaled_mdl_tensorSpec, existing_mdl_tensorSpec, 
                                        existing_contrib_value,
                                        TensorOperation::SUBTRACTION);
      }

      // (3) assign the updated scaled Tensor Value to Scaled Model TensorSpec
      *(scaled_variable->mutable_plaintext_tensor()->mutable_tensor_spec()->mutable_value()) = aggregated_result;

      // (4) Update the Scaled Model with the values of the New Model
      aggregated_result.clear();
      auto &new_mdl_tensorSpec = new_model->variables(index).plaintext_tensor().tensor_spec();
      aggregated_result = MergeTensors(scaled_mdl_tensorSpec, new_mdl_tensorSpec, new_contrib_value, TensorOperation::ADDITION);

      *(scaled_variable->mutable_plaintext_tensor()->mutable_tensor_spec()->mutable_value()) = aggregated_result;
    
    } // End If

    //TODO: Place CipherText logic here. 


  } // End For

}

void FederatedRollingAverageBase::UpdateCommunityModel() {

  /*
    This function iterates through all the Model_Variables of a Model `wc_scaled_model`. 
  */

  // (1) Reset the previous community model. 
  community_model.clear_model();

  // (2) Using the `wc_scaled_model` we iterate through all the Model_Variables
  for (const auto &scaled_mdl_variable: wc_scaled_model.variables()) {

    auto cm_variable = community_model.mutable_model()->add_variables();

    // (2.a) Initialize the scaled_mdl_variable as a cm_variable.
    *cm_variable = scaled_mdl_variable;

    if (scaled_mdl_variable.has_plaintext_tensor()) {

     std::string scaled_result;

     /* (3) The Model_Variables of TensorSpec are de-scaled 
        (4) The new updated scaled values are serialize back and saved in the Model_Variable tensor.
     */
     scaled_result = ScaleTensors(scaled_mdl_variable.plaintext_tensor().tensor_spec(), community_score_z, TensorOperation::DIVIDE);
     *(cm_variable->mutable_plaintext_tensor()->mutable_tensor_spec()->mutable_value()) = scaled_result;

    } // End If

    //TODO: Place CipherText logic here. 

  } // End For
}

}

