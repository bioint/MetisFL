
#ifndef METISFL_METISFL_CONTROLLER_AGGREGATION_FED_ROLL_H_
#define METISFL_METISFL_CONTROLLER_AGGREGATION_FED_ROLL_H_

#include "metisfl/controller/common/proto_tensor_serde.h"
#include "metisfl/proto/model.pb.h"

using metisfl::proto::DeserializeTensor;
using metisfl::proto::SerializeTensor;

namespace metisfl::controller {

enum TensorOperation { MULTIPLY, DIVIDE, SUBTRACTION, ADDITION };

template <typename T>
class FederatedRollingAverageBase {
 protected:
  int num_contributors = 0;
  double score_z = 0;
  Model model;
  Model wc_scaled_model;

  void InitializeModel(const Model *init_model, double init_contrib_value);

  void UpdateScaledModel(const Model *existing_model, const Model *new_model,
                         double existing_contrib_value,
                         double new_contrib_value);

  void UpdateCommunityModel();

 private:
  std::string MergeTensors(const Tensor &tensor_spec_left,
                           const Tensor &tensor_spec_right,
                           double scaling_factor_right, TensorOperation op);

  std::string ScaleTensor(const Tensor &tensor, double scaling_factor,
                          TensorOperation op);
};
}  // namespace metisfl::controller

#endif  // METISFL_METISFL_CONTROLLER_AGGREGATION_FED_ROLL_H_
