
#ifndef METISFL_METISFL_CONTROLLER_AGGREGATION_FED_ROLL_H_
#define METISFL_METISFL_CONTROLLER_AGGREGATION_FED_ROLL_H_

#include "metisfl/controller/common/proto_tensor_serde.h"
#include "metisfl/proto/model.pb.h"

using ::proto::PrintSerializedTensor;

namespace metisfl::controller {

class FederatedRollingAverageBase {
 protected:
  double community_score_z = 0;

  FederatedModel community_model;

  Model wc_scaled_model;

  void InitializeModel(const Model *init_model, double init_contrib_value);

  void UpdateScaledModel(const Model *existing_model, const Model *new_model,
                         double existing_contrib_value,
                         double new_contrib_value);

  void UpdateCommunityModel();
};
}  // namespace metisfl::controller

#endif  // METISFL_METISFL_CONTROLLER_AGGREGATION_FED_ROLL_H_
