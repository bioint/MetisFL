
#ifndef PROJECTMETIS_RC_PROJECTMETIS_CONTROLLER_AGGREGATIONS_FED_ROLL_H_
#define PROJECTMETIS_RC_PROJECTMETIS_CONTROLLER_AGGREGATIONS_FED_ROLL_H_

#include "metisfl/controller/common/proto_tensor_serde.h"
#include "metisfl/proto/model.pb.h"

using ::proto::PrintSerializedTensor;

namespace projectmetis::controller {

class FederatedRollingAverageBase {
 protected:
  double community_score_z = 0; // This keeps track of the z-score of the models.
  FederatedModel community_model; // This keeps track of the cumulative community model.
  Model wc_scaled_model; // This is the scaled (weighted) model.

  void InitializeModel(const Model *init_model, double init_contrib_value);

  void UpdateScaledModel(const Model *existing_model, const Model *new_model,
                         double existing_contrib_value, double new_contrib_value);

  void UpdateCommunityModel();

};
}


#endif //PROJECTMETIS_RC_PROJECTMETIS_CONTROLLER_AGGREGATIONS_FED_ROLL_H_
