
#include "metisfl/controller/aggregation/federated_stride.h"

namespace metisfl::controller {

Model FederatedStride::Aggregate(
    std::vector<std::vector<std::pair<const Model *, double>>> &pairs) {
  for (auto pair : pairs) {
    const Model *latest_model = pair.front().first;
    double contrib_value = pair.front().second;

    if (num_contributors == 0) {
      InitializeModel(latest_model, contrib_value);
    } else {
      Model dummy_model;
      double dummy_value = 0;

      score_z += contrib_value;

      UpdateScaledModel(&dummy_model, latest_model, dummy_value, contrib_value);

      UpdateCommunityModel();

      num_contributors++;
    }
  }

  return model;
}

void FederatedStride::Reset() {
  score_z = 0;
  num_contributors = 0;
  model = Model();
  wc_scaled_model = Model();
}

}  // namespace metisfl::controller
