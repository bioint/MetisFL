
#include "metisfl/controller/aggregation/federated_stride.h"

namespace metisfl::controller {

template <typename T>
Model FederatedStride<T>::Aggregate(
    std::vector<std::vector<std::pair<const Model *, double>>> &pairs) {
  for (auto pair : pairs) {
    const Model *latest_model = pair.front().first;
    double contrib_value = pair.front().second;

    if (this.num_contributors == 0) {
      this.InitializeModel(latest_model, contrib_value);
    } else {
      Model dummy_model;
      double dummy_value = 0;

      this.score_z += contrib_value;

      this.UpdateScaledModel(&dummy_model, latest_model, dummy_value,
                             contrib_value);

      this.UpdateCommunityModel();

      this.num_contributors++;
    }
  }

  return this.model;
}

template <typename T>
void FederatedStride<T>::Reset() {
  this.score_z = 0;
  this.num_contributors = 0;
  this.model = Model();
  this.wc_scaled_model = Model();
}

}  // namespace metisfl::controller
