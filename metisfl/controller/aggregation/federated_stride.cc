
#include "metisfl/controller/aggregation/federated_stride.h"

namespace metisfl::controller {

template <typename T>
FederatedModel FederatedStride<T>::Aggregate(
    std::vector<std::vector<std::pair<Model *, double>>> &pairs) {
  for (auto pair : pairs) {
    const Model *latest_model = pair.front().first;
    double contrib_value = pair.front().second;

    if (this.community_model.num_contributors() == 0) {
      this.InitializeModel(latest_model, contrib_value);
    } else {
      Model dummy_model;
      double dummy_value = 0;

      this.community_score_z += contrib_value;

      this.UpdateScaledModel(&dummy_model, latest_model, dummy_value,
                             contrib_value);

      this.UpdateCommunityModel();

      this.community_model.set_num_contributors(
          this.community_model.num_contributors() + 1);
    }
  }

  return this.community_model;
}

template <typename T>
void FederatedStride<T>::Reset() {
  this.community_score_z = 0;
  if (this.community_model.has_model()) {
    this.community_model.clear_model();
    this.community_model.clear_num_contributors();
    this.community_model.clear_global_iteration();
  }
  this.wc_scaled_model.clear_tensors();
}

}  // namespace metisfl::controller
