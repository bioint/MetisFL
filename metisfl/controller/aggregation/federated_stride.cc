
#include "metisfl/controller/aggregation/federated_stride.h"

namespace metisfl::controller {

FederatedModel FederatedStride::Aggregate(AggregationPairs &pairs) {
  /*
      Once for every batch cycle we need to initialize the models once.
      Once a batch-cycle is complete the community_model needs to reset back to
     0

      Input Format: { { (learner_1, 0.1) }, { (learner_1, 0.2) } }
      Once the models are initialized we would go over all pairs
      forming the batch.
  */

  for (auto pair : pairs) {
    const Model *latest_model = pair.front().first;
    double contrib_value = pair.front().second;

    if (community_model.num_contributors() == 0) {
      InitializeModel(latest_model, contrib_value);
    } else {
      Model dummy_model;
      double dummy_value = 0;

      community_score_z += contrib_value;

      // Update the scaled model.
      UpdateScaledModel<T>(&dummy_model, latest_model, dummy_value,
                           contrib_value);

      // Update Community Model.
      UpdateCommunityModel<T>();

      community_model.set_num_contributors(community_model.num_contributors() +
                                           1);
    }
  }

  // This will return a partial community model until
  // all the batches are processed.

  return community_model;
}

void FederatedStride::Reset() {
  /*
      A. Need to reset the class variables in batching.
      B. We don't need to reset anything in async.
  */

  community_score_z = 0;
  if (community_model.has_model()) {
    community_model.clear_model();
    community_model.clear_num_contributors();
    community_model.clear_global_iteration();
  }
  wc_scaled_model.clear_tensors();
}

}  // namespace metisfl::controller
