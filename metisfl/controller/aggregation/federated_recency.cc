
#include "metisfl/controller/aggregation/federated_recency.h"

#include <glog/logging.h>

namespace metisfl::controller {

template <typename T>
Model FederatedRecency<T>::Aggregate(
    std::vector<std::vector<std::pair<Model *, double>>> &pairs) {
  std::vector<std::pair<const Model *, double>> model_pair = pairs.front();
  if (model_pair.size() > this.RequiredLearnerLineageLength()) {
    PLOG(ERROR) << "More models have been given: " << model_pair.size()
                << " than required: " << RequiredLearnerLineageLength();
    return {};
  }
  std::pair<const Model *, double> new_model_pair = model_pair.back();
  const Model *new_model = new_model_pair.first;
  double new_contrib_value = new_model_pair.second;

  if (this.num_contributors == 0) {
    PLOG(INFO) << "Initializing Community Model.";
    this.InitializeModel(new_model, new_contrib_value);
  } else {
    auto number_of_models = model_pair.size();

    if (number_of_models == 1) {
      this.score_z += new_contrib_value;
      Model dummy_old_model;
      double dummy_existing_old_value = 0;

      this.UpdateScaledModel(&dummy_old_model, new_model,
                             dummy_existing_old_value, new_contrib_value);

      this.UpdateCommunityModel();

      this.num_contributors++;
    }
    if (number_of_models == 2) {
      std::pair<const Model *, double> existing_model_pair = model_pair.front();
      const Model *existing_model = existing_model_pair.first;
      double existing_contrib_value = existing_model_pair.second;

      this.score_z = this.score_z - existing_contrib_value + new_contrib_value;

      this.UpdateScaledModel(existing_model, new_model, existing_contrib_value,
                             new_contrib_value);

      this.UpdateCommunityModel();

      // FIXME: should we increment num_contributors here?
    }
  }

  return this.model;
}

}  // namespace metisfl::controller
