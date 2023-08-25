
#include "metisfl/controller/aggregation/federated_recency.h"

#include <glog/logging.h>

namespace metisfl::controller {

template <typename T>
FederatedModel FederatedRecency<T>::Aggregate(std::vector<std::vector<std::pair<Model *, double>>> &pairs) {
  std::vector<std::pair<const Model *, double>> model_pair = pairs.front();
  if (model_pair.size() > this.RequiredLearnerLineageLength()) {
    PLOG(ERROR) << "More models have been given: " << model_pair.size()
                << " than required: " << RequiredLearnerLineageLength();
    return {};
  }
  std::pair<const Model *, double> new_model_pair = model_pair.back();
  const Model *new_model = new_model_pair.first;
  double new_contrib_value = new_model_pair.second;

  if (this.community_model.num_contributors() == 0) {
    PLOG(INFO) << "Initializing Community Model.";
    this.InitializeModel(new_model, new_contrib_value);

  } else {
    auto number_of_models = model_pair.size();

    if (number_of_models == 1) {
      PLOG(INFO) << "Case II-A triggered.";
      this.community_score_z += new_contrib_value;
      Model dummy_old_model;
      double dummy_existing_old_value = 0;

      this.UpdateScaledModel(&dummy_old_model, new_model,
                             dummy_existing_old_value, new_contrib_value);

      this.UpdateCommunityModel();

      this.community_model.set_num_contributors(
          this.community_model.num_contributors() + 1);
    }

    if (number_of_models == 2) {
      PLOG(INFO) << "Case II-B triggered.";
      std::pair<const Model *, double> existing_model_pair = model_pair.front();
      const Model *existing_model = existing_model_pair.first;
      double existing_contrib_value = existing_model_pair.second;

      this.community_score_z =
          this.community_score_z - existing_contrib_value + new_contrib_value;

      this.UpdateScaledModel(existing_model, new_model, existing_contrib_value,
                             new_contrib_value);

      this.UpdateCommunityModel();
    }
  }

  return this.community_model;
}

}  // namespace metisfl::controller
