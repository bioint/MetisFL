
#include "metisfl/controller/aggregation/federated_recency.h"

#include <glog/logging.h>

#include "federated_recency.h"

namespace metisfl::controller {

Model FederatedRecency::Aggregate(
    std::vector<std::vector<std::pair<const Model *, double>>> &pairs) {
  std::vector<std::pair<const Model *, double>> model_pair = pairs.front();
  if (model_pair.size() > RequiredLearnerLineageLength()) {
    LOG(ERROR) << "More models have been given: " << model_pair.size()
               << " than required: " << RequiredLearnerLineageLength();
    return {};  // FIXME: can the caller handle this?
  }
  std::pair<const Model *, double> new_model_pair = model_pair.back();
  const Model *new_model = new_model_pair.first;
  double new_contrib_value = new_model_pair.second;

  if (num_contributors == 0) {
    LOG(INFO) << "Initializing Community Model.";
    InitializeModel(new_model, new_contrib_value);
  } else {
    auto number_of_models = model_pair.size();
    if (number_of_models == 1) {
      score_z += new_contrib_value;
      Model dummy_old_model;
      double dummy_existing_old_value = 0;

      UpdateScaledModel(&dummy_old_model, new_model, dummy_existing_old_value,
                        new_contrib_value);
      UpdateCommunityModel();
      num_contributors++;
    }
    if (number_of_models == 2) {
      std::pair<const Model *, double> existing_model_pair = model_pair.front();
      const Model *existing_model = existing_model_pair.first;
      double existing_contrib_value = existing_model_pair.second;
      score_z = score_z - existing_contrib_value + new_contrib_value;
      UpdateScaledModel(existing_model, new_model, existing_contrib_value,
                        new_contrib_value);
      UpdateCommunityModel();
    }
  }
  return model;
}

void FederatedRecency::Reset() {}
}  // namespace metisfl::controller
