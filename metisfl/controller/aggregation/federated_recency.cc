
#include <glog/logging.h>

#include "metisfl/controller/aggregation/federated_recency.h"

namespace metisfl::controller {

FederatedModel FederatedRecency::Aggregate(std::vector<std::vector<std::pair<const Model *, double>>> &pairs) {
  /*
      Input argument pairs can be
      (1) At least One Entry: Meaning {new} model pair.
      (2) At most Two Entries: Meaning {old, new} model pair.
  */
  std::vector<std::pair<const Model *, double>> model_pair = pairs.front();
  if (model_pair.size() > RequiredLearnerLineageLength()) {
    PLOG(ERROR) << "More models have been given: "
                << model_pair.size()
                << " than required: "
                << RequiredLearnerLineageLength();
    return {};
  }
  // We always consider the most recent model pair entry to be at the end of the vector.
  std::pair<const Model *, double> new_model_pair = model_pair.back();
  const Model *new_model = new_model_pair.first;
  double new_contrib_value = new_model_pair.second;

  if (community_model.num_contributors() == 0) {
    /*
     * Case: Initialize the community model.
     * This function would be invoked by the first
     * learner to submit a model.
     * */
    PLOG(INFO) << "Initializing Community Model.";
    InitializeModel(new_model, new_contrib_value);

  } else {

    auto number_of_models = model_pair.size();

    if (number_of_models == 1) {
      /*
       * Case: Community model is already initialized.
       * The learners are submitting their models
       * but only have one model (latest) with them.
       * */
      PLOG(INFO) << "Case II-A triggered.";
      community_score_z += new_contrib_value;

      /* We need dummy models to keep the
       function interface valid. The dummy
       model is just a fake value for old
       model which does not exist at this point.
      */
      Model dummy_old_model;
      double dummy_existing_old_value = 0;

      // update the scaled model.
      UpdateScaledModel(&dummy_old_model,
                        new_model,
                        dummy_existing_old_value,
                        new_contrib_value);


      // Update Community Model.
      UpdateCommunityModel();

      //Increase the number of contributors, since this is a new learner.
      community_model.set_num_contributors(community_model.num_contributors() + 1);
    }

    if (number_of_models == 2) {
      /*
       * Case: Community model is already initialized.
       * The learners are submitting their models
       * and now have two models (previous, latest)
       * with them. We remove the previous from
       * community and add the latest model to the
       * community.
      */
      PLOG(INFO) << "Case II-B triggered.";
      std::pair<const Model *, double> existing_model_pair = model_pair.front();
      const Model *existing_model = existing_model_pair.first;
      double existing_contrib_value = existing_model_pair.second;

      community_score_z = community_score_z - existing_contrib_value + new_contrib_value;

      UpdateScaledModel(existing_model,
                        new_model,
                        existing_contrib_value,
                        new_contrib_value);

      // Update Community Model.
      UpdateCommunityModel();
    }

  }

  return community_model;

}

void FederatedRecency::Reset() {
  /*
      A. No need to reset the class variables in batching.
      B. We don't need to reset anything in async.
  */

  // pass
}

}
