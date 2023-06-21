
#include <cmath>

#include <glog/logging.h>

#include "metisfl/proto/model.pb.h"
#include "metisfl/proto/metis.pb.h"
#include "metisfl/controller/scenarios/scenarios_common.h"

using namespace projectmetis::controller;

#define STRIDE_LENGTH 2

int main(int argc, char *argv[]) {

  // Verify Input Parameters
  if (argc < 3) {
    throw std::runtime_error("Insufficient input arguments. Need to provide values for:\n"
                             "Num-0f-Learners, Number-of-Tensors, Values-Per-Tensor");
  }

  // Set flags picked up by glog before initialization.
  FLAGS_log_dir = "/tmp";
  FLAGS_alsologtostderr = true;
  google::InitGoogleLogging(argv[0]);

  int num_of_learners = stoi(argv[1], nullptr, 10);
  int num_of_tensors = stoi(argv[2], nullptr, 10);
  int values_per_tensor = stoi(argv[3], nullptr, 10);

  LOG(INFO) << "Number of learners: " << num_of_learners;
  LOG(INFO) << "Number of tensors: " << num_of_tensors;
  LOG(INFO) << "Values per tensor: " << values_per_tensor;

  auto model_store_specs = projectmetis::ModelStoreSpecs();
  auto no_eviction = projectmetis::NoEviction();
  *model_store_specs.mutable_no_eviction() = no_eviction;
  auto in_memory_store_config = projectmetis::InMemoryStore();
  *in_memory_store_config.mutable_model_store_specs() = model_store_specs;
  auto controller_params = ScenariosCommon::CreateDefaultControllerParams();
  *controller_params.mutable_model_store_config()->mutable_in_memory_store() = in_memory_store_config;
  ScenariosCommon scenarios_common(controller_params);

  std::vector<std::string> learner_ids;
  // Create learners based on provided input.
  for (int index = 1; index <= num_of_learners; index++) {
    std::string learner_id = "learner_";
    learner_id += std::to_string(index);
    learner_ids.push_back(learner_id);
  }

  // Step 1: Generate the model. We can define the size of the model here.
  ::projectmetis::Model model = ScenariosCommon::GenerateModel(
      num_of_tensors, values_per_tensor, 1, projectmetis::DType_Type_FLOAT32);

  // Generate a learner state.
  ::projectmetis::LearnerState learner_state =
      ScenariosCommon::CreateLearnerState(100, 100, 100);

  LOG(INFO) << "Start! Assigning models to learners.";

  for (const auto& learner_id: learner_ids) {
    // STEP 1: Assign learner state.
    scenarios_common.AssignLearnerState(learner_id, learner_state);

    // STEP 2: Insert Models into the model_store's internal storage map.
    scenarios_common.InsertModelsIntoStore({
      std::pair<std::string, ::projectmetis::Model>(learner_id, model)
    });
  }

  LOG(INFO) << "Model insertion memory usage: " << ScenariosCommon::GetTotalMemory();
  LOG(INFO) << "End! Assigned models to all learners.";

  auto community_model = scenarios_common.ComputeCommunityModelOPT(learner_ids);

  if (community_model.num_contributors() > 0) {
    LOG(INFO) << "Number of contributors: " << community_model.num_contributors();
  } else {
    LOG(INFO) << "Not enough contributors.";
  }

  scenarios_common.ResetStore();
  LOG(INFO) << scenarios_common.GetMetadata().DebugString();

  return 0;
}
