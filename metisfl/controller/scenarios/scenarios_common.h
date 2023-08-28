
#include <errno.h>
#include <stdio.h>
#include <sys/resource.h>

#include <cmath>
#include <utility>

#include "metisfl/controller/core/controller.h"
#include "metisfl/controller/core/controller_utils.h"
#include "metisfl/controller/core/proto_tensor_serde.h"
#include "metisfl/controller/model_aggregation/federated_average.h"
#include "metisfl/controller/model_aggregation/federated_recency.h"
#include "metisfl/controller/model_aggregation/federated_rolling_average_base.h"
#include "metisfl/controller/model_aggregation/federated_stride.h"
#include "metisfl/controller/scaling/scaling.h"
#include "metisfl/controller/store/store.h"
#include "metisfl/proto/model.pb.h"

namespace metisfl::controller {

extern int errno;
using namespace std;

class ScenariosCommon {
  /*
   * Defined a Scenarios Common Class for both fed_roll_test.cc which is a unit
   * testing file and aggregation_stress_test.cc file which is an integrated
   * testing file used for performing experimentation.
   * */
 private:
  ControllerParams params_;
  RuntimeMetadata metadata_;
  std::unique_ptr<AggregationFunction> aggregation_function_;
  std::unique_ptr<ModelStore> model_store_;
  std::unique_ptr<ScalingFunction> scaler_;
  std::unique_ptr<Scheduler> scheduler_;
  std::unique_ptr<Selector> selector_;

  /* learners_ variable is used as follows:
     (1) In Redis Store : Bring models from Store and keep in learners_
     ephemerally. (2) In In-Memory: Keep models in learners_ permanently to
     avoid double memory cost.
  */
  std::map<std::string, LearnerState> learners_;
  std::map<std::string, std::vector<Model>> learners_models_;

 public:
  /**
   * Constructors.
   */
  explicit ScenariosCommon(const ControllerParams &params);

  /**
   * Static class functions.
   */
  static ControllerParams CreateDefaultControllerParams();

  static LearnerState CreateLearnerState(int num_train_examples,
                                         int num_validation_examples,
                                         int num_test_examples);

  static Model GenerateModel(int num_of_tensors, int values_per_tensor,
                             int padding, DType_Type data_type);

  static long GetTotalMemory();

  /**
   * Member class functions.
   */
  void AssignLearnerState(const std::string &learner_id,
                          const LearnerState &learner_state);

  Model ComputeCommunityModelOPT(const std::vector<std::string> &learners_ids);

  void InsertModelsIntoStore(
      std::vector<std::pair<std::string, Model>> learner_pairs);

  RuntimeMetadata GetMetadata();

  void ResetStore();
};

}  // namespace metisfl::controller
