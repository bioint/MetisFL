
#ifndef METISFL_METISFL_CONTROLLER_CORE_CONTROLLER_H_
#define METISFL_METISFL_CONTROLLER_CORE_CONTROLLER_H_

#include <glog/logging.h>
#include <google/protobuf/util/time_util.h>
#include <grpcpp/client_context.h>
#include <grpcpp/completion_queue.h>
#include <grpcpp/create_channel.h>
#include <grpcpp/impl/codegen/async_unary_call.h>

#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/memory/memory.h"
#include "absl/status/statusor.h"
#include "metisfl/controller/common/bs_thread_pool.h"
#include "metisfl/controller/common/macros.h"
#include "metisfl/controller/common/proto_tensor_serde.h"
#include "metisfl/controller/core/controller_utils.h"
#include "metisfl/controller/core/model_manager.h"
#include "metisfl/proto/controller.grpc.pb.h"
#include "metisfl/proto/learner.grpc.pb.h"

namespace metisfl::controller {

// Indexed by learner_id
typedef std::unique_ptr<LearnerService::Stub> LearnerStub;
typedef std::unique_ptr<std::string, LearnerStub> LearnerStubMap;
typedef absl::flat_hash_map<std::string, LearnerDescriptor *> LearnersMap;
typedef absl::flat_hash_map<std::string, TrainParams> TrainParamsMap;
typedef absl::flat_hash_map<std::string, EvaluationParams> EvaluationParamsMap;

// Indexed by task_id
typedef absl::flat_hash_map<std::string, CommunityModelEvaluation *>
    CommunityModelEvaluationMap;
typedef absl::flat_hash_map<std::string, TrainingMetadata *>
    TrainingMetadataMap;
typedef std::unique_ptr<std::string, RuntimeMetadata> RuntimeMetadataMap;

class Controller {
 public:
  virtual ~Controller() = default;

  virtual const ServerParams &GetServerParams() const = 0;

  virtual std::vector<std::string> GetLearnerIds() const = 0;

  virtual uint32_t GetNumLearners() const = 0;

  virtual const FederatedModel &CommunityModel() const = 0;

  virtual absl::Status SetInitialModel(const Model &model) = 0;

  virtual absl::StatusOr<std::string> AddLearner(
      const LearnerDescriptor &learner) = 0;

  virtual absl::Status StartTraining() = 0;

  virtual absl::Status RemoveLearner(const std::string &learner_id) = 0;

  virtual absl::Status TrainDone(const TrainDoneRequest &task) = 0;

  virtual std::vector<RuntimeMetadata> GetRuntimeMetadataLineage(
      uint32_t num_steps) = 0;

  virtual std::vector<CommunityModelEvaluation> GetEvaluationLineage(
      uint32_t num_steps) = 0;

  virtual std::vector<TaskExecutionMetadata> GetLocalTaskLineage(
      const std::string &learner_id, uint32_t num_steps) = 0;

  virtual std::string GenerateTaskId() = 0;

  virtual void Shutdown() = 0;

 private:
  std::unique_ptr<ModelManager> model_manager_;

  ServerParams server_params_;
  GlobalTrainParams global_train_params_;
  ModelStoreParams model_store_params_;

  std::mutex learners_mutex_;
  BS::thread_pool scheduling_pool_;
  LearnersMap learners_;
  LearnerStubMap learners_stub_;
  TrainParamsMap learners_train_params_;
  EvaluationParamsMap learners_eval_params_;

  TaskMetadataMap local_tasks_metadata_;
  TrainingMetadataMap metadata_;
  CommunityModelEvaluationMap community_model_evaluations_;

  grpc::CompletionQueue run_tasks_cq_;
  grpc::CompletionQueue eval_tasks_cq_;

 public:
  static std::unique_ptr<Controller> New(
      const ServerParams &server_params,
      const GlobalTrainParams &global_train_params,
      const ModelStoreParams &model_store_params);
};

}  // namespace metisfl::controller

#endif  // METISFL_METISFL_CONTROLLER_CORE_CONTROLLER_H_
