#ifndef METISFL_CONTROLLER_CORE_TYPES_H_
#define METISFL_CONTROLLER_CORE_TYPES_H_

#include "absl/container/flat_hash_map.h"
#include "metisfl/proto/controller.grpc.pb.h"
#include "metisfl/proto/learner.grpc.pb.h"

typedef struct ServerParams {
  std::string hostname;
  int port;
  std::string public_certificate;
  std::string private_key;
  std::string root_certificate;
} ServerParams;

typedef struct GlobalTrainParams {
  std::string aggregation_rule;
  std::string communication_protocol;
  std::string scaling_factor;
  float participation_ratio;
  int stride_length;

  int he_batch_size;
  int he_scaling_factor_bits;
  std::string he_crypto_context_file;

  float semi_sync_lambda;
  int semi_sync_recompute_num_updates;
} GlobalTrainParams;

typedef struct ModelStoreParams {
  std::string model_store;
  int lineage_length;
  std::string hostname;
  int port;
} ModelStoreParams;

typedef std::unique_ptr<LearnerService::Stub> LearnerStub;

// Indexed by learner_id
typedef std::unique_ptr<std::string, LearnerStub> LearnerStubMap;
typedef absl::flat_hash_map<std::string, Learner *> LearnersMap;
typedef absl::flat_hash_map<std::string, TrainParams> TrainParamsMap;
typedef absl::flat_hash_map<std::string, EvaluationParams> EvaluationParamsMap;

// Indexed by task_id
typedef absl::flat_hash_map<std::string, std::string>
    TaskLearnerMap;  // task_id -> learner_id
typedef absl::flat_hash_map<std::string, TrainingMetadata *>
    TrainingMetadataMap;
typedef absl::flat_hash_map<std::string, EvaluationMetadata *>
    EvaluationMetadataMap;
typedef std::unique_ptr<std::string, ModelMetadata> ModelMetadataMap;

#endif  // METISFL_CONTROLLER_CORE_TYPES_H_