
#include <cmath>
#include <utility>

#include <google/protobuf/util/time_util.h>

#include "metisfl/controller/core/controller.h"
#include "metisfl/controller/core/controller_utils.h"
#include "metisfl/controller/scenarios/scenarios_common.h"
#include "metisfl/controller/common/proto_tensor_serde.h"
#include "metisfl/proto/model.pb.h"


namespace metisfl::controller
{

  using google::protobuf::util::TimeUtil;

  ControllerParams ScenariosCommon::CreateDefaultControllerParams()
  {
    // Construct default (testing) parameters to initialize controller.
    ControllerParams params;

    // Set controller server connection parameters.
    params.mutable_server_entity()->set_hostname("0.0.0.0");
    params.mutable_server_entity()->set_port(50051);

    // Set federated training protocol specifications.
    params.mutable_global_model_specs()
        ->set_learners_participation_ratio(1);

    //  FedAvg fed_avg;
    //  *params.mutable_global_model_specs()->mutable_aggregation_rule()->mutable_fed_avg() =
    //      fed_avg;
    FedStride fed_stride;
    fed_stride.set_stride_length(2);
    *params.mutable_global_model_specs()->mutable_aggregation_rule()->mutable_fed_stride() =
        fed_stride;
    AggregationRuleSpecs aggregation_rule_specs;
    aggregation_rule_specs.set_scaling_factor(AggregationRuleSpecs_ScalingFactor_NUM_TRAINING_EXAMPLES);
    *params.mutable_global_model_specs()->mutable_aggregation_rule()->mutable_aggregation_rule_specs() =
        aggregation_rule_specs;
    params.mutable_communication_specs()->set_protocol(
        CommunicationSpecs::SYNCHRONOUS);

    // Set Fully EncryptionScheme EncryptionScheme specifications.
    //  *params.mutable_fhe_scheme() = FHEScheme();

    // Set model store specifications.
    ModelStoreConfig model_store_config;
    *model_store_config.mutable_in_memory_store() = InMemoryStore();
    *params.mutable_model_store_config() = model_store_config;

    // Set model hyperparams.
    params.mutable_model_hyperparams()->set_epochs(10);
    params.mutable_model_hyperparams()->set_batch_size(100);
    params.mutable_model_hyperparams()->set_percent_validation(0);

    return params;
  }

  LearnerState ScenariosCommon::CreateLearnerState(
      int num_train_examples, int num_validation_examples, int num_test_examples)
  {
    Learner _Learner;
    DatasetSpec _dataSpec;
    LearnerState _learnerState;

    _dataSpec.set_num_training_examples(num_train_examples);
    _dataSpec.set_num_validation_examples(num_validation_examples);
    _dataSpec.set_num_test_examples(num_test_examples);

    *_Learner.mutable_dataset_spec() = _dataSpec;
    *_learnerState.mutable_learner() = _Learner;

    return _learnerState;
  }

  Model ScenariosCommon::GenerateModel(
      int num_of_tensors, int values_per_tensor, int padding, DType_Type data_type)
  {

    // The added padding is only for visual symmetry
    LOG(INFO) << "Generating model...";
    LOG(INFO) << "Tensors: " << num_of_tensors;
    LOG(INFO) << "Values per tensor: " << values_per_tensor;
    PlaintextTensor plainTextTensor;
    Tensor tensor;
    Model_Variable modelVariable;
    Model model;

    // Create a vector of the given data type with indexed-based values
    // and assign its string representation to the tensor spec.
    if (data_type == DType_Type_FLOAT32)
    {
      std::vector<float> deserialized_tensor(values_per_tensor);
      for (int index = 0; index < values_per_tensor; ++index)
      {
        deserialized_tensor[index] = (float)(index + padding);
      }
      auto serialized_tensor = proto::SerializeTensor<float>(deserialized_tensor);
      std::string serialized_tensor_str(serialized_tensor.begin(), serialized_tensor.end());
      tensor.set_value(serialized_tensor_str);
    }
    else if (data_type == DType_Type_FLOAT64)
    {
      std::vector<double> deserialized_tensor(values_per_tensor);
      for (int index = 0; index < values_per_tensor; ++index)
      {
        deserialized_tensor[index] = (index + padding);
      }
      auto serialized_tensor = proto::SerializeTensor<double>(deserialized_tensor);
      std::string serialized_tensor_str(serialized_tensor.begin(), serialized_tensor.end());
      tensor.set_value(serialized_tensor_str);
    }
    else
    {
      throw std::runtime_error("Unsupported data type.");
    }

    // Assign all remaining values to the tensor.
    tensor.set_length(values_per_tensor);
    tensor.add_dimensions(values_per_tensor);
    tensor.mutable_type()->set_type(DType_Type_FLOAT64);
    tensor.mutable_type()->set_byte_order(DType_ByteOrder_LITTLE_ENDIAN_ORDER);
    tensor.mutable_type()->set_fortran_order(false);

    (*plainTextTensor.mutable_tensor_spec()) = tensor;

    /* (1) Build a Model_Variable
       (2) Assign the plaintextTensor to Model_Variable
       (3) Assign the Model_Variable to the Model.
       (4) Repeat (1)
       (5) Return Model
    */
    for (int index = 0; index < num_of_tensors; index++)
    {
      string name = string("arr_") + to_string(index);
      modelVariable.set_name(name);
      modelVariable.set_trainable(true);
      (*modelVariable.mutable_plaintext_tensor()) = plainTextTensor;
      (*model.add_tensors()) = modelVariable;
    }

    LOG(INFO) << "Model generated!";

    return model;
  }

  ScenariosCommon::ScenariosCommon(const ControllerParams &params)
  {
    params_ = params;
    aggregation_function_ = CreateAggregator(params.global_model_specs().aggregation_rule());
    model_store_ = CreateModelStore(params.model_store_config());
    scaler_ = CreateScaler(params.global_model_specs().aggregation_rule().aggregation_rule_specs());
    scheduler_ = CreateScheduler(params.communication_specs());
    selector_ = CreateSelector();
  }

  void ScenariosCommon::AssignLearnerState(
      const std::string &learner_id, const LearnerState &learner_state)
  {
    learners_[learner_id] = learner_state;
  }

  Model ScenariosCommon::ComputeCommunityModelOPT(const std::vector<std::string> &learners_ids)
  {

    // Sentinel variables to Record community model aggregation time.
    *metadata_.mutable_model_aggregation_started_at() = TimeUtil::GetCurrentTime();
    auto start_time_aggregation = std::chrono::high_resolution_clock::now();

    Model community_model; // return variable.
    absl::flat_hash_map<std::string, LearnerState *> participating_states;

    // Select a sub-set of learners who are participating in the experiment.
    // The selection needs to be reference to learnerState to avoid copy.
    for (const auto &id : learners_ids)
    {
      participating_states[id] = &learners_.at(id);
    }

    LOG(INFO) << "Participants memory usage: " << GetTotalMemory();

    auto scaling_factors = scaler_->ComputeScalingFactors(
        community_model,
        participating_states,
        absl::flat_hash_map<std::string, TrainResults *>{});

    uint32_t aggregation_stride_length = participating_states.size();
    if (params_.global_model_specs().aggregation_rule().has_fed_stride())
    {
      auto fed_stride_length =
          params_.global_model_specs().aggregation_rule().fed_stride().stride_length();
      if (fed_stride_length > 0)
      {
        aggregation_stride_length = fed_stride_length;
      }
    }

    // Compute number of blocks.
    int num_of_blocks =
        std::ceil(participating_states.size() / aggregation_stride_length);
    LOG(INFO) << "Stride length: "
               << aggregation_stride_length
               << " and total blocks to compute: "
               << num_of_blocks;

    // Since absl does not support crbeing() or iterator decrement (--) we need to use this.
    // method to find the itr of the last element.
    absl::flat_hash_map<std::string, LearnerState *>::iterator last_elem_itr;
    for (auto itr = participating_states.begin(); itr != participating_states.end(); itr++)
    {
      last_elem_itr = itr;
    }

    std::vector<std::pair<std::string, int>> to_select_block;                      // e.g., { (learner_id, stride_length), ...}
    std::vector<std::vector<std::pair<const Model *, double>>> to_aggregate_block; // e.g., { {m1*, 0.1}, {m2*, 0.3}, ...}
    std::vector<std::pair<const Model *, double>> to_aggregate_learner_models_tmp;

    for (auto itr = participating_states.begin(); itr != participating_states.end(); itr++)
    {
      auto const &learner_id = itr->first;

      // This represents the number of models to be fetched from the back-end.
      // We need to check if the back-end has stored more models than the
      // required model number of the aggregation strategy.
      const auto learner_lineage_length =
          model_store_->GetLearnerLineageLength(learner_id);
      int select_lineage_length =
          (learner_lineage_length >= aggregation_function_->RequiredLearnerLineageLength())
              ? aggregation_function_->RequiredLearnerLineageLength()
              : learner_lineage_length;
      to_select_block.emplace_back(learner_id, select_lineage_length);

      uint32_t block_size = to_select_block.size();
      if (block_size == aggregation_stride_length || itr == last_elem_itr)
      {

        LOG(INFO) << "Computing for block size: " << block_size;
        *metadata_.mutable_model_aggregation_block_size()->Add() = block_size;

        /*! --- SELECT MODELS ---
         * Here, we retrieve models from the back-end model store.
         * We need to import k-number of models from the model store.
         * Number k depends on the number of models required by the aggregator or
         * the number of local models stored for each learner, whichever is smaller.
         *
         *  Case (1): Redis Store: we select models from an outside (external) store.
         *  Case (2): In-Memory Store: we select models from the in-memory hash map.
         *
         *  In both cases, a pointer would be returned for the models stored in the model store.
         */
        auto start_time_selection = std::chrono::high_resolution_clock::now();
        std::map<std::string, std::vector<const Model *>> selected_models =
            model_store_->SelectModels(to_select_block);
        auto end_time_selection = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed_time_selection =
            end_time_selection - start_time_selection;
        auto avg_time_selection_per_model = elapsed_time_selection.count() / block_size;
        for (auto const &[selected_learner_id, selected_learner_models] : selected_models)
        {
          (*metadata_.mutable_model_selection_duration_ms())[selected_learner_id] =
              avg_time_selection_per_model;
        }

        /* --- CONSTRUCT MODELS TO AGGREGATE --- */
        for (auto const &[selected_learner_id, selected_learner_models] : selected_models)
        {
          auto scaling_factor = scaling_factors[selected_learner_id];
          for (auto it : selected_learner_models)
          {
            to_aggregate_learner_models_tmp.emplace_back(it, scaling_factor);
          }
          to_aggregate_block.push_back(to_aggregate_learner_models_tmp);
          to_aggregate_learner_models_tmp.clear();
        }

        /* --- AGGREGATE MODELS --- */
        auto start_time_block_aggregation = std::chrono::high_resolution_clock::now();
        community_model = aggregation_function_->Aggregate(to_aggregate_block);
        auto end_time_block_aggregation = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed_time_block_aggregation =
            end_time_block_aggregation - start_time_block_aggregation;
        *metadata_.mutable_model_aggregation_block_duration_ms()->Add() =
            elapsed_time_block_aggregation.count();

        long block_memory = GetTotalMemory();
        LOG(INFO) << "Aggregate block memory usage: " << block_memory;
        *metadata_.mutable_model_aggregation_block_memory_kb()->Add() = block_memory;

        // Cleanup. Clear sentinel block variables and reset
        // model_store's state to reclaim unused memory.
        to_select_block.clear();
        to_aggregate_block.clear();
        model_store_->ResetState();

      } // end-if

    } // end-for

    // Reset aggregation function's state for the next step.
    aggregation_function_->Reset();

    // Compute elapsed time for the entire aggregation - global model computation function.
    auto end_time_aggregation = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed_time_aggregation =
        end_time_aggregation - start_time_aggregation;
    metadata_.set_model_aggregation_total_duration_ms(elapsed_time_aggregation.count());
    *metadata_.mutable_model_aggregation_completed_at() = TimeUtil::GetCurrentTime();

    return community_model;
  }

  void ScenariosCommon::InsertModelsIntoStore(std::vector<std::pair<std::string, Model>> learner_pairs)
  {
    auto start_time_insert = std::chrono::high_resolution_clock::now();
    auto learner_id = learner_pairs.front().first;
    model_store_->InsertModel(std::move(learner_pairs));
    auto end_time_insert = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed_time_insert =
        end_time_insert - start_time_insert;
    (*metadata_.mutable_model_insertion_duration_ms())[learner_id] =
        elapsed_time_insert.count();
  }

  void ScenariosCommon::ResetStore()
  {
    model_store_->Expunge();
  }

  RuntimeMetadata ScenariosCommon::GetMetadata()
  {
    return metadata_;
  }

}
