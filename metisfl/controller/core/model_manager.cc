#include "metisfl/controller/core/model_manager.h"

namespace metisfl::controller {
class ModelManager {
 public:
  ModelManager(const GlobalTrainParams &global_train_params,
               const ModelStoreParams &model_store_params,
               const DType_Type tensor_dtype)
      : model_store_mutex_(),
        community_model_(),
        global_train_params_(global_train_params) {
    model_store_ = CreateModelStore(model_store_params);
    scaler_ = CreateScaler(global_train_params.scaling_factor);
  }

  void InitializeAggregator(const DType_Type tensor_dtype) {
    aggregator_ = CreateAggregator(global_train_params_, tensor_dtype);
    is_initialized_ = true;
  }

  void SetInitialModel(const Model &model) override {
    PLOG(INFO) << "Received initial model.";
    *model_.mutable_model() = model;

    // Initialize the aggregator with the model's tensor type.
    InitializeAggregator(model_.model().tensors(0).type().type());
  }

  const TensorQuantifier &GetModelSize() const override {
    // TODO:
  }

  void InsertModel(const std::vector<std::string> &learner_id,
                   const Model model) override {
    std::lock_guard<std::mutex> model_store_guard(model_store_mutex_);

    model_store_->InsertModel(std::vector<std::pair<std::string, Model>>{
        std::pair<std::string, Model>(learner_id, model)});
  }

  const EraseModels(const std::vector<std::string> &learner_id) override {
    std::lock_guard<std::mutex> model_store_guard(model_store_mutex_);
    model_store_->EraseModels(learner_id);
  }

  void UpdateModel(LearnersMap selected_learners,
                   TaskMetadataMap selected_task_metadata,
                   const uint32_t &metadata_ref_idx) {
    // Handles the case where the community model is requested for the
    // first time and has the original (random) initialization state.
    if (global_iteration_ < 1 && community_model_.IsInitialized()) {
      return community_model_;
    }

    std::lock_guard<std::mutex> model_store_guard(model_store_mutex_);

    *metadata_.at(metadata_ref_idx).mutable_model_aggregation_started_at() =
        TimeUtil::GetCurrentTime();
    auto start_time_aggregation = std::chrono::high_resolution_clock::now();

    FederatedModel new_community_model;

    auto scaling_factors = scaler_->ComputeScalingFactors(
        community_model_, learners_, selected_learners, selected_task_metadata);

    // Defines the length of the aggregation stride, i.e., how many models
    // to fetch from the model store and feed to the aggregation function.
    // Only FedStride does this stride-based aggregation. All other aggregation
    // rules use the entire list of participating models.
    uint32_t aggregation_stride_length = selected_learners.size();
    if (global_train_params_.aggregation_rule == "FedStride") {
      auto fed_stride_length = global_train_params_.stride_length;
      if (fed_stride_length > 0) {
        aggregation_stride_length = fed_stride_length;
      }
    }

    /* Since absl does not support crbeing() or iterator decrement (--) we need
       to use this. method to find the itr of the last element. */
    absl::flat_hash_map<std::string, LearnerDescriptor *>::iterator
        last_elem_itr;
    for (auto itr = selected_learners.begin(); itr != selected_learners.end();
         itr++) {
      last_elem_itr = itr;
    }

    std::vector<std::pair<std::string, int>>
        to_select_block;  // e.g., { (learner_id, stride_length), ...}
    AggregationPairs
        to_aggregate_block;  // e.g., { {m1*, 0.1}, {m2*, 0.3}, ...}
    std::vector<std::pair<const Model *, double>>
        to_aggregate_learner_models_tmp;

    for (auto itr = selected_learners.begin(); itr != selected_learners.end();
         itr++) {
      auto const &learner_id = itr->first;

      // This represents the number of models to be fetched from the back-end.
      // We need to check if the back-end has stored more models than the
      // required model number of the aggregation strategy.
      const auto learner_lineage_length =
          model_store_->GetLearnerLineageLength(learner_id);

      int select_lineage_length =
          (learner_lineage_length >=
           aggregator_->RequiredLearnerLineageLength())
              ? aggregator_->RequiredLearnerLineageLength()
              : learner_lineage_length;
      to_select_block.emplace_back(learner_id, select_lineage_length);

      uint32_t block_size = to_select_block.size();
      if (block_size == aggregation_stride_length || itr == last_elem_itr) {
        PLOG(INFO) << "Computing for block size: " << block_size;
        *metadata_.at(metadata_ref_idx)
             .mutable_model_aggregation_block_size()
             ->Add() = block_size;

        /*! --- SELECT MODELS ---
         * Here, we retrieve models from the back-end model store.
         * We need to import k-number of models from the model store.
         * Number k depends on the number of models required by the aggregator
         * or the number of local models stored for each learner, whichever is
         * smaller.
         *
         *  Case (1): Redis Store: we select models from an outside (external)
         * store. Case (2): In-Memory Store: we select models from the in-memory
         * hash map.
         *
         *  In both cases, a pointer would be returned for the models stored in
         * the model store.
         */
        auto start_time_selection = std::chrono::high_resolution_clock::now();

        std::map<std::string, std::vector<const Model *>> selected_models =
            model_store_->SelectModels(to_select_block);

        auto end_time_selection = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> elapsed_time_selection =
            end_time_selection - start_time_selection;

        auto avg_time_selection_per_model =
            elapsed_time_selection.count() / block_size;

        for (auto const &[selected_learner_id, selected_learner_models] :
             selected_models) {
          (*metadata_.at(metadata_ref_idx)
                .mutable_model_selection_duration_ms())[selected_learner_id] =
              avg_time_selection_per_model;
        }

        /* --- CONSTRUCT MODELS TO AGGREGATE --- */
        for (auto const &[selected_learner_id, selected_learner_models] :
             selected_models) {
          auto scaling_factor = scaling_factors[selected_learner_id];
          for (auto it : selected_learner_models) {
            to_aggregate_learner_models_tmp.emplace_back(it, scaling_factor);
          }
          to_aggregate_block.push_back(to_aggregate_learner_models_tmp);
          to_aggregate_learner_models_tmp.clear();
        }

        /* --- AGGREGATE MODELS --- */
        // FIXME(@stripeli): When using Redis as backend and setting to
        //  store all models (i.e., EvictionPolicy = NoEviction) then
        //  the collection of models passed here are all models and
        //  therefore the number of variables becomes inconsistent with the
        //  number of variables of the original model. For instance, if a
        //  learner has two models stored in the collection and each model
        //  has 6 variables then the total variables of the sampled model
        //  will be 12 not 6 (the expected)!. Try the following for testing:
        //    auto total_sampled_vars = sample_model->variables_size();
        //    PLOG(INFO) << "TOTAL SAMPLED VARS:" << total_sampled_vars;

        auto start_time_block_aggregation =
            std::chrono::high_resolution_clock::now();

        new_community_model = aggregator_->Aggregate(to_aggregate_block);

        auto end_time_block_aggregation =
            std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli>
            elapsed_time_block_aggregation =
                end_time_block_aggregation - start_time_block_aggregation;

        *metadata_.at(metadata_ref_idx)
             .mutable_model_aggregation_block_duration_ms()
             ->Add() = elapsed_time_block_aggregation.count();

        long block_memory = GetTotalMemory();

        PLOG(INFO) << "Aggregate block memory usage (kb): " << block_memory;

        *metadata_.at(metadata_ref_idx)
             .mutable_model_aggregation_block_memory_kb()
             ->Add() = (double)block_memory;

        // Cleanup
        to_select_block.clear();
        to_aggregate_block.clear();
        model_store_->ResetState();

      }  // end-if

    }  // end for loop

    // Reset aggregation function's state for the next step.
    aggregator_->Reset();

    // Compute elapsed time for the entire aggregation
    auto end_time_aggregation = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed_time_aggregation =
        end_time_aggregation - start_time_aggregation;
    metadata_.at(metadata_ref_idx)
        .set_model_aggregation_total_duration_ms(
            elapsed_time_aggregation.count());
    *metadata_.at(metadata_ref_idx).mutable_model_aggregation_completed_at() =
        TimeUtil::GetCurrentTime();

    return new_community_model;
  }
};
