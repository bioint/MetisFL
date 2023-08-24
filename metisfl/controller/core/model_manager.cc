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
  }

  void InitializeAggregator(const DType_Type tensor_dtype) {
    aggregator_ = CreateAggregator(global_train_params_, tensor_dtype);
    is_initialized_ = true;
  }

  void SetInitialModel(const Model &model) override {
    PLOG(INFO) << "Received initial model.";
    *model_.mutable_model() = model;
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

  void UpdateModel(const std::vector<std::string> &learner_ids,
                   const absl::flat_hash_map<std::string, double>
                       &scaling_factors) override {
    std::lock_guard<std::mutex> model_store_guard(model_store_mutex_);
    auto update_id = metisfl::controller::GenerateRadnomId();

    metadata_[update_id] = std::make_unique<Metadata>();
    *metadata_[update_id].mutable_model_aggregation_started_at() =
        TimeUtil::GetCurrentTime();
    auto start_time_aggregation = std::chrono::high_resolution_clock::now();

    int stride_length = GetStrideLength(learner_ids.size());

    /* Since absl does not support crbeing() or iterator decrement (--) we need
       to use this. method to find the itr of the last element. */
    absl::flat_hash_map<std::string, Learner *>::iterator last_elem_itr;
    for (auto itr = learner_ids.begin(); itr != learner_ids.end(); itr++) {
      last_elem_itr = itr;
    }

    for (auto leraner_id : learner_ids) {
      std::vector<std::pair<std::string, int>> to_select_block;
      to_select_block.emplace_back(learner_id, lineage_length);
      uint32_t block_size = to_select_block.size();

      if (block_size == stride_length || itr == last_elem_itr) {
        *metadata_[update_id].mutable_model_aggregation_block_size()->Add() =
            block_size;

        auto selected_models = SelectModels(to_select_block);

        auto to_aggregate_block =
            GetAggregationPairs(selected_models, scaling_factors);

        model_ = aggregator_->Aggregate(to_aggregate_block);

        RecordTime();

        // Cleanup
        model_store_->ResetState();

      }  // end-if

    }  // end for loop

    aggregator_->Reset();

    // Compute elapsed time for the entire aggregation
    auto end_time_aggregation = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed_time_aggregation =
        end_time_aggregation - start_time_aggregation;
    metadata_[update_id].set_model_aggregation_total_duration_ms(
        elapsed_time_aggregation.count());
    *metadata_[update_id].mutable_model_aggregation_completed_at() =
        TimeUtil::GetCurrentTime();
  }

 private:
  int GetStrideLength(int num_learners) {
    uint32_t stride_length = num_learners;
    if (global_train_params_.aggregation_rule == "FedStride") {
      auto fed_stride_length = global_train_params_.stride_length;
      if (fed_stride_length > 0) {
        stride_length = fed_stride_length;
      }
    }
    return stride_length;
  }

  int GetLineageLength(std::string &learner_id) override {
    const auto lineage_length =
        model_store_->GetLearnerLineageLength(learner_id);

    int select_lineage_length =
        (lineage_length >= aggregator_->RequiredLearnerLineageLength())
            ? aggregator_->RequiredLearnerLineageLength()
            : lineage_length;

    return lineage_length;
  }

  std::map<std::string, std::vector<const Model *>> SelectModels(
      const std::vector<std::pair<std::string, int>> &to_select_block)
      override {
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
      (*metadata_[update_id]
            .mutable_model_selection_duration_ms())[selected_learner_id] =
          avg_time_selection_per_model;
    }
    return selected_models;
  }

  AggregationPairs GetAggregationPairs(
      std::map<std::string, std::vector<const Model *>> selected_models,
      const absl::flat_hash_map<std::string, double> scaling_factors) override {
    AggregationPairs to_aggregate_block;

    std::vector<std::pair<const Model *, double>>
        to_aggregate_learner_models_tmp;

    for (auto const &[selected_learner_id, selected_learner_models] :
         selected_models) {
      auto scaling_factor = scaling_factors[selected_learner_id];

      for (auto it : selected_learner_models) {
        to_aggregate_learner_models_tmp.emplace_back(it, scaling_factor);
      }

      to_aggregate_block.push_back(to_aggregate_learner_models_tmp);
      to_aggregate_learner_models_tmp.clear();
    }

    return to_aggregate_block;
  }

  void RecordTime() override {
    auto start_time_block_aggregation =
        std::chrono::high_resolution_clock::now();

    auto end_time_block_aggregation = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> elapsed_time_block_aggregation =
        end_time_block_aggregation - start_time_block_aggregation;

    *metadata_[update_id].mutable_model_aggregation_block_duration_ms()->Add() =
        elapsed_time_block_aggregation.count();

    long block_memory = GetTotalMemory();

    *metadata_[update_id].mutable_model_aggregation_block_memory_kb()->Add() =
        (double)block_memory;
  }
};
