
#ifndef METISFL_METISFL_CONTROLLER_STORE_HASH_MAP_HASH_MAP_MODEL_STORE_H_
#define METISFL_METISFL_CONTROLLER_STORE_HASH_MAP_HASH_MAP_MODEL_STORE_H_

#include "metisfl/controller/store/model_store.h"
#include "metisfl/proto/model.pb.h"

namespace metisfl::controller {

class HashMapModelStore : public ModelStore {
  std::mutex m_model_store_cache_mutex;

 public:
  // Cannot be initialized without an external store referenced by ref_learners.
  explicit HashMapModelStore(const int lineage_length);

  ~HashMapModelStore() = default;

  void Expunge() override;

  void EraseModels(const std::vector<std::string> &learner_ids) override;

  int GetConfiguredLineageLength() override;

  int GetLearnerLineageLength(std::string learner_id) override;

  void InsertModel(
      std::vector<std::pair<std::string, Model>> learner_pairs) override;

  void ResetState() override;

  std::map<std::string, std::vector<const Model *>> SelectModels(
      std::vector<std::pair<std::string, int>> learner_pairs) override;

  void Shutdown() override;

  inline std::string Name() override { return "HashMapModelStore"; }
};

}  // namespace metisfl::controller

#endif  // METISFL_METISFL_CONTROLLER_STORE_HASH_MAP_HASH_MAP_MODEL_STORE_H_
