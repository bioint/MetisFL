
#ifndef PROJECTMETIS_RC_PROJECTMETIS_CONTROLLER_MODEL_STORING_HASH_MAP_MODEL_STORE_H_
#define PROJECTMETIS_RC_PROJECTMETIS_CONTROLLER_MODEL_STORING_HASH_MAP_MODEL_STORE_H_

#include "metisfl/controller/model_storing/model_store.h"
#include "metisfl/proto/model.pb.h"

namespace projectmetis::controller {

class HashMapModelStore : public ModelStore {
 public:
  // Cannot be initialized without an external store referenced by ref_learners. 
  explicit HashMapModelStore(const InMemoryStore &config);
  ~HashMapModelStore() = default;
  void Expunge() override;
  void EraseModels(const std::vector<std::string> &learner_ids) override;
  int GetConfiguredLineageLength() override;
  int GetLearnerLineageLength(std::string learner_id) override;
  void InsertModel(std::vector<std::pair<std::string, Model>> learner_pairs) override;
  void ResetState() override;
  
  std::map<std::string, std::vector<const Model*>>
  SelectModels(std::vector<std::pair<std::string, int>> learner_pairs) override;
  void Shutdown() override;

  inline std::string Name() override {
    return "HashMapModelStore";
  }

};

}

#endif //PROJECTMETIS_RC_PROJECTMETIS_CONTROLLER_MODEL_STORING_HASH_MAP_MODEL_STORE_H_
