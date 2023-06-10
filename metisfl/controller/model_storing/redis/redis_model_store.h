
#ifndef PROJECTMETIS_RC_PROJECTMETIS_CONTROLLER_MODEL_STORING_REDIS_MODEL_STORE_H_
#define PROJECTMETIS_RC_PROJECTMETIS_CONTROLLER_MODEL_STORING_REDIS_MODEL_STORE_H_

#include "metisfl/controller/model_storing/model_store.h"
#include "hiredis/hiredis.h"

#include <iostream>
#include <sstream>

namespace projectmetis::controller {

class RedisModelStore : public ModelStore {

 public:
  explicit RedisModelStore(const RedisDBStore &config);
  ~RedisModelStore() override;
  void Expunge() override;
  void ResetState() override;
  void EraseModels(const std::vector<std::string> &learner_ids) override;
  int GetConfiguredLineageLength() override;
  int GetLearnerLineageLength(std::string learner_id) override;
  void InsertModel(std::vector<std::pair<std::string, Model>> learner_pairs) override;
  std::map<std::string, std::vector<const Model*>>
  SelectModels(std::vector<std::pair<std::string, int>> learner_pairs) override;
  void Shutdown() override;

  inline std::string Name() override {
    return "RedisModelStore";
  }

 private:
  // A ctr that keeps track of models key numbers. The model key is not reusable.
  std::map<std::string, int> map_model_key_counter; 
  
  // Track the model_keys' associated with the learner. 
  std::map<std::string, std::vector<std::string>> learner_lineage_;

  redisContext *m_redis_context = nullptr;

  void EraseModel(const std::pair<std::string, std::string>& key_pair);

  // Retrieve the model_keys upto index specified.
  // Latest to oldest. <latest, prev, prev>
  std::vector<std::string> FindModelKeys(const std::string& learner_id, int index);

  // Create a model key, as we need to for pushing models on redis.
  std::string GenerateModelKey(const std::string& learner_id);

  void MakeConnection(const std::string& localhost, int port);

  // This will restrict multiple threads/learners from inserting 
  // their models in Redis using C API as its not concurrency-safe.
  std::mutex learner_mutex;

};

}

#endif //PROJECTMETIS_RC_PROJECTMETIS_CONTROLLER_MODEL_STORING_REDIS_MODEL_STORE_H_
