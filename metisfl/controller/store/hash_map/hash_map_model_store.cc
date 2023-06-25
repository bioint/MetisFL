
#include "metisfl/controller/store/hash_map/hash_map_model_store.h"
#include "metisfl/proto/metis.pb.h"
#include "metisfl/proto/model.pb.h"

namespace metisfl::controller {

HashMapModelStore::HashMapModelStore(const InMemoryStore &config) : ModelStore(config.model_store_specs()) {
  PLOG(INFO) << "Using InMemoryStore (HashMapStore) as model store backend.";
}

void HashMapModelStore::Expunge() {
  // This will clear all
  m_model_store_cache.clear();
}

void HashMapModelStore::EraseModels(const std::vector<std::string> &learner_ids) {

  for (auto &learner_id: learner_ids) {
    m_model_store_cache[learner_id].clear();
  }
}

int HashMapModelStore::GetConfiguredLineageLength() {
  int lineage_length = -1;
  if (m_model_store_specs.has_lineage_length_eviction()) {
    lineage_length = (int) m_model_store_specs.lineage_length_eviction().lineage_length();
  }
  return lineage_length;
}

int HashMapModelStore::GetLearnerLineageLength(std::string learner_id) {
  return (int) m_model_store_cache[learner_id].size();
}

void HashMapModelStore::InsertModel(std::vector<std::pair<std::string, Model>> learner_pairs) {
  /*
    std::vector<...> represents multiple learners. 
    std::pair<std::string, Model> represents learner_id and Model for single learner. 

    This function can input <learner_id,Model> pairs for multiple learners. 

    learner_pairs -> multiple learners. 
    learner_pair -> pair for one learner.
  */

  for (auto &learner_pair: learner_pairs) {

    std::string learner_id = learner_pair.first;
    auto model = learner_pair.second;

    // This is only applicable on the k-Recent-Models policy.
    if (m_model_store_specs.has_lineage_length_eviction()) {
      // Check to see is learner_id is present in collection
      if ((m_model_store_cache.find(learner_id) != m_model_store_cache.end())) {
        // Yes, learner_id is present in collection.
        // Check if the model being inserted is greater than max length.
        if (m_model_store_cache[learner_id].size() >=
              m_model_store_specs.lineage_length_eviction().lineage_length()) {
          auto itr_first_elem = m_model_store_cache[learner_id].begin();
          PLOG(INFO) << "Reached max limit. Erasing oldest model.";
          m_model_store_cache[learner_id].erase(itr_first_elem);
        }
      }
    }
    
    PLOG(INFO) << "Inserting model in learner_id: " << learner_id;
    m_model_store_cache[learner_id].push_back(model);

  }

}

void HashMapModelStore::ResetState() {
  // Do nothing. We do not need to clear the In-Memory store of its state.
}

std::map<std::string, std::vector<const Model*>>
HashMapModelStore::SelectModels(std::vector<std::pair<std::string, int>> learner_pairs) {

  // Order of insertion expected {old, old, old, new}
  std::map<std::string, std::vector<const Model*>> reply_models;
    
  // learner_pair - first  - learner_id as string
  // learner_pair - second - the number of models to get. 

  for (auto &learner_pair: learner_pairs) {

    std::string learner_id = learner_pair.first;
    int index = learner_pair.second; // The number of models to select from store.
    int history_size = GetLearnerLineageLength(learner_id);

    PLOG(INFO) << "Select models for learner_id: " << learner_id << " index: " << index;

    // Check if index is less than size of lineage
    // return empty models.
    if (index > m_model_store_cache[learner_id].size()) {
      PLOG(WARNING) << "Index larger than lineage size";
      reply_models[learner_id].clear();
      continue;
    }

    // If non-positive (x <= 0): reply all models
    if (index <= 0) {
      // This will return pointer to all the models stored against learner_id.
      index = history_size;
    }

    // If (x>0) reply current and num-1 latest runtime metadata.
    for (auto hidx = index; hidx > 0; hidx--) {
      const Model *ptr_latest_model = &m_model_store_cache[learner_id][history_size - hidx];
      reply_models[learner_id].push_back(ptr_latest_model);
    }

  }

  return reply_models;
}


void HashMapModelStore::Shutdown() {}

}
