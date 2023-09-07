
#include "metisfl/controller/store/redis/redis_model_store.h"

namespace metisfl::controller {

RedisModelStore::RedisModelStore(const std::string &hostname, const int port,
                                 const int lineage_length)
    : ModelStore(lineage_length) {
  MakeConnection(hostname, port);
  LOG(INFO) << "Using RedisDB as model store backend.";
}

RedisModelStore::~RedisModelStore() {
  redisFree(m_redis_context);
  LOG(INFO) << "Disconnected from Redis.";
}

void RedisModelStore::Expunge() {
  // WARNING: flushing the entire database.
  LOG(WARNING) << "Flush Redis Database.";
  auto *redis_reply = (redisReply *)redisCommand(m_redis_context, "flushdb");
  freeReplyObject(redis_reply);

  learner_lineage_.clear();
}

int RedisModelStore::GetConfiguredLineageLength() { return m_lineage_length; }

int RedisModelStore::GetLearnerLineageLength(std::string learner_id) {
  return (int)learner_lineage_[learner_id].size();
}

void RedisModelStore::EraseModels(const std::vector<std::string> &learner_ids) {
  // Means to remove all models against learner_id.
  // We would need to remove models before removing entry form Controller's
  // learners_ collection.

  for (auto &learner_id : learner_ids) {
    // Get all model keys associated with the learner_id.
    std::vector<std::string> model_keys = FindModelKeys(learner_id, 0);
    for (const auto &model_key : model_keys) {
      LOG(INFO) << "Erasing models: " << model_key << std::endl;
      std::string get_command = "DEL " + model_key;

      auto *redis_reply =
          (redisReply *)redisCommand(m_redis_context, get_command.c_str());
      freeReplyObject(redis_reply);
    }
    learner_lineage_[learner_id].clear();
  }
}

void RedisModelStore::InsertModel(
    std::vector<std::pair<std::string, Model>> learner_pairs) {
  std::lock_guard<std::mutex> lock(learner_mutex);

  for (auto &learner_pair : learner_pairs) {
    std::string learner_id = learner_pair.first;
    Model model = learner_pair.second;

    // This is only applicable on the k-Recent-Models policy.
    if (m_lineage_length > 0) {
      // Check to see is learner_id is present in collection
      if ((learner_lineage_.find(learner_id) != learner_lineage_.end())) {
        // Yes, learner_id is present in collection.
        // Check if the model being inserted is greater than max length.
        if (learner_lineage_[learner_id].size() >= m_lineage_length) {
          auto itr_first_elem = learner_lineage_[learner_id].begin();
          LOG(INFO) << "Reached max limit.";
          EraseModel(
              std::pair<std::string, std::string>(learner_id, *itr_first_elem));
        }
      }
    }

    // Generate a unique model_key for learner_id
    std::string model_key = GenerateModelKey(learner_id);

    LOG(INFO) << "Adding Model in Redis for Model key " << model_key;

    // The Model is inserted a List where each entry is a serialized
    // Model_Variable We choose this design over serializing whole model for
    // scalability.
    Model &to_serialize_mdl = model;

    for (int index = 0; index < (int)to_serialize_mdl.tensors_size(); index++) {
      std::string tensor_serialized;
      const Tensor &to_serialize_mv = to_serialize_mdl.tensors(index);
      to_serialize_mv.SerializeToString(&tensor_serialized);

      auto *redis_reply = (redisReply *)redisCommand(
          m_redis_context, "RPUSH %b %b", model_key.c_str(),
          (size_t)model_key.length(), tensor_serialized.c_str(),
          (size_t)tensor_serialized.length());
      // TODO(stripeli) Need to check for error (if any) and handle it.
      freeReplyObject(redis_reply);
    }

    // Model Inserted into Redis Successfully. Update learner_lineage_
    // reference.
    learner_lineage_[learner_id].push_back(model_key);
  }
}

void RedisModelStore::ResetState() {
  // Erase all models as they are no longer needed. Reclaim the memory.
  LOG(INFO) << "Removing Models! Processed Batch Size: "
             << m_model_store_cache.size();
  m_model_store_cache.clear();
}

std::map<std::string, std::vector<const Model *>> RedisModelStore::SelectModels(
    std::vector<std::pair<std::string, int>> learner_pairs) {
  std::lock_guard<std::mutex> lock(learner_mutex);

  // Order of insertion expected {old, old, old, new}
  std::map<std::string, std::vector<const Model *>> reply_models;

  // learner_pair - first  - learner_id as string
  // learner_pair - second - the number of models to get.

  // The current select implementation performs a multi-transaction at the model
  // level. That is it tries to fetch multiple models per learner through a
  // single transaction.
  // TODO(stripeli) We could extend this to support multi-transaction per
  // learner ids
  //  and models. This would be the Batch-Selection query.
  for (auto &learner_pair : learner_pairs) {
    std::string learner_id = learner_pair.first;
    int index =
        learner_pair.second;  // The number of models to select from store.
    int lineage_length = GetLearnerLineageLength(learner_id);

    // Check if index is less than size of
    // lineage return empty models.
    if (index > learner_lineage_[learner_id].size()) {
      LOG(WARNING) << "Index larger than lineage size";
      reply_models[learner_id].clear();
      continue;
    }

    // If non-positive (x <= 0): reply all models
    if (index <= 0) {
      // This will return pointer to all the models stored against learner_id.
      index = lineage_length;
    }

    // Get the keys for the models we want to return.
    std::vector<std::string> model_keys = FindModelKeys(learner_id, index);

    // Step #1: Start the Redis Transaction to submit all the queries.
    auto *redis_reply = (redisReply *)redisCommand(m_redis_context, "MULTI");
    freeReplyObject(redis_reply);

    auto start_select_time = std::chrono::high_resolution_clock::now();

    // Step #2: Submit all queries.
    for (const auto &model_key : model_keys) {
      LOG(INFO) << "Select from Redis, Model: " << model_key
                 << " learner_id: " << learner_id;

      std::string get_command = "LRANGE " + model_key + " 0 -1";

      redis_reply =
          (redisReply *)redisCommand(m_redis_context, get_command.c_str());

      freeReplyObject(redis_reply);
    }

    // Step #3: End Transaction, and get the result of the whole transaction.
    redis_reply = (redisReply *)redisCommand(m_redis_context, "EXEC");

    auto elapsed_start_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - start_select_time);
    LOG(INFO) << "Model Select Time: " << elapsed_start_time.count() << " ms";

    auto start_model_desz = std::chrono::high_resolution_clock::now();  // temp

    // Step #4: Deserialize the model from the Redis Reply.
    // The values in the reply are stored in-order of their transaction query.
    for (auto list_index = 0; list_index < (int)redis_reply->elements;
         list_index++) {
      auto *model_reply = redis_reply->element[list_index];
      Model model;

      for (auto idx1 = 0; idx1 < (int)model_reply->elements; idx1++) {
        /* Parse The Serialized String into a regular String for building new
         * Model */
        std::string str_construct_tensor;
        for (auto idx2 = 0; idx2 < (int)model_reply->element[idx1]->len;
             idx2++) {
          str_construct_tensor += model_reply->element[idx1]->str[idx2];
        }

        ::metisfl::Tensor modelVariable;
        modelVariable.ParseFromString(str_construct_tensor);  // String //
        (*model.add_tensors()) = modelVariable;
      }
      /* We need to store the Models imported from Redis into
      a variable that lives till batch completion. Defined in step#1 and step#2
    */

      // Step#1: Store into the local temp store.
      m_model_store_cache[learner_id].push_back(model);
    }

    freeReplyObject(redis_reply);

    auto elapsed_model_desz_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - start_model_desz);
    LOG(INFO) << "Model Desz Time " << elapsed_model_desz_time.count()
               << " ms";

    /* Step#2: Return reference to the local temp store.
    This formula ensures that if multiple models are required
    they are all to be returned as pointer in correct order. */

    for (auto hidx = index; hidx > 0; hidx--) {
      const Model *ptr_latest_model =
          &m_model_store_cache[learner_id][lineage_length - hidx];
      reply_models[learner_id].push_back(ptr_latest_model);
    }
  }

  return reply_models;
}

std::string RedisModelStore::GenerateModelKey(const std::string &learner_id) {
  // Model Key = <learner_id> + <Local Model ID>
  std::string model_key =
      learner_id + "_" + std::to_string(map_model_key_counter[learner_id]++);

  return model_key;
}

std::vector<std::string> RedisModelStore::FindModelKeys(
    const std::string &learner_id, int index) {
  std::vector<std::string> model_keys_;

  // If non-positive (x <= 0): reply all model keys
  if (index <= 0) {
    return learner_lineage_[learner_id];
  }

  // If (x>0) reply current and num-1 latest runtime metadata.
  uint32_t last_index = learner_lineage_[learner_id].size() - 1;
  int counter = 0;
  while (counter++ < index) {
    model_keys_.push_back(learner_lineage_[learner_id][last_index--]);
  }

  return model_keys_;
}

void RedisModelStore::MakeConnection(const std::string &localhost = "127.0.0.1",
                                     int32_t port = 6379) {
  struct timeval timeout = {1, 500000};  // 1.5 seconds
  m_redis_context = redisConnectWithTimeout(localhost.c_str(), port, timeout);
  if (m_redis_context == nullptr || m_redis_context->err) {
    if (m_redis_context) {
      LOG(ERROR) << "Connection error: " << m_redis_context->errstr;
      redisFree(m_redis_context);
    } else {
      LOG(ERROR) << "Connection error: can't allocate redis context";
    }
    exit(1);
  }

  LOG(INFO) << "Connected to Redis with addr " << localhost << " port " << port
             << ".";
}

void RedisModelStore::EraseModel(
    const std::pair<std::string, std::string> &key_pair) {
  // Delete a specific model key for a learner_id from the collection.

  std::string learner_id = key_pair.first;
  std::string model_key = key_pair.second;

  auto key_to_remove = std::find(learner_lineage_[learner_id].begin(),
                                 learner_lineage_[learner_id].end(), model_key);
  std::string s_key_to_remove = (*key_to_remove);
  if (!s_key_to_remove.empty()) {
    LOG(INFO) << "Erasing model with key: " << s_key_to_remove << std::endl;
    std::string get_command = "DEL " + model_key;
    auto *redis_reply =
        (redisReply *)redisCommand(m_redis_context, get_command.c_str());
    freeReplyObject(redis_reply);

    learner_lineage_[learner_id].erase(key_to_remove);
  }
}

void RedisModelStore::Shutdown() {
  redisFree(m_redis_context);
  LOG(INFO) << "Disconnected from Redis.";
}

}  // namespace metisfl::controller
