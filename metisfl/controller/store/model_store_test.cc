
#include "metisfl/controller/store/store.h"
#include "metisfl/controller/common/proto_tensor_serde.h"

#include <gtest/gtest.h>

#include <iostream>

using namespace std;

// Need to create a test suite for Model key generation.

namespace projectmetis::controller {
namespace {

// This variable needs to set to true once. When running the test suite,
// the constructor is called repeatedly by the main process and the google
// logger is set to true every time. Therefore, we need to add this sentinel
// variable to protect from setting the logger multiple times.
bool GOOGLE_LOGGING_IS_SET = false;

class ModelStoreTest : public ::testing::Test {
 public:
  // The configuration for the backend.
  ModelStoreSpecs store_specs;
  ModelStoreConfig store_config;
  std::map<std::string, LearnerState> learners_;
  LearnerState learner_state;
  std::unique_ptr<ModelStore> model_store;

 public:
  ModelStoreTest() {
    if (!GOOGLE_LOGGING_IS_SET) {
      google::InitGoogleLogging(__FILE__);
      FLAGS_alsologtostderr = true;
      FLAGS_log_dir = "/tmp";
      GOOGLE_LOGGING_IS_SET = true;
    }
  }

  void InitModelStore(const ModelStoreConfig &config) {

    if (config.has_in_memory_store()) {
      model_store = std::make_unique<HashMapModelStore>(config.in_memory_store());
    }

    if (config.has_redis_db_store()) {
      model_store = std::make_unique<RedisModelStore>(config.redis_db_store());
    }

  }

  static Model GenerateModel(int values_per_tensor = 1000, int num_of_tensors = 1000, int padding = 1) {

    PlaintextTensor plainTextTensor;
    TensorSpec tensor_spec;
    Model_Variable modelVariable;
    Model model;

    // Create a vector of doubles.
    std::vector<double> deserialized_tensor(values_per_tensor);
    for (int index = 0; index < values_per_tensor; ++index) {
      // double value = (double) rand()/RAND_MAX;
      deserialized_tensor[index] = (index + padding);
    }

    // Serialize vector of doubles to bytes
    // convert bytes to string.
    auto serialized_tensor = proto::SerializeTensor<double>(deserialized_tensor);
    std::string serialized_tensor_str(serialized_tensor.begin(), serialized_tensor.end());

    // Assign the respective values to tensor_spec.
    tensor_spec.set_length(values_per_tensor);
    tensor_spec.add_dimensions(values_per_tensor);
    tensor_spec.set_value(serialized_tensor_str);
    tensor_spec.mutable_type()->set_type(DType_Type_FLOAT64);
    tensor_spec.mutable_type()->set_byte_order(DType_ByteOrder_LITTLE_ENDIAN_ORDER);
    tensor_spec.mutable_type()->set_fortran_order(false);

    (*plainTextTensor.mutable_tensor_spec()) = tensor_spec;

    /* (1) Build a Model_Variable
       (2) Assign the plaintextTensor to Model_Variable
       (3) Assign the Model_Variable to the Model.
       (4) Repeat (1)
       (5) Return Model
    */
    for (int index = 0; index < num_of_tensors; index++) {
      string name = string("arr_") + to_string(index);
      modelVariable.set_name(name);
      modelVariable.set_trainable(true);
      (*modelVariable.mutable_plaintext_tensor()) = plainTextTensor;

      (*model.add_variables()) = modelVariable;
    }

    return model;
  }

  void InsertOneModelSingleLearner(const ModelStoreConfig &config) {

    InitModelStore(config);
    Model model = GenerateModel();
    std::string learner_id = "localhost::50051";
    std::vector<std::pair<std::string, int>> learner_pairs;

    std::pair<std::string, Model> learner_pair{learner_id, model};
    std::pair<std::string, int> learner_pair_i{learner_id, 1};

    learner_pairs.push_back(learner_pair_i);

    model_store->InsertModel(std::vector<std::pair<std::string, Model>>{learner_pair});
    auto ret = model_store->SelectModels(learner_pairs);

    model_store->Expunge();
    EXPECT_EQ(ret[learner_id].size(), 1);
  }

  void InsertThreeModelsSingleLearner(const ModelStoreConfig &config) {
    InitModelStore(config);
    Model model = GenerateModel();
    std::string learner_id = "localhost::50051";
    std::vector<std::pair<std::string, int>> learner_pairs;

    std::pair<std::string, Model> learner_pair{learner_id, model};
    std::pair<std::string, int> learner_pair_i{learner_id, 3};

    learner_pairs.push_back(learner_pair_i);

    int counter = 0;
    while (counter++ < 3) {
      model_store->InsertModel(std::vector<std::pair<std::string, Model>>{learner_pair});
    }
    //std::pair<std::string, int>
    auto ret =
        model_store->SelectModels(std::vector<std::pair<std::string, int>>{std::pair<std::string, int>(learner_id, 3)});

    model_store->Expunge();
    EXPECT_EQ(ret[learner_id].size(), 3);
  }

  void InsertThreeModelsMultipleLearners(const ModelStoreConfig &config) {
    InitModelStore(config);
    Model model = GenerateModel();
    std::string learner_id_1 = "localhost::50051";
    std::string learner_id_2 = "localhost::50052";
    std::vector<std::pair<std::string, int>> learner_pairs;

    std::pair<std::string, Model> learner_pair_1{learner_id_1, model};
    std::pair<std::string, Model> learner_pair_2{learner_id_2, model};

    learner_pairs.emplace_back(std::pair<std::string, int>(learner_id_1, 3));
    learner_pairs.emplace_back(std::pair<std::string, int>(learner_id_2, 3));

    int counter = 0;
    while (counter++ < 3) {
      model_store->InsertModel(std::vector<std::pair<std::string, Model>>{learner_pair_1, learner_pair_2});
    }
    auto ret = model_store->SelectModels(learner_pairs);

    model_store->Expunge();
    EXPECT_EQ(ret[learner_id_1].size(), 3);
    EXPECT_EQ(ret[learner_id_2].size(), 3);

  }

  void TestCountOfModelsInserted(const ModelStoreConfig &config, int count_models_to_insert) {

    InitModelStore(config);
    Model model = GenerateModel();
    std::string learner_id = "localhost::50051";
    std::vector<std::pair<std::string, int>> learner_pairs;

    std::pair<std::string, Model> learner_pair{learner_id, model};
    std::pair<std::string, int> learner_pair_i{learner_id, 3};

    learner_pairs.push_back(learner_pair_i);

    int counter = 0;
    while (counter++ < count_models_to_insert) {
      model_store->InsertModel(std::vector<std::pair<std::string, Model>>{learner_pair});
    }
    //std::pair<std::string, int>
    int model_count = model_store->GetLearnerLineageLength(learner_id);

    model_store->Expunge();
    EXPECT_EQ(model_count, count_models_to_insert);

  }

  void RequestMoreModelsThanInsertedSingleLearner(const ModelStoreConfig &config) {
    // requested more models than in store -> return 0 models.
    InitModelStore(config);
    Model model = GenerateModel();
    std::string learner_id = "localhost::50051";
    std::vector<std::pair<std::string, int>> learner_pairs;

    std::pair<std::string, Model> learner_pair{learner_id, model};
    std::pair<std::string, int> learner_pair_i{learner_id, 4};

    learner_pairs.push_back(learner_pair_i);

    int counter = 0;
    while (counter++ < 3) {
      model_store->InsertModel(std::vector<std::pair<std::string, Model>>{learner_pair});
    }
    auto ret = model_store->SelectModels(learner_pairs);

    model_store->Expunge();
    EXPECT_EQ(ret[learner_id].size(), 0);
  }

  void RequestMoreModelsThanInsertedMultipleLearners(const ModelStoreConfig &config) {
    // requested more models than in store -> return 0 models for all learners.
    InitModelStore(config);
    Model model = GenerateModel();
    std::string learner_id_1 = "localhost::50051";
    std::string learner_id_2 = "localhost::50052";
    std::vector<std::pair<std::string, int>> learner_pairs;

    std::pair<std::string, Model> learner_pair_1{learner_id_1, model};
    std::pair<std::string, Model> learner_pair_2{learner_id_2, model};

    learner_pairs.emplace_back(std::pair<std::string, int>(learner_id_1, 4));
    learner_pairs.emplace_back(std::pair<std::string, int>(learner_id_2, 4));

    int counter = 0;
    while (counter++ < 3) {
      model_store->InsertModel(std::vector<std::pair<std::string, Model>>{learner_pair_1, learner_pair_2});
    }
    auto ret = model_store->SelectModels(learner_pairs);

    model_store->Expunge();
    EXPECT_EQ(ret[learner_id_1].size(), 0);
    EXPECT_EQ(ret[learner_id_2].size(), 0);
  }

  void ActivateEvictionModelsRedisSingleLearner(const ModelStoreConfig &config) {
    InitModelStore(config);
    Model model = GenerateModel();
    std::string learner_id = "localhost::50051";
    std::vector<std::pair<std::string, int>> learner_pairs;

    std::pair<std::string, Model> learner_pair{learner_id, model};
    std::pair<std::string, int> learner_pair_i{learner_id, 10};

    learner_pairs.push_back(learner_pair_i);

    // Insert 10 models and verify size.
    int counter = 0;
    while (counter++ < 10) {
      model_store->InsertModel(std::vector<std::pair<std::string, Model>>{learner_pair});
    }
    auto ret = model_store->SelectModels(learner_pairs);
    EXPECT_EQ(ret[learner_id].size(), 10);

    // Insert 11th model and verify size.
    model_store->InsertModel(std::vector<std::pair<std::string, Model>>{learner_pair});
    auto ret2 = model_store->SelectModels(learner_pairs);
    EXPECT_EQ(ret2[learner_id].size(), 10);

    // Insert 12th model and verify size.
    counter = 0;
    while (counter++ < 5) {
      model_store->InsertModel(std::vector<std::pair<std::string, Model>>{learner_pair});
    }
    auto ret3 = model_store->SelectModels(learner_pairs);
    EXPECT_EQ(ret3[learner_id].size(), 10);

    model_store->Expunge();
  }

  void TestNoEvictionConstructorSettings(int max_lineage_length) const {

    // Create a model
    Model model = GenerateModel();
    std::string learner_id = "localhost::50051";

    // Insert X number of models into the Store.
    int counter = 0;
    while (counter++ < max_lineage_length) {
      model_store->InsertModel(std::vector<std::pair<std::string, Model>>{
          std::pair<std::string, Model>(learner_id, model)});
    }

    // Query the X number of models from the Store.
    auto ret = model_store->SelectModels(std::vector<std::pair<std::string, int>>{
        std::pair<std::string, int>(learner_id, max_lineage_length)});
    EXPECT_EQ(ret[learner_id].size(), max_lineage_length);

    // Insert 5 more models into the Store
    counter = 0;
    while (counter++ < 5) {
      model_store->InsertModel(std::vector<std::pair<std::string, Model>>{
          std::pair<std::string, Model>(learner_id, model)});
    }

    // Query the X+5 models from the Store.
    auto ret2 = model_store->SelectModels(std::vector<std::pair<std::string, int>>{
        std::pair<std::string, int>(learner_id, max_lineage_length + 5)});
    EXPECT_EQ(ret2[learner_id].size(), (max_lineage_length + 5));

    // Result X+5 models will never get evicted.

  }

  void TestEvictionConstructorSettings(int max_lineage_length) const {

    EXPECT_EQ(model_store->GetConfiguredLineageLength(), max_lineage_length);

    Model model = GenerateModel();
    std::string learner_id = "localhost::50051";

#define MODEL_INSERT model_store->InsertModel(std::vector<std::pair<std::string, Model>>{std::pair<std::string, Model>(learner_id, model)})
#define MODEL_SELECT  model_store->SelectModels(std::vector<std::pair<std::string, int>>{std::pair<std::string, int>(learner_id, max_lineage_length)})

    // Insert X models and verify size.
    int counter = 0;
    while (counter++ < max_lineage_length) {
      MODEL_INSERT;
    }
    auto ret = MODEL_SELECT;
    EXPECT_EQ(ret[learner_id].size(), max_lineage_length);

    // Insert one model and verify size.
    MODEL_INSERT;
    auto ret2 = MODEL_SELECT;
    EXPECT_EQ(ret2[learner_id].size(), max_lineage_length);

    // Insert X more model and verify size.
    counter = 0;
    while (counter++ < max_lineage_length) {
      MODEL_INSERT;
    }
    auto ret3 = MODEL_SELECT;
    EXPECT_EQ(ret3[learner_id].size(), max_lineage_length);


    // At the end of all iterations, the value of the max_mode_size should remain the same, and all other models should drop.
    model_store->Expunge();
  }

};

class InMemoryModelStoreTest : public ModelStoreTest {
 public:
  InMemoryStore in_memory_store;

  void ConfigModelStore(int32_t lineage_length) {
    if (lineage_length == -1) {
      NoEviction no_eviction;
      *store_specs.mutable_no_eviction() = no_eviction;
    } else {
      LineageLengthEviction lineage_length_eviction;
      lineage_length_eviction.set_lineage_length(lineage_length);
      *store_specs.mutable_lineage_length_eviction() = lineage_length_eviction;
    }
    (*in_memory_store.mutable_model_store_specs()) = store_specs;
    (*store_config.mutable_in_memory_store()) = in_memory_store;
  }

};

class RedisModelStoreTest : public ModelStoreTest {
 public:
  RedisDBStore redis_db_store;
  ServerEntity server_entity;

  void ConfigModelStore(int32_t lineage_length) {
    if (lineage_length == -1) {
      NoEviction no_eviction;
      *store_specs.mutable_no_eviction() = no_eviction;
    } else {
      LineageLengthEviction lineage_length_eviction;
      lineage_length_eviction.set_lineage_length(lineage_length);
      *store_specs.mutable_lineage_length_eviction() = lineage_length_eviction;
    }
    server_entity.set_hostname("127.0.0.1");
    server_entity.set_port(6379);
    (*redis_db_store.mutable_model_store_specs()) = store_specs;
    (*redis_db_store.mutable_server_entity()) = server_entity;
    (*store_config.mutable_redis_db_store()) = redis_db_store;
  }

};

/**
 * Design a test case to insert 1 model for one learner.
 * **/
TEST_F(InMemoryModelStoreTest, InsertOneModelSingleLearnerInMemoryStore) {
  InMemoryModelStoreTest::ConfigModelStore(1);
  InsertOneModelSingleLearner(store_config);
}

TEST_F(RedisModelStoreTest, InsertOneModelSingleLearnerRedis) {
  RedisModelStoreTest::ConfigModelStore(1);
  InsertOneModelSingleLearner(store_config);
}

/**
 * Design a test case to insert 3 models for the same learner.
 * **/
TEST_F(InMemoryModelStoreTest, InsertThreeModelsSingleLearnerInMemoryStore) {
  InMemoryModelStoreTest::ConfigModelStore(3);
  InsertThreeModelsSingleLearner(store_config);
}

TEST_F(RedisModelStoreTest, InsertThreeModelsSingleLearnerRedis) {
  RedisModelStoreTest::ConfigModelStore(3);
  InsertThreeModelsSingleLearner(store_config);
}

/**
 * Design a test case to insert exactly 3 models, one for each learner.
 * **/
TEST_F(InMemoryModelStoreTest, InsertThreeModelsMultipleLearnersInMemoryStore) {
  InMemoryModelStoreTest::ConfigModelStore(3);
  InsertThreeModelsMultipleLearners(store_config);
}

TEST_F(RedisModelStoreTest, InsertThreeModelsMultipleLearnersRedis) {
  RedisModelStoreTest::ConfigModelStore(3);
  InsertThreeModelsMultipleLearners(store_config);
}

/**
 * Design a test case to ask for more models than just the single one inserted.
 * **/
TEST_F(InMemoryModelStoreTest, RequestMoreModelsThanInsertedSingleLearnerInMemoryStore) {
  InMemoryModelStoreTest::ConfigModelStore(3);
  RequestMoreModelsThanInsertedSingleLearner(store_config);
}

TEST_F(RedisModelStoreTest, RequestMoreModelsThanInsertedStoreSingleLearnerRedis) {
  RedisModelStoreTest::ConfigModelStore(3);
  RequestMoreModelsThanInsertedSingleLearner(store_config);
}

/**
 * Design a test case to ask for more models than the ones already inserted.
 * **/
TEST_F(InMemoryModelStoreTest, RequestMoreModelsThanInsertedStoreMultipleLearnersInMemoryStore) {
  InMemoryModelStoreTest::ConfigModelStore(3);
  RequestMoreModelsThanInsertedMultipleLearners(store_config);
}

TEST_F(RedisModelStoreTest, RequestMoreModelsThanInsertedStoreMultipleLearnersRedis) {
  RedisModelStoreTest::ConfigModelStore(3);
  RequestMoreModelsThanInsertedMultipleLearners(store_config);
}

/**
 * Design a test case to test the NO eviction policy.
 * **/
TEST_F(InMemoryModelStoreTest, TestEvictionConstructorNoEvictionInMemoryStore) {
  NoEviction no_eviction;
  *store_specs.mutable_no_eviction() = no_eviction;
  (*in_memory_store.mutable_model_store_specs()) = store_specs;
  (*store_config.mutable_in_memory_store()) = in_memory_store;

  // Check pre-configured object.

  InitModelStore(store_config);

  EXPECT_EQ(model_store->GetConfiguredLineageLength(), -1);
  TestNoEvictionConstructorSettings(10);
}

TEST_F(RedisModelStoreTest, TestEvictionConstructorNoEvictionRedis) {
  NoEviction no_eviction;
  *store_specs.mutable_no_eviction() = no_eviction;
  server_entity.set_hostname("127.0.0.1");
  server_entity.set_port(6379);

  (*redis_db_store.mutable_model_store_specs()) = store_specs;
  (*redis_db_store.mutable_server_entity()) = server_entity;
  (*store_config.mutable_redis_db_store()) = redis_db_store;
  InitModelStore(store_config);
  EXPECT_EQ(model_store->GetConfiguredLineageLength(), -1);
  TestNoEvictionConstructorSettings(10);
}

/**
 * Design a test case to test the NON definition of the last k models eviction policy.
 * **/
TEST_F(InMemoryModelStoreTest, TestEvictionConstructorLastKModelsNotDefinedInMemoryStore) {
  LineageLengthEviction lineage_length_eviction;
  *store_specs.mutable_lineage_length_eviction() = lineage_length_eviction;
  (*in_memory_store.mutable_model_store_specs()) = store_specs;
  (*store_config.mutable_in_memory_store()) = in_memory_store;
  InitModelStore(store_config);
  TestEvictionConstructorSettings(1);
}

TEST_F(RedisModelStoreTest, TestEvictionConstructorLastKModelsNotDefinedRedis) {
  // Populate the configuration for the RedisStore.
  LineageLengthEviction lineage_length_eviction;
  *store_specs.mutable_lineage_length_eviction() = lineage_length_eviction;
  server_entity.set_hostname("127.0.0.1");
  server_entity.set_port(6379);
  (*redis_db_store.mutable_model_store_specs()) = store_specs;
  (*redis_db_store.mutable_server_entity()) = server_entity;
  (*store_config.mutable_redis_db_store()) = redis_db_store;

  // Check pre-configured object.
  InitModelStore(store_config);
  TestEvictionConstructorSettings(1);
}

/**
 * Design a test case to test the definition of the last k models eviction policy.
 * **/
TEST_F(InMemoryModelStoreTest, TestEvictionConstructorLastKModelsDefinedInMemoryStore) {
  int max_lineage_length = 5;
  InMemoryModelStoreTest::ConfigModelStore(max_lineage_length);
  InitModelStore(store_config);
  TestEvictionConstructorSettings(max_lineage_length);
}

TEST_F(RedisModelStoreTest, TestEvictionConstructorLastKModelsDefinedRedis) {
  int max_lineage_length = 5;
  RedisModelStoreTest::ConfigModelStore(max_lineage_length);
  InitModelStore(store_config);
  TestEvictionConstructorSettings(max_lineage_length);
}

/**
 * Design a test case to get the number of inserted models in the model store.
 * **/
TEST_F(InMemoryModelStoreTest, TestCountOfModelsInsertedForOneLearnerInMemoryStore) {
  int max_lineage_length = 5;
  int count_of_models_to_insert = 3;
  InMemoryModelStoreTest::ConfigModelStore(max_lineage_length);
  TestCountOfModelsInserted(store_config, count_of_models_to_insert);
}

TEST_F(RedisModelStoreTest, TestCountOfModelsInsertedForOneLearnerRedis) {
  int max_lineage_length = 5;
  int count_of_models_to_insert = 3;
  RedisModelStoreTest::ConfigModelStore(max_lineage_length);
  TestCountOfModelsInserted(store_config, count_of_models_to_insert);
}

} // namespace
} // namespace projectmetis::controller
