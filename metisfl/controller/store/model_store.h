
#ifndef METISFL_METISFL_CONTROLLER_STORE_MODEL_STORE_H_
#define METISFL_METISFL_CONTROLLER_STORE_MODEL_STORE_H_

#include <glog/logging.h>
#include <vector>
#include <map>

#include "metisfl/proto/model.pb.h"

namespace metisfl::controller
{

  class ModelStore
  {

  public:
    ModelStore() = default;
    explicit ModelStore(const int lineage_length);
    virtual ~ModelStore() = default;

    // Delete everything. Clean-up.
    virtual void Expunge() = 0;

    // Delete all models from the cache for a given list of learners.
    virtual void EraseModels(const std::vector<std::string> &learner_ids) = 0;

    // For every learner, model pair insert the model inside the model cache.
    // We pass the models as a copy because when this function is called
    // from the controller the task.model() object is destroyed.
    // *** CAUTION ***
    // The convention is that multiple learners can insert a single model.
    virtual void InsertModel(std::vector<std::pair<std::string, Model>> learner_pairs) = 0;

    // Remove the models from ephermal state of the model store only.
    virtual void ResetState() = 0;

    // Select a number of models (int value) for each learner and return a map
    // where key is the learner id and value the learner's model collection.
    // We return pointers to avoid duplicating the returned models. SelectModels()
    // function brings the models inside the controller's allocated memory, and
    // therefore we want to avoid redundant copying because of reference.
    // *** CAUTION ***
    // The convention we follow in the select model function is to
    // return models in the ascending committed (time) order:
    //    <older committed model> to <late committed model>
    virtual std::map<std::string, std::vector<const Model *>>
    SelectModels(std::vector<std::pair<std::string, int>> learner_pairs) = 0;

    // Proper release of resources and model store shutdown.
    virtual void Shutdown() = 0;

    // Name of model cache.
    virtual std::string Name() = 0;

    // Number of models saved in model_store type
    // based on the configured eviction policy.
    virtual int GetConfiguredLineageLength() = 0;

    // Returns the count of models inserted for each learner.
    virtual int GetLearnerLineageLength(std::string learner_id) = 0;

  protected:
    int m_lineage_length;

    // Keep track of the models that have been part of the model_store.
    std::map<std::string, std::vector<Model>> m_model_store_cache;
  };

}

#endif // METISFL_METISFL_CONTROLLER_STORE_MODEL_STORE_H_
