
#include "metisfl/controller/store/model_store.h"
#include "metisfl/proto/metis.pb.h"
#include "metisfl/proto/model.pb.h"

namespace projectmetis::controller {

ModelStore::ModelStore(const projectmetis::ModelStoreSpecs &specs) {
  // For every model cache we need to configure the total number of
  // models that we need to save in the model store for every learner
  // For this reason, we always need to inspect
  // the eviction policy and accordingly set the size of the cache.
  if (specs.has_no_eviction()) {
    PLOG(INFO) << "The NO_EVICTION policy is set";
    // All models need to be saved, hence -1.
    *m_model_store_specs.mutable_no_eviction() = specs.no_eviction();
  } else if (specs.has_lineage_length_eviction()) {
    PLOG(INFO) << "The LAST_K_MODELS policy is set!";
    *m_model_store_specs.mutable_lineage_length_eviction() = specs.lineage_length_eviction();
    if (specs.lineage_length_eviction().lineage_length() == 0) {
      PLOG(WARNING) << "The model_cache_size field is not defined, using size of 1.";
      m_model_store_specs.mutable_lineage_length_eviction()->set_lineage_length(1);
    }
  } else {
    PLOG(ERROR) << "Unknown model eviction policy.";
  }
}

}
