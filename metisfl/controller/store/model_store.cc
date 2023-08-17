
#include "metisfl/controller/store/model_store.h"
#include "metisfl/proto/metis.pb.h"
#include "metisfl/proto/model.pb.h"

namespace metisfl::controller
{

  ModelStore::ModelStore(const int lineage_length)
  {
    // For every model cache we need to configure the total number of
    // models that we need to save in the model store for every learner
    // For this reason, we always need to inspect
    // the eviction policy and accordingly set the size of the cache.

    // if 0 then it's equivalent to NoEviction
    m_lineage_length = lineage_length;
  }
}
