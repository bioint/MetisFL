
#include "metisfl/controller/store/model_store.h"

#include "metisfl/proto/model.pb.h"

namespace metisfl::controller {

ModelStore::ModelStore(const int lineage_length) {
  // if 0 then it's equivalent to NoEviction
  m_lineage_length = lineage_length;
}
}  // namespace metisfl::controller
