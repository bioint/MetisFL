// MIT License
//
// Copyright (c) 2021 Project Metis
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "projectmetis/controller/model_aggregation/federated_average.h"

#include "projectmetis/proto/model.pb.h"

namespace projectmetis::controller {
namespace {

// TODO (canast02) Move this under the util package.
template<typename Tensor>
void ScaleTensor(const Tensor &tensor, double scale, Tensor *scaled) {
  *scaled->mutable_spec() = tensor.spec();
  scaled->clear_values();
  for (const auto &value : tensor.values()) {
    scaled->add_values(scale * value);
  }
}

}

FederatedModel
FederatedAverage::Aggregate(std::vector<std::pair<const Model*, double>> pairs) {
  double z = 0;
  for (const auto &pair : pairs) {
    z += pair.second;
  }

  FederatedModel community_model;
  for (const auto &pair : pairs) {
    for (const auto &variable : pair.first->variables()) {
      auto contrib_value = pair.second / z;
      auto scaled_variable = community_model.mutable_model()->add_variables();
      scaled_variable->set_name(variable.name());
      scaled_variable->set_trainable(variable.trainable());

      if (variable.has_double_tensor()) {
        ScaleTensor(variable.double_tensor(),
                    contrib_value,
                    scaled_variable->mutable_double_tensor());
      } else if (variable.has_int_tensor()) {
        ScaleTensor(variable.int_tensor(),
                    contrib_value,
                    scaled_variable->mutable_int_tensor());
      } else {
        throw std::runtime_error("unsupported variable type");
      }
    }
  }

  community_model.set_num_contributors(pairs.size());
  return community_model;
}

} // namespace projectmetis::controller