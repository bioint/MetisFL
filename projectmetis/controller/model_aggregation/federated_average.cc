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

//// TODO (canast02) Move this under the util package.
//template<typename Tensor>
//void ScaleTensor(const Tensor &tensor, double scale, Tensor *scaled) {
//  *scaled->mutable_spec() = tensor.spec();
//  scaled->clear_values();
//  for (const auto &value : tensor.values()) {
//    scaled->add_values(scale * value);
//  }
//}

template<typename Tensor>
void AddScaledValues(const Tensor &tensor,
                     double scaling_factor,
                     Tensor *scaled) {
  for (int i = 0; i < tensor.values_size(); ++i) {
    auto unscaled = tensor.values(i);

    auto current = scaled->values(i);
    scaled->set_values(i, current + scaling_factor * unscaled);
  }
}

}

FederatedModel
FederatedAverage::Aggregate(
    std::vector<std::pair<const Model*, double>>& pairs) {
  double z = 0;
  for (const auto &pair : pairs) {
    z += pair.second;
  }

  // Initializes the community model.
  FederatedModel community_model;
  const auto& sample_model = pairs.front().first;
  for (const auto& sample_variable: sample_model->variables()) {
    auto* variable = community_model.mutable_model()->add_variables();
    variable->set_name(sample_variable.name());
    variable->set_trainable(sample_variable.trainable());
    if (sample_variable.has_int_tensor()) {
      *variable->mutable_int_tensor()->mutable_spec() =
          sample_variable.int_tensor().spec();
      for (int j = 0; j < sample_variable.int_tensor().values_size(); ++j) {
        variable->mutable_int_tensor()->add_values(0);
      }
    } else if (sample_variable.has_float_tensor()) {
      *variable->mutable_float_tensor()->mutable_spec() =
          sample_variable.float_tensor().spec();
      for (int j = 0; j < sample_variable.float_tensor().values_size(); ++j) {
        variable->mutable_float_tensor()->add_values(0.0f);
      }
    } else if (sample_variable.has_double_tensor()) {
      *variable->mutable_double_tensor()->mutable_spec() =
          sample_variable.double_tensor().spec();
      for (int j = 0; j < sample_variable.double_tensor().values_size(); ++j) {
        variable->mutable_double_tensor()->add_values(0.0);
      }
    } else {
      // TODO(canast02) Need to catch the error or exit entirely.
      throw std::runtime_error("unsupported variable type");
    }
  }

  // TODO(dstripelis) We could add support to aggregate only the trainable
  //  weights. For now, we aggregate all matrices, but if we aggregate only the
  //  trainable, then what should be the value of the non-trainable weights?
  // Aggregates the input models.
  for (const auto &pair : pairs) {
    const auto* model = pair.first;
    const double scale = pair.second;
    for (int i = 0; i < model->variables_size(); ++i) {
      const auto& variable = model->variables(i);

      auto contrib_value = scale / z;
      auto community_variable =
          community_model.mutable_model()->mutable_variables(i);

      if (variable.has_int_tensor()) {
        AddScaledValues(variable.int_tensor(),
                        contrib_value,
                        community_variable->mutable_int_tensor());
      } else if (variable.has_float_tensor()) {
        AddScaledValues(variable.float_tensor(),
                        contrib_value,
                        community_variable->mutable_float_tensor());
      } else if (variable.has_double_tensor()) {
        AddScaledValues(variable.double_tensor(),
                        contrib_value,
                        community_variable->mutable_double_tensor());
      } else {
        throw std::runtime_error("unsupported variable type");
      }
    }
  }

  // Sets the number of contributors to the number of input models.
  community_model.set_num_contributors(pairs.size());
  return community_model;
}

} // namespace projectmetis::controller
