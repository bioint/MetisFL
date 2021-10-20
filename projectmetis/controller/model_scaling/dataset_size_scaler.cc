//
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

#include "projectmetis/controller/model_scaling/dataset_size_scaler.h"

namespace projectmetis::controller {

absl::flat_hash_map<std::string,
                    double> DatasetSizeScaler::ComputeScalingFactors(
    const FederatedModel &community_model,
    const absl::flat_hash_map<std::string, LearnerState> &states) {
  long dataset_size = 0;
  for (const auto&[_, state] : states) {
    dataset_size += state.learner().dataset_spec().num_validation_examples();
  }

  absl::flat_hash_map<std::string, double> scaling_factors;
  for (const auto&[learner_id, state] : states) {
    long
        num_examples = state.learner().dataset_spec().num_validation_examples();
    double scaling_factor =
        static_cast<double>(num_examples) / static_cast<double>(dataset_size);
    scaling_factors[learner_id] = scaling_factor;
  }

  return scaling_factors;
}

} // namespace projectmetis::controller
