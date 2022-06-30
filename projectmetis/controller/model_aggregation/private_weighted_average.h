//
// MIT License
//
// Copyright (c) 2022 Project Metis
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
#ifndef PROJECTMETIS_RC_PROJECTMETIS_CONTROLLER_MODEL_AGGREGATION_PRIVATE_WEIGHTED_AVERAGE_H_
#define PROJECTMETIS_RC_PROJECTMETIS_CONTROLLER_MODEL_AGGREGATION_PRIVATE_WEIGHTED_AVERAGE_H_

#include "encryption/palisade/fhe/fhe_helper.h"

#include "projectmetis/controller/model_aggregation/aggregation_function.h"
#include "projectmetis/proto/model.pb.h"
#include "projectmetis/proto/metis.pb.h"

namespace projectmetis::controller {

class PWA : public AggregationFunction {
 public:
  explicit PWA(const FHEScheme &fhe_scheme) :
  fhe_scheme_(fhe_scheme),
  fhe_helper_(fhe_scheme.name(), fhe_scheme.batch_size(), fhe_scheme.scaling_bits()) {
    fhe_helper_.load_crypto_params();
  }
  explicit PWA(FHEScheme &&fhe_scheme) :
  fhe_scheme_(std::move(fhe_scheme)),
  fhe_helper_(fhe_scheme.name(), fhe_scheme.batch_size(), fhe_scheme.scaling_bits()) {
    fhe_helper_.load_crypto_params();
  }
  FederatedModel Aggregate(std::vector<std::pair<const Model*, double>>& pairs) override;

  inline std::string name() override {
    return "PWA";
  }

 private:
  FHEScheme fhe_scheme_;
  FHE_Helper fhe_helper_;
};

} // namespace projectmetis::controller

#endif //PROJECTMETIS_RC_PROJECTMETIS_CONTROLLER_MODEL_AGGREGATION_PRIVATE_WEIGHTED_AVERAGE_H_
