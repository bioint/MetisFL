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
#ifndef PROJECTMETIS_RC_PROJECTMETIS_CONTROLLER_MODEL_SELECTION_SELECTOR_H_
#define PROJECTMETIS_RC_PROJECTMETIS_CONTROLLER_MODEL_SELECTION_SELECTOR_H_

#include "projectmetis/proto/metis.pb.h"

namespace projectmetis::controller {

// A selector picks the models that need to be considered during aggregation.
class Selector {
 public:
  virtual ~Selector() = default;

  // Returns the ids of the learners that need to be selected from the
  // collections of scheduled learners, which refers to learners that
  // will run the next training task and active learners, which refers to
  // learners that are currently part of the federation.
  virtual std::vector<std::string> Select(
      const std::vector<std::string> &scheduled_learners,
      const std::vector<LearnerDescriptor> &active_learners) = 0;

  virtual std::string name() = 0;
};

} // namespace projectmetis::controller


#endif //PROJECTMETIS_RC_PROJECTMETIS_CONTROLLER_MODEL_SELECTION_SELECTOR_H_
