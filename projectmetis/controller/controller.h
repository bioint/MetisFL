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

#ifndef PROJECTMETIS_RC_PROJECTMETIS_CONTROLLER_CONTROLLER_H_
#define PROJECTMETIS_RC_PROJECTMETIS_CONTROLLER_CONTROLLER_H_

#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "projectmetis/proto/controller.grpc.pb.h"

namespace projectmetis::controller {

// TODO(stripeli): Add a nice description about what the controller is about :)
class Controller {
 public:
  virtual ~Controller() = default;

  // Returns the parameters with which the controller was initialized.
  ABSL_MUST_USE_RESULT
  virtual const ControllerParams &GetParams() const = 0;

  // Returns the list of all the active learners.
  ABSL_MUST_USE_RESULT
  virtual std::vector<LearnerDescriptor> GetLearners() const = 0;

  // Returns the number of active learners.
  ABSL_MUST_USE_RESULT
  virtual uint32_t GetNumLearners() const = 0;

  // Attempts to add a new learner to the federation. If successful, a
  // LearnerState instance is returned. Otherwise, it returns null.
  virtual absl::StatusOr<LearnerDescriptor>
  AddLearner(const ServerEntity &server_entity,
             const DatasetSpec &dataset_spec) = 0;

  // Removes a learner from the federation. If successful it returns OK.
  // Otherwise, it returns an error.
  virtual absl::Status RemoveLearner(const std::string &learner_id,
                                     const std::string &token) = 0;

  virtual absl::Status LearnerCompletedTask(const std::string &learner_id,
                                            const std::string &token,
                                            const CompletedLearningTask &task) = 0;

  virtual std::vector<ModelEvaluation>
  GetEvaluationLineage(const std::string &learner_id, uint32_t num_steps) = 0;

public:
  // Creates a new controller using the default implementation, i.e., in-memory.
  static std::unique_ptr<Controller> New(const ControllerParams &params);
};

} // namespace projectmetis::controller

#endif // PROJECTMETIS_RC_PROJECTMETIS_CONTROLLER_CONTROLLER_H_
