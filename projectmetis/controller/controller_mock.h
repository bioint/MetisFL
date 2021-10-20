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

#ifndef PROJECTMETIS_RC_PROJECTMETIS_CONTROLLER_CONTROLLER_MOCK_H_
#define PROJECTMETIS_RC_PROJECTMETIS_CONTROLLER_CONTROLLER_MOCK_H_

#include <gmock/gmock.h>

#include "projectmetis/controller/controller.h"

namespace projectmetis::controller {

class MockController : public Controller {
 public:
  MOCK_METHOD(ControllerParams &, GetParams, (), (const, override));
  MOCK_METHOD(std::vector<LearnerDescriptor>,
              GetLearners,
              (),
              (const, override));
  MOCK_METHOD(uint32_t, GetNumLearners, (), (const, override));
  MOCK_METHOD(absl::StatusOr<LearnerDescriptor>,
              AddLearner,
              (const ServerEntity &server_entity, const DatasetSpec &dataset_spec),
              (override));
  MOCK_METHOD(absl::Status,
              RemoveLearner,
              (const std::string &learner_id, const std::string &token),
              (override));
  MOCK_METHOD(absl::Status,
              LearnerCompletedTask,
              (const std::string &learner_id, const std::string &token, const CompletedLearningTask &task),
              (override));
};

} // namespace projectmetis::controller

#endif //PROJECTMETIS_RC_PROJECTMETIS_CONTROLLER_CONTROLLER_MOCK_H_
