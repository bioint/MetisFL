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

#ifndef PROJECTMETIS_RC_PROJECTMETIS_CONTROLLER_CONTROLLER_UTILS_H_
#define PROJECTMETIS_RC_PROJECTMETIS_CONTROLLER_CONTROLLER_UTILS_H_

#include <string>

#include "absl/strings/str_cat.h"
#include "projectmetis/proto/controller.pb.h"

namespace projectmetis::controller {

// Generates a unique identifier for the provided learner entity. In the current
// implementation, the generated id is in the format of `<hostname>:<port>`.
inline std::string GenerateLearnerId(const ServerEntity &server_entity) {
  return absl::StrCat(server_entity.hostname(), ":", server_entity.port());
}

} // namespace projectmetis::controller

#endif // PROJECTMETIS_RC_PROJECTMETIS_CONTROLLER_CONTROLLER_UTILS_H_
