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

#ifndef PROJECTMETIS_RC_PROJECTMETIS_CONTROLLER_CONTROLLER_SERVICER_H_
#define PROJECTMETIS_RC_PROJECTMETIS_CONTROLLER_CONTROLLER_SERVICER_H_

#include <memory>
#include <sstream>
#include <fstream>
#include <filesystem>

#include "projectmetis/controller/controller.h"
#include "projectmetis/proto/controller.grpc.pb.h"

namespace projectmetis::controller {

class ControllerServicer : public ControllerService::Service {
public:
  ABSL_MUST_USE_RESULT
  virtual const Controller *GetController() const = 0;

  // Starts the gRPC service.
  virtual void StartService() = 0;

  // Waits for the gRPC service to shut down.
  virtual void WaitService() = 0;

  // Stops the gRPC service.
  virtual void StopService() = 0;

public:
  static std::unique_ptr<ControllerServicer> New(Controller *controller);
};

} // namespace projectmetis::controller

#endif // PROJECTMETIS_RC_PROJECTMETIS_CONTROLLER_CONTROLLER_SERVICER_H_
