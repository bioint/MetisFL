//
// Created by Chrysovalantis Anastasiou on 3/10/21.
//

#ifndef PROJECTMETIS_RC_PROJECTMETIS_CONTROLLER_CONTROLLER_SERVICE_H_
#define PROJECTMETIS_RC_PROJECTMETIS_CONTROLLER_CONTROLLER_SERVICE_H_

#include "projectmetis/proto/controller.grpc.pb.h"

namespace projectmetis::controller {

std::unique_ptr<Controller::Service> New();

}  // namespace projectmetis::controller

#endif //PROJECTMETIS_RC_PROJECTMETIS_CONTROLLER_CONTROLLER_SERVICE_H_
