#include <iostream>

//#include <grpc/grpc.h>
//#include <grpcpp/server.h>
//#include <grpcpp/server_builder.h>
//#include <grpcpp/server_context.h>
//#include <grpcpp/security/server_credentials.h>
#include "projectmetis/controller/protos/controller.grpc.pb.h"
//#include "projectmetis/controller/protos/controller.pb.h"

int main()
{
    projectmetis::controller::GetCommunityModelLineageRequest request;
    request.set_num_backtracks(5);

    std::cout << "Hello, World!" << std::endl;
    std::cout << request.DebugString() << std::endl;
    return 0;
}
