############################################################
# CMake setup
############################################################
cmake_minimum_required(VERSION 3.17.5)
project(projectmetis-rc)

set(CMAKE_CXX_STANDARD 17)

# To suppress warning: "Policy CMP0042 is not set: MACOSX_RPATH is enabled by default."
set(CMAKE_MACOSX_RPATH 1)

option(USE_SYSTEM_GRPC "Use system installed gRPC" OFF)
option(CMAKE_CROSSCOMPILING "Cross platform compilation" OFF)

if(NOT USE_SYSTEM_GRPC)
    # Build gRPC using FetchContent.
    message(STATUS "Using gRPC via add_subdirectory (FetchContent).")
    include(FetchContent)
    # gRPC building guide: https://github.com/grpc/grpc/blob/v1.35.0/src/cpp/README.md#fetchcontent
    FetchContent_Declare(
            gRPC
            GIT_REPOSITORY https://github.com/grpc/grpc
            # when using gRPC, you will actually set this to an existing tag, such as
            # v1.25.0, v1.26.0 etc..
            GIT_TAG        v1.36.2)
    set(FETCHCONTENT_QUIET OFF)
    FetchContent_MakeAvailable(grpc)
    # Since FetchContent uses add_subdirectory under the hood, we can use
    # the grpc targets directly from this build.
    set(_PROTOBUF_LIBPROTOBUF libprotobuf)
    set(_REFLECTION grpc++_reflection)
    set(_PROTOBUF_PROTOC $<TARGET_FILE:protoc>)
    set(_GRPC_GRPCPP grpc++)
else()
    # This branch assumes that gRPC and all its dependencies are already installed
    # on this system, so they can be located by find_package().
    message(STATUS "Using existing gRPC installation.")

    # Find Protobuf installation.
    # Looks for protobuf-config.cmake file installed by Protobuf's cmake installation.
    set(protobuf_MODULE_COMPATIBLE TRUE)
    find_package(Protobuf CONFIG REQUIRED)
    message(STATUS "Using protobuf ${Protobuf_VERSION}")

    set(_PROTOBUF_LIBPROTOBUF protobuf::libprotobuf)
    set(_REFLECTION gRPC::grpc++_reflection)
    if(CMAKE_CROSSCOMPILING)
        find_program(_PROTOBUF_PROTOC protoc)
    else()
        set(_PROTOBUF_PROTOC $<TARGET_FILE:protobuf::protoc>)
    endif()

    # Find gRPC installation
    # Looks for gRPCConfig.cmake file installed by gRPC's cmake installation.
    find_package(gRPC CONFIG REQUIRED)
    message(STATUS "Using gRPC ${gRPC_VERSION}")

    set(_GRPC_GRPCPP gRPC::grpc++)
    if(CMAKE_CROSSCOMPILING)
        find_program(_GRPC_CPP_PLUGIN_EXECUTABLE grpc_cpp_plugin)
    else()
        set(_GRPC_CPP_PLUGIN_EXECUTABLE $<TARGET_FILE:gRPC::grpc_cpp_plugin>)
    endif()
endif()

add_executable(projectmetis-rc projectmetis/controller/main.cpp)
target_link_libraries(projectmetis-rc grpc++)