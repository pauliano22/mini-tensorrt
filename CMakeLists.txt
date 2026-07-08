cmake_minimum_required(VERSION 3.15)
project(MiniTensorRT CXX)

# Require modern C++
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Locate the Protobuf libraries installed on your system
find_package(Protobuf REQUIRED)

# Tell the compiler where to look for header files
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${PROTOBUF_INCLUDE_DIR})

# Define the final executable name and its source files (updated with ir.cpp)
add_executable(mini_tensorrt 
    src/main.cpp 
    src/ir.cpp 
    src/parser.cpp 
    src/optimizer.cpp
    src/backend.cpp   
    src/onnx.pb.cc
)

# Link the executable against the Protobuf library
target_link_libraries(mini_tensorrt ${PROTOBUF_LIBRARIES})
