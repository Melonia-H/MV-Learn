cmake_minimum_required(VERSION 2.8)
project(testTensor)

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -std=c++11 -W")

#------------------------------ C++11 support------------------------------
#include(CheckCXXCompilerFlag)
#CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
#CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
#if (COMPILER_SUPPORTS_CXX11)
#    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
#elseif (COMPILER_SUPPORTS_CXX0X)
#    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
#else ()
#    message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
#endif ()
add_definitions(-std=c++14)

set(TENSORFLOW_INCLUDES
        /usr/local/include/tf/
        /usr/local/include/tf/bazel-genfiles
        /usr/local/include/tf/tensorflow/
        /usr/local/include/tf/tensorflow/contrib/makefile/gen/protobuf/include
        /usr/local/include/tf/third_party
        /usr/local/include/eigen3/
        /usr/local/include/tf/tensorflow/absl/
        /usr/local/include/tf/tensorflow/contrib/makefile/downloads/farmhash/dev/
        /usr/local/include/tf/tensorflow/python/lib/core/
)

set(TENSORFLOW_LIBS
    /usr/local/lib/libtensorflow_cc.so
    /usr/local/lib/libtensorflow_framework.so.1.13.1
    /usr/local/include/tf/tensorflow/contrib/makefile/gen/protobuf/lib/libprotobuf.so
    /usr/local/include/tf/tensorflow/contrib/makefile/gen/protobuf/lib/libprotoc.so)

include_directories(${TENSORFLOW_INCLUDES})

#------------------------opencv------------------------------
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

##------------------------python------------------------------
find_package(PythonLibs 3.5 REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})

##------------------------numpy------------------------------
set(NUMPY_INCLUDE_PATH /usr/local/lib/python3.5/dist-packages/numpy/core/include)
include_directories(${NUMPY_INCLUDE_PATH})

FILE(GLOB SRC_FILES "./*.cpp"  "./*.cc")
FILE(GLOB HEAD_FILES "./*.h" "./*.hpp")

# add library
add_executable(${PROJECT_NAME}  ${SRC_FILES} ${HEAD_FILES})
#add_executable(${PROJECT_NAME}  "test02.cpp")


target_include_directories(${PROJECT_NAME} PUBLIC ${PYTHON_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${TENSORFLOW_LIBS}  ${OpenCV_LIBS} ${PYTHON_LIBRARIES})








