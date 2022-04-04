cmake_minimum_required(VERSION 3.1)
set(CMAKE_VERBOSE_MAKEFILE ON)
PROJECT(cudaSift LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 11)
find_package(OpenCV REQUIRED)
include_directories(include src ${OpenCV_INCLUDE_DIRS})
file(GLOB all_SRCS "${PROJECT_SOURCE_DIR}/include/*" "${PROJECT_SOURCE_DIR}/src/*")
add_executable(cudaSift ${all_SRCS})
set_target_properties( cudaSift
        PROPERTIES CUDA_SEPARABLE_COMPILATION ON
                      )
target_link_libraries(cudaSift ${OpenCV_LIBS})
