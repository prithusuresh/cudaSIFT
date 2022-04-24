cmake_minimum_required(VERSION 3.1)
PROJECT(cudaSift LANGUAGES C CXX CUDA)

#set(CMAKE_CXX_STANDARD 11)
#find_package(OpenCV REQUIRED)
#include_directories(include src ${OpenCV_INCLUDE_DIRS})
#message(STATUS "OpenCV library status:")
#message(STATUS "    version: ${OpenCV_VERSION}")
#message(STATUS "    libraries: ${OpenCV_LIBS}")
#message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
#
#file(GLOB all_SRCS "include/*" "src/*")
#add_executable(cudaSift ${all_SRCS})
#set_target_properties(cudaSift PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
#target_link_libraries(cudaSift ${OpenCV_LIBS})

set(cudaSift_VERSION_MAJOR 2)
set(cudaSift_VERSION_MINOR 0)
set(cudaSift_VERSION_PATCH 0)

set(CPACK_PACKAGE_VERSION_MAJOR "${cudaSift_VERSION_MAJOR}")
set(CPACK_PACKAGE_VERSION_MINOR "${cudaSift_VERSION_MINOR}")
set(CPACK_PACKAGE_VERSION_PATCH "${cudaSift_VERSION_PATCH}")
set(CPACK_GENERATOR "ZIP")
include(CPack)

find_package(OpenCV REQUIRED)
find_package(CUDA)
if (NOT CUDA_FOUND)
  message(STATUS "CUDA not found. Project will not be built.")
endif(NOT CUDA_FOUND)

if (WIN32)
  set(EXTRA_CXX_FLAGS "/DVERBOSE /D_CRT_SECURE_NO_WARNINGS ")
  list(APPEND CUDA_NVCC_FLAGS "--compiler-options;-O2;-DVERBOSE")
endif()
if (UNIX)
  if (APPLE)
    set(EXTRA_CXX_FLAGS "-DVERBOSE -march=native")
    list(APPEND CUDA_NVCC_FLAGS "--compiler-options;-O2;-DVERBOSE")
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -march=native")
    list(APPEND CUDA_NVCC_FLAGS "-lineinfo;--compiler-options;-O2;-D_FORCE_INLINES;-DVERBOSE_NOT")
  endif()
endif()

set(cuda_sources
  src/cudaImage.cu
  include/cudaImage.h
  src/cudaSiftH.cu
  include/cudaSiftH.h
  src/matching.cu
  include/cudaSiftD.h
  include/cudaSift.h
  include/cudautils.h
)

set(sources
  src/geomFuncs.cpp
  src/mainSift.cpp
)

include_directories(include src ${OpenCV_INCLUDE_DIRS})

#SET(CUDA_SEPARABLE_COMPILATION ON)

cuda_add_executable(cudasift ${cuda_sources} ${sources} OPTIONS)


set_target_properties(cudasift PROPERTIES
  COMPILE_FLAGS "${EXTRA_CXX_FLAGS}"
)

target_link_libraries(cudasift ${CUDA_cudadevrt_LIBRARY} ${OpenCV_LIBS})
#  /usr/local/cuda/lib64/libcudadevrt.a ${OpenCV_LIBS}
#)

install(FILES
  ${cuda_sources}
  ${sources}
  src/cudaSiftD.cu
  CMakeLists.txt
  Copyright.txt
  DESTINATION .
)
install(FILES data/left.pgm data/righ.pgm
  DESTINATION data
)

