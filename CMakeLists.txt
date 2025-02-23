cmake_minimum_required(VERSION 3.20)

project(Langford LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)

add_library(grand_entiers  grands_entiers.cu)
set_target_properties(grand_entiers PROPERTIES CUDA_ARCHITECTURES "72;80")

add_executable(solver main.cu general.cu host_fonctions.cu langford_God_omp16.cu)
target_link_libraries(solver PRIVATE grand_entiers)
target_link_libraries(solver PRIVATE OpenMP::OpenMP_CXX)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_80 -rdc=true")

target_compile_options(solver PRIVATE -Wall -Wextra)

set_target_properties(solver PROPERTIES CUDA_ARCHITECTURES "72;80")
set_property(TARGET solver PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)


