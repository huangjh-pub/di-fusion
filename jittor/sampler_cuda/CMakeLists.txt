project(Sampler LANGUAGES CXX CUDA)
cmake_minimum_required(VERSION 3.8)

find_package(CLI11 CONFIG REQUIRED)
find_package(Eigen3 3.3 REQUIRED)
find_package(Pangolin REQUIRED)
find_package(flann REQUIRED)

add_executable(PreprocessMeshCUDA PreprocessMesh.cu ShaderProgram.cpp Utils.cu)
target_link_libraries(PreprocessMeshCUDA PRIVATE Eigen3::Eigen CLI11::CLI11 pangolin flann_cuda -lcurand)
target_compile_features(PreprocessMeshCUDA PRIVATE cxx_std_14)
set_target_properties(PreprocessMeshCUDA PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin")
target_compile_options(PreprocessMeshCUDA PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>: --use_fast_math>
  $<$<COMPILE_LANGUAGE:CXX>: -fPIC -O3 -march=native >
)
set_target_properties(PreprocessMeshCUDA PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
