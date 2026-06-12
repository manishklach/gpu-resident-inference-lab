cmake_minimum_required(VERSION 3.18)
project(xlpk_cuda_smoke LANGUAGES CUDA CXX)

# Require CUDA
find_package(CUDAToolkit REQUIRED)

# Set CUDA architecture (adjust for your GPU)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "70;80;86")
endif()

# Compiler flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CXX_STANDARD 17)

# Include directories (headers + .cuh stage helpers)
set(INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Sources: the mega-kernel, the baseline comparison kernel, and the host launcher.
# The stage helpers (.cuh) are included by the mega-kernel at compile time;
# they are NOT compiled separately.
add_executable(xlpk_cuda_smoke
    ${CMAKE_CURRENT_SOURCE_DIR}/src/xl_persistent_megakernel.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/baseline_host_decode_kernel.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/resident_scheduler_kernel.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/kv_prefetch_planner_kernel.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/dma_aware_kv_movement_planner_kernel.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/tiered_kv_staging_kernel.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/kv_pressure_eviction_kernel.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/kv_tier_residency_kernel.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/trace_replay_admission_kernel.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/compacted_decode_kernel.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/resident_sparse_decode_pipeline_kernel.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/sparse_kv_gather_kernel.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/verify_commit_kernel.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/host_launcher.cpp
)

set_source_files_properties(
    ${CMAKE_CURRENT_SOURCE_DIR}/src/host_launcher.cpp
    PROPERTIES LANGUAGE CUDA
)

target_include_directories(xlpk_cuda_smoke PRIVATE ${INCLUDE_DIRS})

target_link_libraries(xlpk_cuda_smoke CUDA::cudart)

install(TARGETS xlpk_cuda_smoke DESTINATION bin)
