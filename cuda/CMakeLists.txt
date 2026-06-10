cmake_minimum_required(VERSION 3.18)
project(XLPersistentKernelCUDA LANGUAGES CUDA CXX)

# Require CUDA
find_package(CUDAToolkit REQUIRED)

# Set CUDA architecture (adjust for your GPU)
# Common options: sm_70 (V100), sm_80 (A100), sm_86 (RTX 3090), sm_89 (RTX 4090)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "70;80;86")
endif()

# Compiler flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CXX_STANDARD 17)

# Build the persistent decode stub
add_executable(persistent_decode_stub
    persistent_decode_stub.cu
)

target_include_directories(persistent_decode_stub PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Link CUDA runtime
target_link_libraries(persistent_decode_stub
    CUDA::cudart
)

# Install target (optional)
install(TARGETS persistent_decode_stub DESTINATION bin)
