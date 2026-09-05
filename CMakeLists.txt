cmake_minimum_required(VERSION 3.24)
project(llama_cuda_runtime LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ---------------------------------------------------------------------------
# nlohmann/json, the one allowed third-party library. Pulled as a single header
# at configure time rather than vendored so the repo stays small.
# ---------------------------------------------------------------------------
set(LCR_JSON_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/third_party/nlohmann/json.hpp)
if(NOT EXISTS ${LCR_JSON_HEADER})
  message(STATUS "Downloading nlohmann/json v3.11.3 single header")
  file(DOWNLOAD
    https://github.com/nlohmann/json/releases/download/v3.11.3/json.hpp
    ${LCR_JSON_HEADER}
    EXPECTED_HASH SHA256=9bea4c8066ef4a1c206b2be5a36302f8926f7fdc6087af5d20b417d0cf103ea6
    TLS_VERIFY ON
    STATUS LCR_JSON_DOWNLOAD_STATUS)
  list(GET LCR_JSON_DOWNLOAD_STATUS 0 LCR_JSON_DOWNLOAD_CODE)
  if(NOT LCR_JSON_DOWNLOAD_CODE EQUAL 0)
    file(REMOVE ${LCR_JSON_HEADER})
    message(FATAL_ERROR "json.hpp download failed: ${LCR_JSON_DOWNLOAD_STATUS}")
  endif()
endif()

add_library(lcr_json INTERFACE)
target_include_directories(lcr_json INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/third_party)

# ---------------------------------------------------------------------------
# Host library. Everything here is plain C++17 and builds on a machine with no
# NVIDIA hardware, which is what lets the loader and tokenizer be developed and
# tested away from the GPU box.
# ---------------------------------------------------------------------------
add_library(lcr_host STATIC
  src/common.cpp
  src/config.cpp
  src/safetensors.cpp
  src/tokenizer.cpp
  src/unicode_data.cpp
)
target_include_directories(lcr_host PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(lcr_host PUBLIC lcr_json)
target_compile_options(lcr_host PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wno-unused-parameter>)

add_executable(lcr-tokenize tools/tokenize_main.cpp)
target_link_libraries(lcr-tokenize PRIVATE lcr_host)

add_executable(lcr-inspect tools/inspect_main.cpp)
target_link_libraries(lcr-inspect PRIVATE lcr_host)

# ---------------------------------------------------------------------------
# CUDA half of the build. Optional: without nvcc the host targets still build.
# ---------------------------------------------------------------------------
include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
    # Ampere through Hopper. Override with -DCMAKE_CUDA_ARCHITECTURES=89 for a
    # single-target build, which is several times faster to compile.
    set(CMAKE_CUDA_ARCHITECTURES 80 86 89 90)
  endif()
  find_package(CUDAToolkit REQUIRED)

  # bf16 by default, matching the Llama 3.2 checkpoint. fp16 switches the whole
  # pipeline; it is one typedef, not a second code path.
  set(LCR_DTYPE "bf16" CACHE STRING "Element type for weights and activations")
  set_property(CACHE LCR_DTYPE PROPERTY STRINGS bf16 fp16)

  add_library(lcr_cuda STATIC
    src/arena.cu
    src/cuda_utils.cu
    src/kv_cache.cu
    src/model.cu
    src/kernels/elementwise.cu
    src/kernels/rmsnorm.cu
    src/kernels/rope.cu
    src/kernels/attention.cu
    src/kernels/sampling.cu
  )
  target_link_libraries(lcr_cuda PUBLIC lcr_host CUDA::cublas CUDA::cudart)
  if(LCR_DTYPE STREQUAL "fp16")
    target_compile_definitions(lcr_cuda PUBLIC LCR_ELEM_FP16)
  elseif(NOT LCR_DTYPE STREQUAL "bf16")
    message(FATAL_ERROR "LCR_DTYPE must be bf16 or fp16, got ${LCR_DTYPE}")
  endif()
  set_target_properties(lcr_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  target_compile_options(lcr_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -lineinfo>)

  add_executable(llama-run src/main.cu)
  target_link_libraries(llama-run PRIVATE lcr_cuda)

  add_executable(lcr-validate tools/validate_main.cu)
  target_link_libraries(lcr-validate PRIVATE lcr_cuda)

  add_executable(test_kernels tests/test_kernels.cu)
  target_include_directories(test_kernels PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
  target_link_libraries(test_kernels PRIVATE lcr_cuda)
  add_test(NAME test_kernels COMMAND test_kernels)

  message(STATUS "CUDA enabled, architectures: ${CMAKE_CUDA_ARCHITECTURES}")
else()
  message(STATUS "No CUDA compiler found. Building host targets only "
                 "(safetensors, config, tokenizer and their tests).")
endif()

# ---------------------------------------------------------------------------
# Tests. Plain assert-driven binaries registered with CTest, no test framework.
# ---------------------------------------------------------------------------
enable_testing()

function(lcr_add_host_test name)
  add_executable(${name} tests/${name}.cpp)
  target_link_libraries(${name} PRIVATE lcr_host)
  target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
  add_test(NAME ${name} COMMAND ${name})
endfunction()

lcr_add_host_test(test_safetensors)
lcr_add_host_test(test_config)
lcr_add_host_test(test_tokenizer)
