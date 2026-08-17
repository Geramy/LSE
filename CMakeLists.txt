cmake_minimum_required(VERSION 3.24)

# GCC 16 ships P2996 as -std=c++26 -freflection. Pick it before project()
# so the first configure does not lock in g++-15.
if(NOT CMAKE_CXX_COMPILER)
  find_program(LSE_GXX16 NAMES g++-16)
  if(LSE_GXX16)
    set(CMAKE_CXX_COMPILER "${LSE_GXX16}")
  endif()
endif()

project(lse
  VERSION 0.1.0
  DESCRIPTION "Lemon Seed Engine — modular LLM training/inference engine on HRX"
  LANGUAGES CXX)

# Host is C++26 (graph, kir, runtime). Device source is HIP via comgr and
# stays at 20 so generated kernels do not depend on host-only language.
# Shared ABI headers (dtype, elem, quant layouts) stay plain data.
set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(LSE_DEVICE_CXX_STANDARD 20 CACHE STRING "C++ standard for HIP device code")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "" FORCE)
endif()

include(LSEOptions)
include(LSECompilerFlags)
include(LSERust)

# ---------------------------------------------------------------------------
# core — dtype/status/shape/quant. No device dependency; always builds.
# ---------------------------------------------------------------------------
add_library(lse_core
  src/core/debug.cpp
  src/core/dtype.cpp
  src/core/shape.cpp
  src/core/status.cpp
  src/quant/group_affine.cpp
)
add_library(lse::core ALIAS lse_core)

target_include_directories(lse_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_include_directories(lse_core SYSTEM PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/third_party/vendor>)
lse_apply_common_flags(lse_core)

# ---------------------------------------------------------------------------
# ir — the kernel IR: types, values, ops, structured regions, the affine index
# vocabulary, the optimization passes and the lowering to source. Sits below
# the backends, which record into it, and below lse_graph, which is a *tensor*
# level IR and a different thing.
# ---------------------------------------------------------------------------
add_library(lse_ir
  src/ir/body.cpp
  src/ir/index.cpp
  src/ir/lower.cpp
  src/ir/recorder.cpp
  src/ir/spell.cpp
  src/ir/verify.cpp
  src/ir/pass/cse.cpp
  src/ir/pass/dce.cpp
  src/ir/pass/lds_fold.cpp
  src/ir/pass/pipeline.cpp
)
add_library(lse::ir ALIAS lse_ir)
target_link_libraries(lse_ir PUBLIC lse_core)
lse_apply_common_flags(lse_ir)

# ---------------------------------------------------------------------------
# backends — each is an independent module under src/backends/<name>/.
# ---------------------------------------------------------------------------
add_subdirectory(src/backends)

# ---------------------------------------------------------------------------
# communication — the transport seam: listen on an endpoint or connect to one,
# then send and receive asynchronously on a loop the caller drives. Above
# nothing but core, because a peer that only moves bytes must not drag in the
# IR or a backend; whether an endpoint is a socket, a DMA ring or a queue pair
# is chosen from the endpoint string and the declared capabilities, never
# named by a caller.
# ---------------------------------------------------------------------------
add_library(lse_communication
  src/communication/endpoint.cpp
  src/communication/capabilities.cpp
  src/communication/transport.cpp
  src/communication/reactor.cpp
  src/communication/poller_epoll.cpp
  src/communication/stream_link.cpp
  src/communication/socket.cpp
  src/communication/rdma.cpp
)
add_library(lse::communication ALIAS lse_communication)
# ${CMAKE_DL_LIBS} is for dlopen, which is how a fabric transport asks this
# machine what it has. It is not a build dependency on any fabric: there is none.
target_link_libraries(lse_communication PUBLIC lse_core ${CMAKE_DL_LIBS})
# Transports self-register from a TU with no referenced symbols.
target_link_options(lse_communication INTERFACE
  "LINKER:--whole-archive,$<TARGET_FILE:lse_communication>,--no-whole-archive")
lse_apply_common_flags(lse_communication)

# ---------------------------------------------------------------------------
# dist — transports and collectives. Deliberately below lse_graph: a backend
# registers its device codec engine here, so this must not depend on lse_graph
# (which link-depends on the backends).
# ---------------------------------------------------------------------------
add_library(lse_dist
  src/dist/transport.cpp
  src/dist/loopback.cpp
  src/dist/codec.cpp
  src/dist/collective.cpp
)
add_library(lse::dist ALIAS lse_dist)
target_link_libraries(lse_dist PUBLIC lse_core lse_ir lse_backend_registry lse_communication)
# Transports self-register from a TU with no referenced symbols.
target_link_options(lse_dist INTERFACE
  "LINKER:--whole-archive,$<TARGET_FILE:lse_dist>,--no-whole-archive")
lse_apply_common_flags(lse_dist)

# ---------------------------------------------------------------------------
# probe — device qualification: what a pool member measurably does, and the
# cost model that reads it. Below lse_graph for the same reason dist is: a
# backend registers its device probe here, so this must not depend on lse_graph.
# ---------------------------------------------------------------------------
add_library(lse_probe
  src/probe/profile.cpp
  src/probe/device_probe.cpp
  src/probe/link_probe.cpp
  src/probe/cost_model.cpp
  src/probe/profile_store.cpp
  src/probe/pool.cpp
)
add_library(lse::probe ALIAS lse_probe)
target_link_libraries(lse_probe PUBLIC lse_core lse_ir lse_dist lse_backend_registry)
lse_apply_common_flags(lse_probe)

add_library(lse_graph
  src/graph/graph.cpp
  src/graph/ops.cpp
  src/graph/partitioner.cpp
  src/graph/workgroup.cpp
  src/graph/program.cpp
  src/graph/primitive.cpp
  src/graph/primitives_builtin.cpp
  src/graph/primitive_library.cpp
  src/graph/fallback.cpp
  src/graph/codegen.cpp
  src/graph/jit_cache.cpp
  src/graph/interpreter.cpp
  src/graph/scheduler.cpp
  src/graph/stream_plan.cpp
  src/graph/sharding.cpp
)
add_library(lse::graph ALIAS lse_graph)
target_link_libraries(lse_graph PUBLIC lse_core lse_ir lse_backend_registry)
# No backend edge here, in either direction of the build: lse_backend_hrx links
# lse_graph because it implements graph's emitter and compiler seams, so an edge
# back would be a cycle. Executables link lse_backends to make one selectable.
# Primitives self-register from a TU with no referenced symbols.
target_link_options(lse_graph INTERFACE
  "LINKER:--whole-archive,$<TARGET_FILE:lse_graph>,--no-whole-archive")

lse_apply_common_flags(lse_graph)
# Review copies of generated HIP land in ${CMAKE_BINARY_DIR}/hip/.
target_compile_definitions(lse_graph PRIVATE
  "LSE_BUILD_DIR=\"${CMAKE_BINARY_DIR}\"")

lse_add_rust_staticlib(fastokens_ffi
  "${CMAKE_CURRENT_SOURCE_DIR}/third_party/fastokens-ffi" fastokens_ffi)

if(TARGET fastokens_ffi)
  add_library(lse_tokenizer src/tokenizer/tokenizer.cpp)
  add_library(lse::tokenizer ALIAS lse_tokenizer)
  target_link_libraries(lse_tokenizer PUBLIC lse_core fastokens_ffi)
  lse_apply_common_flags(lse_tokenizer)
endif()

add_library(lse_ops
  src/ops/activation.cpp
  src/ops/attention.cpp
  src/ops/linear_attention.cpp
  src/ops/moe.cpp
  src/ops/norm.cpp
  src/ops/rope.cpp
)
add_library(lse::ops ALIAS lse_ops)
target_link_libraries(lse_ops PUBLIC lse_core lse_graph)
lse_apply_common_flags(lse_ops)

add_library(lse_runtime
  src/runtime/generator.cpp
  src/runtime/sampler.cpp
  src/runtime/session.cpp
)
add_library(lse::runtime ALIAS lse_runtime)
lse_apply_common_flags(lse_runtime)

add_library(lse_model
  src/model/config.cpp
  src/model/hybrid_lm.cpp
  src/model/layer.cpp
  src/model/registry.cpp
  src/model/weights.cpp
  src/models/lemonseed.cpp
  src/models/qwen3_5.cpp
  src/models/qwen3_5_common.cpp
  src/models/qwen3_5_moe.cpp
)
add_library(lse::model ALIAS lse_model)
# Model kernels self-register from a TU nothing references.
target_link_options(lse_model INTERFACE
  "LINKER:--whole-archive,$<TARGET_FILE:lse_model>,--no-whole-archive")
target_link_libraries(lse_model PUBLIC lse_core lse_graph lse_ops)
lse_apply_common_flags(lse_model)

target_link_libraries(lse_runtime PUBLIC lse_core lse_graph lse_model)

# ---------------------------------------------------------------------------
# cli — needs a tokenizer, so it is built only when the Rust shim is available.
# ---------------------------------------------------------------------------
if(TARGET lse_tokenizer)
  add_executable(lse src/cli/main.cpp)
  if(LSE_ROCM_RUNTIME_DIR)
    target_compile_definitions(lse PRIVATE
      LSE_HSA_RUNTIME_DIR="${LSE_ROCM_RUNTIME_DIR}")
  endif()
  target_link_libraries(lse PRIVATE lse_runtime lse_model lse_tokenizer
                                    lse_probe lse_backends)
  lse_apply_common_flags(lse)

  if(EXISTS "$ENV{LSE_SCRATCH}/layerdump.cpp")
    add_executable(layerdump "$ENV{LSE_SCRATCH}/layerdump.cpp")
    target_link_libraries(layerdump PRIVATE lse_runtime lse_model lse_tokenizer
                                            lse_probe lse_backends)
    lse_apply_common_flags(layerdump)
    add_executable(qlindiff "$ENV{LSE_SCRATCH}/qlindiff.cpp")
    target_link_libraries(qlindiff PRIVATE lse_runtime lse_model lse_tokenizer
                                           lse_probe lse_backends)
    lse_apply_common_flags(qlindiff)
  endif()
endif()

# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------
if(LSE_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------
message(STATUS "")
message(STATUS "Lemon Seed Engine ${PROJECT_VERSION}")
message(STATUS "  build type      : ${CMAKE_BUILD_TYPE}")
message(STATUS "  host compiler   : ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  host std        : C++${CMAKE_CXX_STANDARD}")
message(STATUS "  device std      : C++${LSE_DEVICE_CXX_STANDARD}")
message(STATUS "  HRX backend     : ${LSE_ENABLE_HRX}")
message(STATUS "  HIP available   : ${LSE_HAVE_HIP}")
message(STATUS "  comgr (JIT)     : ${LSE_HAVE_COMGR}")
message(STATUS "  HRX linked      : ${LSE_HAVE_HRX}")
message(STATUS "  HSA runtime     : ${LSE_ROCM_RUNTIME_DIR}")
message(STATUS "  GPU targets     : ${LSE_GPU_TARGETS}")
message(STATUS "  tests           : ${LSE_BUILD_TESTS}")
message(STATUS "")
