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
  VERSION 0.2.1
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
  src/ir/alias.cpp
  src/ir/pass/factor_hoist.cpp
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
# opt — optimization policy. The counting and the decisions that rest on it,
# owned once so no backend reimplements them. Depends on the backend vocabulary
# (device facts, kernel resources) and on nothing that lowers or emits.
# ---------------------------------------------------------------------------
add_library(lse_opt
  src/opt/arrangement.cpp
  src/opt/occupancy.cpp
  src/opt/fusion.cpp
  src/opt/measurements.cpp
  src/opt/traffic.cpp
)
add_library(lse::opt ALIAS lse_opt)
target_link_libraries(lse_opt PUBLIC lse_core lse_backend_registry)
lse_apply_common_flags(lse_opt)

# ---------------------------------------------------------------------------
# dist — transports and collectives. Deliberately below lse_graph: a backend
# registers its device codec engine here, so this must not depend on lse_graph
# (which link-depends on the backends).
# ---------------------------------------------------------------------------
add_library(lse_dist
  src/dist/transport.cpp
  src/dist/loopback.cpp
  src/dist/peer.cpp
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

# ---------------------------------------------------------------------------
# trace — what the work costs, continuously, and the Perfetto export of it.
# Distinct from probe on purpose: probe measures the hardware ONCE at admission,
# trace measures the WORK as it happens. Below lse_graph for the same reason
# probe and dist are — a backend registers its tracer here — and deliberately
# not linked against lse_graph, because lse_backend_hrx links both and an edge
# back would be the cycle the comment on lse_graph warns about.
# ---------------------------------------------------------------------------
add_library(lse_trace
  src/trace/record.cpp
  src/trace/collector.cpp
  src/trace/tracer.cpp
  src/trace/export.cpp
)
add_library(lse::trace ALIAS lse_trace)
# A tracer self-registers from its backend's TU, which is already whole-archived
# by that backend, so this needs no whole-archive of its own.
target_link_libraries(lse_trace PUBLIC lse_core lse_backend_registry)
lse_apply_common_flags(lse_trace)

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
target_link_libraries(lse_graph PUBLIC lse_core lse_ir lse_backend_registry lse_opt)

# Device kernel primitives. Written against the kir IR and a dialect source
# table, so they belong to the engine rather than to whichever backend
# happens to lower them.
add_library(lse_kernels
  src/kernels/argmax.cpp
  src/kernels/concat.cpp
  src/kernels/conv.cpp
  src/kernels/gather.cpp
  src/kernels/gdn.cpp
  src/kernels/kv_write.cpp
  src/kernels/lds_linear.cpp
  src/kernels/linear_indexed.cpp
  src/kernels/linked.cpp
  src/kernels/matmul.cpp
  src/kernels/quant_embedding.cpp
  src/kernels/quant_linear.cpp
  src/kernels/reduction.cpp
  src/kernels/rope.cpp
  src/kernels/scatter.cpp
  src/kernels/flash_sdpa.cpp
  src/kernels/sdpa.cpp
  src/kernels/slice.cpp
  src/kernels/topk.cpp
  src/kernels/transpose.cpp
  src/kernels/wmma_linear.cpp
  src/kernels/wmma_quant_linear.cpp
)
add_library(lse::kernels ALIAS lse_kernels)
target_link_libraries(lse_kernels PUBLIC lse_core lse_ir lse_graph)
# Each primitive self-registers from a TU nothing references, so the archive
# has to be linked whole or none of them exist at run time.
target_link_options(lse_kernels INTERFACE
  "LINKER:--whole-archive,$<TARGET_FILE:lse_kernels>,--no-whole-archive")
lse_apply_common_flags(lse_kernels)

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

lse_provide_fastokens("${CMAKE_CURRENT_SOURCE_DIR}/reference/fastokens")
lse_add_rust_staticlib(fastokens_ffi
  "${CMAKE_CURRENT_SOURCE_DIR}/third_party/fastokens-ffi" fastokens_ffi)

if(TARGET fastokens_ffi)
  add_library(lse_tokenizer src/tokenizer/tokenizer.cpp)
  add_library(lse::tokenizer ALIAS lse_tokenizer)
  target_link_libraries(lse_tokenizer PUBLIC lse_core fastokens_ffi)
  lse_apply_common_flags(lse_tokenizer)
endif()

# ---------------------------------------------------------------------------
# kv — the paged attention pool: block geometry, a refcounted free list and the
# admit/preempt/swap policy. Core only: no device, no graph, no session, so the
# allocation policy is testable with nothing attached.
# ---------------------------------------------------------------------------
add_library(lse_kv
  src/kv/allocator.cpp
  src/kv/block.cpp
  src/kv/policy.cpp
)
add_library(lse::kv ALIAS lse_kv)
target_link_libraries(lse_kv PUBLIC lse_core)
lse_apply_common_flags(lse_kv)

add_library(lse_ops
  src/ops/activation.cpp
  src/ops/attention.cpp
  src/ops/linear_attention.cpp
  src/ops/moe.cpp
  src/ops/norm.cpp
  src/ops/rope.cpp
)
add_library(lse::ops ALIAS lse_ops)
target_link_libraries(lse_ops PUBLIC lse_core lse_graph lse_kv)
lse_apply_common_flags(lse_ops)

add_library(lse_runtime
  src/runtime/batch.cpp
  src/runtime/generator.cpp
  src/runtime/sampler.cpp
  src/runtime/session.cpp
)
add_library(lse::runtime ALIAS lse_runtime)
lse_apply_common_flags(lse_runtime)

add_library(lse_model
  src/model/config.cpp
  src/model/hybrid_lm.cpp
  src/model/mtp.cpp
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
# place — the device SET: which devices this process holds, which of them holds
# which bytes, and where work runs. Deliberately the TOP layer: it depends on
# probe (to measure a pool) and on graph (to hand the set to a scheduler), and
# nothing depends on it. That direction is what keeps graph free of it — graph
# holds the backend::IDeviceSet seam and this implements it.
# ---------------------------------------------------------------------------
add_library(lse_place
  src/place/residency.cpp
  src/place/devices.cpp
  src/place/placement.cpp
)
add_library(lse::place ALIAS lse_place)
target_link_libraries(lse_place PUBLIC lse_core lse_probe lse_graph
                                       lse_backend_registry)
# The ROCm this was configured against, so a build pointed at a non-default one
# still finds its runtime when nobody exported anything.
target_compile_definitions(lse_place PRIVATE
  LSE_CONFIGURED_ROCM_PATH="${LSE_ROCM_PATH}")
# The set registers itself as graph's device-set factory from a TU with no
# referenced symbols, the same way transports and backends register.
target_link_options(lse_place INTERFACE
  "LINKER:--whole-archive,$<TARGET_FILE:lse_place>,--no-whole-archive")
lse_apply_common_flags(lse_place)

# ---------------------------------------------------------------------------
# cli — needs a tokenizer, so it is built only when the Rust shim is available.
# ---------------------------------------------------------------------------
if(TARGET lse_tokenizer)
  add_executable(lse src/cli/main.cpp)

  # Diagnostic: per-block hidden-state comparison across pool shapes.
  # scratch_probe/ is untracked working material, so the target exists only
  # where the file does — a clean clone (CI) has neither.
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/scratch_probe/layer_diff.cpp)
    add_executable(layer_diff scratch_probe/layer_diff.cpp)
    target_link_libraries(layer_diff PRIVATE lse_runtime lse_model
                                             lse_tokenizer lse_place lse_probe
                                             lse_backends)
    lse_apply_common_flags(layer_diff)
  endif()

  # The OpenAI-shaped HTTP surface. httplib and nlohmann/json are vendored
  # headers, so this adds no package to find.
  add_library(lse_server src/server/http_server.cpp src/server/chat.cpp)
  target_link_libraries(lse_server PUBLIC lse_core lse_model lse_runtime lse_tokenizer)
  lse_apply_common_flags(lse_server)
  find_package(Threads REQUIRED)
  target_link_libraries(lse_server PUBLIC Threads::Threads)

  add_executable(lse-server src/cli/server_main.cpp)
  if(LSE_ROCM_RUNTIME_DIR)
    target_compile_definitions(lse PRIVATE
      LSE_HSA_RUNTIME_DIR="${LSE_ROCM_RUNTIME_DIR}")
  endif()
  target_link_libraries(lse PRIVATE lse_runtime lse_model lse_tokenizer
                                    lse_place lse_probe lse_backends)
  lse_apply_common_flags(lse)

  # The server binds the same backends the CLI does; the device is opened by
  # whichever of them comes up, not by the HTTP layer.
  if(LSE_ROCM_RUNTIME_DIR)
    target_compile_definitions(lse-server PRIVATE
      LSE_HSA_RUNTIME_DIR="${LSE_ROCM_RUNTIME_DIR}")
  endif()
  target_link_libraries(lse-server PRIVATE lse_server lse_runtime lse_model
                                           lse_tokenizer lse_place lse_probe
                                           lse_backends)
  lse_apply_common_flags(lse-server)
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
message(STATUS "  loomc (JIT)     : ${LSE_HAVE_LOOMC} ${LSE_LOOMC_VERSION}")
message(STATUS "  HRX linked      : ${LSE_HAVE_HRX}")
message(STATUS "  HSA runtime     : ${LSE_ROCM_RUNTIME_DIR}")
message(STATUS "  GPU targets     : ${LSE_GPU_TARGETS}")
message(STATUS "  tests           : ${LSE_BUILD_TESTS}")
message(STATUS "")
