# Tests use a dependency-free header-only harness (tests/harness.hpp) so the
# suite runs on a bare machine with no package manager involved.

function(lse_add_test name)
  add_executable(${name} ${name}.cpp)
  target_link_libraries(${name} PRIVATE lse_core lse_backend_registry ${ARGN})
  target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  lse_apply_common_flags(${name})
  # Data-driven tests resolve fixtures relative to the repo root, so pin the
  # working directory rather than depending on how ctest was invoked.
  add_test(NAME ${name} COMMAND ${name}
           WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
endfunction()

lse_add_test(test_generation_stats)
lse_add_test(test_http_timings)
target_include_directories(test_http_timings PRIVATE
  ${CMAKE_SOURCE_DIR}/src/server ${CMAKE_SOURCE_DIR}/third_party/vendor)
# Ordinary fused SSA chains need no device or model.
lse_add_test(test_pointwise_fusion lse_graph lse_kernels lse_ir)
target_sources(test_pointwise_fusion PRIVATE
  ${CMAKE_SOURCE_DIR}/src/backends/hrx/loomc/loom_emitter.cpp
  ${CMAKE_SOURCE_DIR}/src/backends/hrx/loomc/loom_print.cpp
  ${CMAKE_SOURCE_DIR}/src/backends/hrx/loomc/loom_types.cpp
  ${CMAKE_SOURCE_DIR}/src/backends/hrx/loomc/loom_sources.cpp)
lse_add_test(test_dispatch_profile)
lse_add_test(test_dtype)
lse_add_test(test_reflect)
# lse_backends is the set of backends this build ships. A test that runs a
# graph links it directly: lse_graph implements the seam backends plug into and
# deliberately does not link any of them, so nothing arrives transitively.
lse_add_test(test_kernel_env lse_graph lse_backends)
lse_add_test(test_ir lse_ir lse_graph lse_backends)
# The Loom text printer only needs kernel IR; keep these regressions usable
# without libhrx, the Loom compiler, or a GPU.
lse_add_test(test_loom_print lse_ir)
target_sources(test_loom_print PRIVATE
  ${CMAKE_SOURCE_DIR}/src/backends/hrx/loomc/loom_print.cpp
  ${CMAKE_SOURCE_DIR}/src/backends/hrx/loomc/loom_types.cpp
  ${CMAKE_SOURCE_DIR}/src/backends/hrx/loomc/loom_sources.cpp)
# Repeat uses the emitter directly; compile-only coverage needs no GPU runtime.
lse_add_test(test_loom_repeat lse_graph lse_kernels lse_ir)
target_sources(test_loom_repeat PRIVATE
  ${CMAKE_SOURCE_DIR}/src/backends/hrx/loomc/loom_emitter.cpp
  ${CMAKE_SOURCE_DIR}/src/backends/hrx/loomc/loom_print.cpp
  ${CMAKE_SOURCE_DIR}/src/backends/hrx/loomc/loom_types.cpp
  ${CMAKE_SOURCE_DIR}/src/backends/hrx/loomc/loom_sources.cpp)
lse_add_test(test_loom_extent lse_ir)
target_sources(test_loom_extent PRIVATE
  ${CMAKE_SOURCE_DIR}/src/backends/hrx/loomc/loom_print.cpp
  ${CMAKE_SOURCE_DIR}/src/backends/hrx/loomc/loom_types.cpp
  ${CMAKE_SOURCE_DIR}/src/backends/hrx/loomc/loom_sources.cpp)
lse_add_test(test_loom_conv lse_graph lse_kernels lse_ir)
target_sources(test_loom_conv PRIVATE
  ${CMAKE_SOURCE_DIR}/src/kernels/conv.cpp
  ${CMAKE_SOURCE_DIR}/src/backends/hrx/loomc/loom_emitter.cpp
  ${CMAKE_SOURCE_DIR}/src/backends/hrx/loomc/loom_print.cpp
  ${CMAKE_SOURCE_DIR}/src/backends/hrx/loomc/loom_types.cpp
  ${CMAKE_SOURCE_DIR}/src/backends/hrx/loomc/loom_sources.cpp)
# GDN register-state emission needs neither a GPU nor a loaded runtime.
lse_add_test(test_loom_gdn lse_graph lse_kernels lse_ir)
target_sources(test_loom_gdn PRIVATE
  ${CMAKE_SOURCE_DIR}/src/backends/hrx/loomc/loom_emitter.cpp
  ${CMAKE_SOURCE_DIR}/src/backends/hrx/loomc/loom_print.cpp
  ${CMAKE_SOURCE_DIR}/src/backends/hrx/loomc/loom_types.cpp
  ${CMAKE_SOURCE_DIR}/src/backends/hrx/loomc/loom_sources.cpp)
lse_add_test(test_loom_words lse_ir)
target_sources(test_loom_words PRIVATE
  ${CMAKE_SOURCE_DIR}/src/backends/hrx/loomc/loom_print.cpp
  ${CMAKE_SOURCE_DIR}/src/backends/hrx/loomc/loom_types.cpp
  ${CMAKE_SOURCE_DIR}/src/backends/hrx/loomc/loom_sources.cpp)
lse_add_test(test_loom_flash lse_graph lse_kernels lse_ir)
target_sources(test_loom_flash PRIVATE
  ${CMAKE_SOURCE_DIR}/src/backends/hrx/loomc/loom_emitter.cpp
  ${CMAKE_SOURCE_DIR}/src/backends/hrx/loomc/loom_print.cpp
  ${CMAKE_SOURCE_DIR}/src/backends/hrx/loomc/loom_types.cpp
  ${CMAKE_SOURCE_DIR}/src/backends/hrx/loomc/loom_sources.cpp)
lse_add_test(test_loom_cache lse_graph lse_kernels lse_ir)
target_sources(test_loom_cache PRIVATE
  ${CMAKE_SOURCE_DIR}/src/backends/hrx/loomc/loom_emitter.cpp
  ${CMAKE_SOURCE_DIR}/src/backends/hrx/loomc/loom_print.cpp
  ${CMAKE_SOURCE_DIR}/src/backends/hrx/loomc/loom_types.cpp
  ${CMAKE_SOURCE_DIR}/src/backends/hrx/loomc/loom_sources.cpp)
lse_add_test(test_quant)
# Compiler/layout contracts only; this test never opens a GPU.
lse_add_test(test_quant_prefill lse_graph lse_kernels lse_backends lse_ir)
if(NOT TARGET lse_backend_hrx)
  target_sources(test_quant_prefill PRIVATE
    ${CMAKE_SOURCE_DIR}/src/backends/hrx/hipc/hip_emitter.cpp
    ${CMAKE_SOURCE_DIR}/src/backends/hrx/hipc/phase_emit.cpp
    ${CMAKE_SOURCE_DIR}/src/backends/hrx/hipc/hip_sources.cpp
    ${CMAKE_SOURCE_DIR}/src/backends/hrx/hipc/hip_types.cpp
    ${CMAKE_SOURCE_DIR}/src/backends/hrx/loomc/loom_emitter.cpp
    ${CMAKE_SOURCE_DIR}/src/backends/hrx/loomc/loom_print.cpp
    ${CMAKE_SOURCE_DIR}/src/backends/hrx/loomc/loom_types.cpp
    ${CMAKE_SOURCE_DIR}/src/backends/hrx/loomc/loom_sources.cpp)
endif()
lse_add_test(test_shape)
lse_add_test(test_server lse_server lse_tokenizer lse_model lse_backends)
# Socket lifecycle coverage never opens a model or device.
find_package(Threads REQUIRED)
add_executable(test_server_shutdown test_server_shutdown.cpp)
target_include_directories(test_server_shutdown PRIVATE
  ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/third_party/vendor)
target_link_libraries(test_server_shutdown PRIVATE Threads::Threads)
lse_apply_common_flags(test_server_shutdown)
add_test(NAME test_server_shutdown COMMAND test_server_shutdown)
set_tests_properties(test_server_shutdown PROPERTIES TIMEOUT 15)

lse_add_test(test_model lse_model lse_backends)

if(LSE_ENABLE_CPU)
  lse_add_test(test_reference lse_model lse_graph lse_backends)
  lse_add_test(test_runtime lse_runtime lse_model lse_graph lse_backends)
endif()

if(TARGET lse_tokenizer)
  lse_add_test(test_tokenizer lse_tokenizer)
endif()

if(LSE_ENABLE_CPU)
  lse_add_test(test_backend_cpu lse_backend_cpu)
  lse_add_test(test_graph lse_graph lse_kernels lse_backends)
  lse_add_test(test_primitive lse_graph lse_backends)
endif()

# Codegen belongs to the backend now, so its test needs one that has codegen.
if(TARGET lse_backend_hrx)
  lse_add_test(test_jit lse_graph lse_backends)
endif()

# Client and server both run in this process over a real socket and a real
# abstract unix pipe, so this suite needs no network and no device. It is
# unguarded on purpose: a transport seam that cannot be exercised on a bare
# machine is one nobody reviews.
lse_add_test(test_communication lse_communication)

# dist runs its ranks on threads and its codecs on whatever device answers;
# with no GPU the device cases skip and the host ones still cover the algebra.
lse_add_test(test_dist lse_dist)
if(TARGET lse_backend_hrx)
  target_link_libraries(test_dist PRIVATE lse_graph lse_backends)
endif()
find_package(Threads REQUIRED)
target_link_libraries(test_dist PRIVATE Threads::Threads)

# The probe: the cost model runs on injected profiles with no device at all,
# and the link matrix runs its ranks on threads over the loopback transport.
lse_add_test(test_probe lse_probe lse_dist)
if(TARGET lse_backend_hrx)
  target_link_libraries(test_probe PRIVATE lse_graph)
endif()
# The live device set is qualified by the same probe, so it is exercised here
# rather than in a suite of its own: place owns the lifetime, probe measures it.
target_link_libraries(test_probe PRIVATE lse_place)
target_link_libraries(test_probe PRIVATE lse_backends)
target_link_libraries(test_probe PRIVATE Threads::Threads)

# The trace module: the writer, the collector and the pftrace bytes need no
# device, and the suite asserts on the encoded bytes rather than on a viewer.
# lse_backends is linked so the same seam can be exercised against a live
# backend's real dispatches where there is one; those cases skip on a bare
# machine.
lse_add_test(test_trace lse_trace)
target_link_libraries(test_trace PRIVATE lse_graph lse_backends Threads::Threads)

# The GPU backend needs the 7.13 HSA runtime on the loader path; without it the
# tests still pass on the CPU backend, just without exercising a device.
if(LSE_ROCM_RUNTIME_DIR)
  get_property(lse_tests DIRECTORY PROPERTY TESTS)
  set_tests_properties(${lse_tests} PROPERTIES
    ENVIRONMENT "LD_LIBRARY_PATH=${LSE_ROCM_RUNTIME_DIR}")
endif()

# Exercise argument handling without opening a model, device or HTTP listener.
find_package(Python3 QUIET COMPONENTS Interpreter)
if(Python3_Interpreter_FOUND AND TARGET lse-server)
  add_test(NAME test_server_cli
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_server_cli.py
            $<TARGET_FILE:lse-server>)
endif()

lse_add_test(test_probe_measurement)
lse_add_test(test_token_ids)
if(TARGET lse_backend_hrx AND LSE_HAVE_LOOMC)
  lse_add_test(test_loom_probe lse_backend_hrx)
  if(TARGET lse_backend_cpu)
    lse_add_test(test_matrix_probe_lifecycle lse_backend_hrx lse_backend_cpu lse_probe)
    target_sources(test_matrix_probe_lifecycle PRIVATE
      ${CMAKE_SOURCE_DIR}/src/backends/hrx/kernels/probe.cpp)
  endif()
endif()
if(TARGET lse_backend_cpu)
  lse_add_test(test_probe_policy lse_probe lse_backend_cpu)
endif()

lse_add_test(test_submission_tuner)
lse_add_test(test_submission_constants lse_graph lse_kernels lse_ir)
target_sources(test_submission_constants PRIVATE
  ${CMAKE_SOURCE_DIR}/src/backends/hrx/loomc/loom_emitter.cpp
  ${CMAKE_SOURCE_DIR}/src/backends/hrx/loomc/loom_print.cpp
  ${CMAKE_SOURCE_DIR}/src/backends/hrx/loomc/loom_types.cpp
  ${CMAKE_SOURCE_DIR}/src/backends/hrx/loomc/loom_sources.cpp)
if(TARGET lse_backend_cpu)
  lse_add_test(test_decode_sample lse_graph lse_kernels lse_backend_cpu)
  lse_add_test(test_submission_decode lse_runtime lse_model lse_graph lse_backends lse_backend_cpu)
endif()

lse_add_test(test_attention_decode lse_graph lse_kernels lse_ir)
target_sources(test_attention_decode PRIVATE
  ${CMAKE_SOURCE_DIR}/src/backends/hrx/loomc/loom_emitter.cpp
  ${CMAKE_SOURCE_DIR}/src/backends/hrx/loomc/loom_print.cpp
  ${CMAKE_SOURCE_DIR}/src/backends/hrx/loomc/loom_types.cpp
  ${CMAKE_SOURCE_DIR}/src/backends/hrx/loomc/loom_sources.cpp)

if(APPLE)
  add_library(lse_runtime_fixture SHARED loaded_runtime_fixture.cpp)
  set_target_properties(lse_runtime_fixture PROPERTIES VERSION 1.2 SOVERSION 1)
  add_executable(test_loaded_runtime test_loaded_runtime.cpp)
  target_link_libraries(test_loaded_runtime PRIVATE lse_core)
  target_include_directories(test_loaded_runtime PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  lse_apply_common_flags(test_loaded_runtime)
  add_dependencies(test_loaded_runtime lse_runtime_fixture)
  add_test(NAME test_loaded_runtime COMMAND ${CMAKE_COMMAND} -E env
    "DYLD_LIBRARY_PATH=$<TARGET_FILE_DIR:lse_runtime_fixture>"
    $<TARGET_FILE:test_loaded_runtime> $<TARGET_FILE:lse_runtime_fixture>)
endif()

lse_add_test(test_hrx_copy_route)

# Exact paired recurrence scheduling and logical binding contracts, no GPU.
lse_add_test(test_gdn_pair lse_graph lse_kernels lse_backends lse_ir)
if(NOT TARGET lse_backend_hrx)
  target_sources(test_gdn_pair PRIVATE
    ${CMAKE_SOURCE_DIR}/src/backends/hrx/hipc/hip_emitter.cpp
    ${CMAKE_SOURCE_DIR}/src/backends/hrx/hipc/phase_emit.cpp
    ${CMAKE_SOURCE_DIR}/src/backends/hrx/hipc/hip_sources.cpp
    ${CMAKE_SOURCE_DIR}/src/backends/hrx/hipc/hip_types.cpp
    ${CMAKE_SOURCE_DIR}/src/backends/hrx/loomc/loom_emitter.cpp
    ${CMAKE_SOURCE_DIR}/src/backends/hrx/loomc/loom_print.cpp
    ${CMAKE_SOURCE_DIR}/src/backends/hrx/loomc/loom_types.cpp
    ${CMAKE_SOURCE_DIR}/src/backends/hrx/loomc/loom_sources.cpp)
endif()

if(TARGET lse_backend_cpu)
  lse_add_test(test_scheduler_epilogue lse_graph lse_kernels lse_ir lse_backend_cpu)
  target_sources(test_scheduler_epilogue PRIVATE
    ${CMAKE_SOURCE_DIR}/src/backends/hrx/loomc/loom_emitter.cpp
    ${CMAKE_SOURCE_DIR}/src/backends/hrx/loomc/loom_print.cpp
    ${CMAKE_SOURCE_DIR}/src/backends/hrx/loomc/loom_types.cpp
    ${CMAKE_SOURCE_DIR}/src/backends/hrx/loomc/loom_sources.cpp)
endif()

lse_add_test(test_loom_matrix lse_ir lse_kernels lse_graph)
target_sources(test_loom_matrix PRIVATE
  ${CMAKE_SOURCE_DIR}/src/backends/hrx/loomc/loom_print.cpp
  ${CMAKE_SOURCE_DIR}/src/backends/hrx/loomc/loom_sources.cpp
  ${CMAKE_SOURCE_DIR}/src/backends/hrx/loomc/loom_types.cpp)
if(LSE_ENABLE_HRX)
  add_executable(compile_loom_matrix compile_loom_matrix.cpp)
  lse_apply_common_flags(compile_loom_matrix)
  target_link_libraries(compile_loom_matrix PRIVATE lse_backend_hrx lse_graph lse_kernels)
endif()

# Mixed-sign dot and activation rounding remain independently typed and gated.
lse_add_test(test_loom_dot lse_graph lse_kernels lse_ir)
target_sources(test_loom_dot PRIVATE
  ${CMAKE_SOURCE_DIR}/src/backends/hrx/loomc/loom_emitter.cpp
  ${CMAKE_SOURCE_DIR}/src/backends/hrx/loomc/loom_print.cpp
  ${CMAKE_SOURCE_DIR}/src/backends/hrx/loomc/loom_types.cpp
  ${CMAKE_SOURCE_DIR}/src/backends/hrx/loomc/loom_sources.cpp)
if(TARGET lse_backend_hrx AND LSE_HAVE_LOOMC)
  lse_add_test(test_loom_dot_native lse_backend_hrx lse_graph lse_kernels lse_ir)
endif()

if(TARGET lse_backend_cpu)
  lse_add_test(test_gdn_scheduler lse_graph lse_kernels lse_ir lse_backend_cpu)
  target_sources(test_gdn_scheduler PRIVATE
    ${CMAKE_SOURCE_DIR}/src/backends/hrx/loomc/loom_emitter.cpp
    ${CMAKE_SOURCE_DIR}/src/backends/hrx/loomc/loom_print.cpp
    ${CMAKE_SOURCE_DIR}/src/backends/hrx/loomc/loom_types.cpp
    ${CMAKE_SOURCE_DIR}/src/backends/hrx/loomc/loom_sources.cpp)
endif()

# Tests of the existing approximate Q4 algorithms explicitly opt in. The
# independent policy suite below exercises the default and both flag values.
if(TEST test_jit)
  set_property(TEST test_jit APPEND PROPERTY ENVIRONMENT "LSE_HRX_INT8=1")
endif()
set_property(TEST test_loom_dot APPEND PROPERTY ENVIRONMENT "LSE_HRX_INT8=1")
find_package(Python3 QUIET COMPONENTS Interpreter)
if(TARGET lse_backend_hrx AND LSE_HAVE_LOOMC AND Python3_Interpreter_FOUND)
  add_executable(test_int8_policy test_int8_policy.cpp)
  target_link_libraries(test_int8_policy PRIVATE lse_backend_hrx lse_graph lse_kernels lse_ir)
  target_include_directories(test_int8_policy PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  lse_apply_common_flags(test_int8_policy)
  add_test(NAME test_int8_policy COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/test_int8_policy.py $<TARGET_FILE:test_int8_policy>)
endif()

add_test(NAME test_loom_matrix_int8 COMMAND ${CMAKE_COMMAND} -E env
  LSE_HRX_INT8=1 $<TARGET_FILE:test_loom_matrix>)

lse_add_test(test_fp8_conversion lse_ir lse_graph)
if(TARGET lse_backend_hrx AND LSE_HAVE_LOOMC)
  lse_add_test(test_fp8_conversion_native lse_backend_hrx lse_graph lse_kernels lse_ir)
endif()

add_executable(test_quant_operand_policy test_quant_operand_policy.cpp)
target_include_directories(test_quant_operand_policy PRIVATE ${PROJECT_SOURCE_DIR}/include)
lse_apply_common_flags(test_quant_operand_policy)
add_test(NAME test_quant_operand_policy COMMAND test_quant_operand_policy)
if(TARGET lse_backend_hrx)
  add_executable(test_quant_operand_cache test_quant_operand_cache.cpp)
  target_link_libraries(test_quant_operand_cache PRIVATE lse_backend_hrx lse_graph lse_kernels lse_ir)
  lse_apply_common_flags(test_quant_operand_cache)
  add_test(NAME test_quant_operand_cache COMMAND test_quant_operand_cache)
endif()

if(TARGET lse_backend_hrx)
  lse_add_test(test_q6_wmma lse_backend_hrx lse_graph lse_kernels lse_ir)
endif()

# Loader overrides must change compiler identity even when CMake linked the
# default image. Fixtures require no GPU or compiler runtime initialization.
lse_add_test(test_compiler_identity)
target_link_libraries(test_compiler_identity PRIVATE ${CMAKE_DL_LIBS})
if(UNIX)
  foreach(variant IN ITEMS default override)
    add_library(lse_compiler_fixture_${variant} SHARED compiler_identity_fixture.cpp)
    set_target_properties(lse_compiler_fixture_${variant} PROPERTIES
      OUTPUT_NAME lse_compiler_identity_fixture
      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/compiler-${variant}")
  endforeach()
  target_compile_definitions(lse_compiler_fixture_override PRIVATE LSE_COMPILER_FIXTURE_VALUE=2)
  add_executable(test_compiler_override test_compiler_override.cpp)
  target_link_libraries(test_compiler_override PRIVATE lse_core lse_compiler_fixture_default ${CMAKE_DL_LIBS})
  lse_apply_common_flags(test_compiler_override)
  add_dependencies(test_compiler_override lse_compiler_fixture_override)
  if(APPLE)
    set(_lse_loader_path DYLD_LIBRARY_PATH)
  else()
    set(_lse_loader_path LD_LIBRARY_PATH)
  endif()
  add_test(NAME compiler_identity_default COMMAND ${CMAKE_COMMAND} -E env
    "${_lse_loader_path}=$<TARGET_FILE_DIR:lse_compiler_fixture_default>"
    $<TARGET_FILE:test_compiler_override> 1 $<TARGET_FILE:lse_compiler_fixture_default>)
  add_test(NAME compiler_identity_override COMMAND ${CMAKE_COMMAND} -E env
    "${_lse_loader_path}=$<TARGET_FILE_DIR:lse_compiler_fixture_override>"
    $<TARGET_FILE:test_compiler_override> 2 $<TARGET_FILE:lse_compiler_fixture_override>)
endif()

lse_add_test(test_cooperative_rms lse_graph lse_kernels lse_ir)
if(TARGET lse_backend_hrx)
  target_link_libraries(test_cooperative_rms PRIVATE lse_backend_hrx)
else()
  target_sources(test_cooperative_rms PRIVATE
    ${CMAKE_SOURCE_DIR}/src/backends/hrx/hipc/hip_emitter.cpp
    ${CMAKE_SOURCE_DIR}/src/backends/hrx/hipc/phase_emit.cpp
    ${CMAKE_SOURCE_DIR}/src/backends/hrx/hipc/hip_sources.cpp
    ${CMAKE_SOURCE_DIR}/src/backends/hrx/hipc/hip_types.cpp
    ${CMAKE_SOURCE_DIR}/src/backends/hrx/loomc/loom_emitter.cpp
    ${CMAKE_SOURCE_DIR}/src/backends/hrx/loomc/loom_print.cpp
    ${CMAKE_SOURCE_DIR}/src/backends/hrx/loomc/loom_types.cpp
    ${CMAKE_SOURCE_DIR}/src/backends/hrx/loomc/loom_sources.cpp)
endif()

if(TARGET lse_backend_hrx)
  lse_add_test(test_q6_m256_profile lse_backend_hrx lse_graph lse_kernels lse_ir)
  lse_add_test(test_tile_vector_load lse_backend_hrx lse_graph lse_kernels lse_ir)
endif()
