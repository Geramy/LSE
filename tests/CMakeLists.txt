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

lse_add_test(test_dtype)
lse_add_test(test_reflect)
# lse_backends is the set of backends this build ships. A test that runs a
# graph links it directly: lse_graph implements the seam backends plug into and
# deliberately does not link any of them, so nothing arrives transitively.
lse_add_test(test_kernel_env lse_graph lse_backends)
lse_add_test(test_ir lse_ir lse_graph lse_backends)
lse_add_test(test_quant)
lse_add_test(test_shape)
lse_add_test(test_server lse_server lse_tokenizer lse_model lse_backends)
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
  lse_add_test(test_graph lse_graph lse_backends)
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
