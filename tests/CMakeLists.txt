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
lse_add_test(test_kernel_env lse_graph)
lse_add_test(test_quant)
lse_add_test(test_shape)
lse_add_test(test_model lse_model)

if(LSE_ENABLE_CPU)
  lse_add_test(test_reference lse_model lse_graph)
  lse_add_test(test_runtime lse_runtime lse_model lse_graph)
endif()

if(TARGET lse_tokenizer)
  lse_add_test(test_tokenizer lse_tokenizer)
endif()

if(LSE_ENABLE_CPU)
  lse_add_test(test_backend_cpu lse_backend_cpu)
  lse_add_test(test_graph lse_graph)
  lse_add_test(test_primitive lse_graph)
endif()

# Codegen belongs to the backend now, so its test needs one that has codegen.
if(TARGET lse_backend_hrx)
  lse_add_test(test_jit lse_graph lse_backend_hrx)
endif()

# The GPU backend needs the 7.13 HSA runtime on the loader path; without it the
# tests still pass on the CPU backend, just without exercising a device.
if(LSE_ROCM_RUNTIME_DIR)
  get_property(lse_tests DIRECTORY PROPERTY TESTS)
  set_tests_properties(${lse_tests} PROPERTIES
    ENVIRONMENT "LD_LIBRARY_PATH=${LSE_ROCM_RUNTIME_DIR}")
endif()
