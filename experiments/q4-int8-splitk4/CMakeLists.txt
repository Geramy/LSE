if(NOT TARGET lse_backend_hrx OR NOT LSE_HAVE_LOOMC)
  message(FATAL_ERROR "The split-K4 probe requires HRX headers/library and Loomc")
endif()
add_executable(q4_int8_splitk4_probe
  native.cpp partial.cpp merge.cpp global_panel.cpp producer.cpp q8_matrix.cpp)
target_link_libraries(q4_int8_splitk4_probe PRIVATE
  lse_graph lse_kernels lse_backends lse_ir)
lse_apply_common_flags(q4_int8_splitk4_probe)
