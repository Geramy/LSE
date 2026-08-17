# HRX backend — the native low-latency path.
#
# This binds the *native* HRX C ABI (hrx_device_t / hrx_buffer_t /
# hrx_stream_dispatch / hrx_executable_load_data). It deliberately does not use
# the HIP compatibility layer HRX also ships (libamdhip64.so); that shim exists
# to run unmodified HIP applications, and going through it would give up the
# low-latency dispatch path that is the reason to be on HRX at all.

add_library(lse_backend_hrx
  hrx_backend.cpp
  hipc/hip_emitter.cpp
  hipc/phase_emit.cpp
  hipc/hip_sources.cpp
  hipc/comgr_compiler.cpp
  trace.cpp
  kernels/matmul.cpp
  kernels/reduction.cpp
  kernels/slice.cpp
  kernels/concat.cpp
  kernels/transpose.cpp
  kernels/gather.cpp
  kernels/scatter.cpp
  kernels/topk.cpp
  kernels/argmax.cpp
  kernels/linear_indexed.cpp
  kernels/conv.cpp
  kernels/rope.cpp
  kernels/sdpa.cpp
  kernels/kv_write.cpp
  kernels/gdn.cpp
  hipc/hip_types.cpp
  kernels/wmma_linear.cpp
  kernels/lds_linear.cpp
  kernels/quant_linear.cpp
  kernels/quant_embedding.cpp
  kernels/linked.cpp
  kernels/dist_codec.cpp
  kernels/probe.cpp
)
add_library(lse::backend_hrx ALIAS lse_backend_hrx)

target_link_libraries(lse_backend_hrx PUBLIC lse_core lse_backend_registry lse_graph lse_dist lse_probe lse_trace)
target_include_directories(lse_backend_hrx SYSTEM PUBLIC ${LSE_HRX_INCLUDE_DIR})
target_include_directories(lse_backend_hrx PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

if(LSE_HAVE_COMGR)
  target_compile_definitions(lse_backend_hrx PUBLIC LSE_HAVE_COMGR=1)
  # comgr does not know where the HIP headers live; the discovered include dir
  # is baked in and can be overridden at runtime with LSE_ROCM_INCLUDE.
  # comgr's bundled clang does not resolve its own resource headers here, so
  # the resource dir is discovered and passed explicitly.
  file(GLOB _lse_clang_res LIST_DIRECTORIES true
       "${LSE_ROCM_PATH}/core-*/lib/llvm/lib/clang/*/include"
       "${LSE_ROCM_PATH}/lib/llvm/lib/clang/*/include")
  list(GET _lse_clang_res 0 _lse_clang_res_dir)
  target_compile_definitions(lse_backend_hrx PRIVATE
    LSE_ROCM_INCLUDE_DIR="${LSE_HIP_INCLUDE_DIR}"
    LSE_CLANG_RESOURCE_DIR="${_lse_clang_res_dir}")
  message(STATUS "  clang resource  : ${_lse_clang_res_dir}")
  target_include_directories(lse_backend_hrx SYSTEM PUBLIC ${LSE_COMGR_INCLUDE_DIR})
  target_link_libraries(lse_backend_hrx PUBLIC ${LSE_COMGR_LIBRARY})
else()
  target_compile_definitions(lse_backend_hrx PUBLIC LSE_HAVE_COMGR=0)
endif()

if(LSE_HRX_LIBRARY)
  target_link_libraries(lse_backend_hrx PUBLIC ${LSE_HRX_LIBRARY})
  target_compile_definitions(lse_backend_hrx PUBLIC LSE_HRX_LINKED=1)
else()
  # Headers are available but libhrx has not been built yet. The backend still
  # compiles (so the API stays honest and type-checked against the real header)
  # but registers itself as unavailable at runtime.
  target_compile_definitions(lse_backend_hrx PUBLIC LSE_HRX_LINKED=0)
endif()

# See the cpu backend: self-registration requires whole-archive linking.
target_link_options(lse_backend_hrx INTERFACE
  "LINKER:--whole-archive,$<TARGET_FILE:lse_backend_hrx>,--no-whole-archive")

lse_apply_common_flags(lse_backend_hrx)
