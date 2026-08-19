# HRX backend — the native low-latency path.
#
# This binds the *native* HRX C ABI (hrx_device_t / hrx_buffer_t /
# hrx_stream_dispatch / hrx_executable_load_data). It deliberately does not use
# the HIP compatibility layer HRX also ships (libamdhip64.so); that shim exists
# to run unmodified HIP applications, and going through it would give up the
# low-latency dispatch path that is the reason to be on HRX at all.

add_library(lse_backend_hrx
  hrx_backend.cpp
  code_object.cpp
  hipc/hip_emitter.cpp
  hipc/phase_emit.cpp
  hipc/hip_sources.cpp
  hipc/comgr_compiler.cpp
  loomc/loomc_compiler.cpp
  loomc/loom_types.cpp
  loomc/loom_sources.cpp
  loomc/loom_print.cpp
  loomc/loom_emitter.cpp
  trace.cpp
  hipc/hip_types.cpp
  kernels/dist_codec.cpp
  kernels/probe.cpp
)
add_library(lse::backend_hrx ALIAS lse_backend_hrx)

target_link_libraries(lse_backend_hrx PUBLIC lse_core lse_backend_registry lse_graph lse_kernels lse_dist lse_probe lse_trace)
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

if(LSE_HAVE_LOOMC)
  target_compile_definitions(lse_backend_hrx PUBLIC LSE_HAVE_LOOMC=1)
  # SYSTEM so loomc's headers are not held to this project's warning flags, and
  # the installed library rather than a rebuild: libloomc exports only loomc_*,
  # statically absorbs its own IREE and pulls in no LLVM, ROCm or HSA, so it
  # sits beside libhrx with nothing to reconcile.
  target_include_directories(lse_backend_hrx SYSTEM PUBLIC ${LSE_LOOMC_INCLUDE_DIR})
  target_link_libraries(lse_backend_hrx PUBLIC ${LSE_LOOMC_LIBRARY})
  target_compile_definitions(lse_backend_hrx PUBLIC LOOMC_USING_SHARED_LIBRARY)
  # loomc publishes no version query and its .so carries no build-id note, so
  # identity() stats the resolved library to notice a rebuild at the same
  # version. Both are baked, not searched for, at runtime.
  target_compile_definitions(lse_backend_hrx PRIVATE
    LSE_LOOMC_VERSION="${LSE_LOOMC_VERSION}"
    LSE_LOOMC_LIBRARY_PATH="${LSE_LOOMC_LIBRARY}")
else()
  target_compile_definitions(lse_backend_hrx PUBLIC LSE_HAVE_LOOMC=0)
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
