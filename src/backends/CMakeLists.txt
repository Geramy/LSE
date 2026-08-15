# Backend modules. Each subdirectory is a self-contained backend that
# self-registers via LSE_REGISTER_BACKEND; linking it in is all that is needed
# to make it selectable by name at runtime.

add_library(lse_backend_registry
  registry.cpp
)
add_library(lse::backend_registry ALIAS lse_backend_registry)
target_link_libraries(lse_backend_registry PUBLIC lse_core)
lse_apply_common_flags(lse_backend_registry)

if(LSE_ENABLE_CPU)
  add_subdirectory(cpu)
endif()

if(LSE_ENABLE_HRX AND LSE_HRX_INCLUDE_DIR)
  add_subdirectory(hrx)
else()
  message(STATUS "HRX backend: skipped (set -DLSE_HRX_ROOT=<install> to enable)")
endif()
