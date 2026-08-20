# Backend modules. Each subdirectory is a self-contained backend that
# self-registers via LSE_REGISTER_BACKEND; linking it in is all that is needed
# to make it selectable by name at runtime.

add_library(lse_backend_registry
  registry.cpp
  resources.cpp
  census.cpp
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

# The backends this build ships. A backend registers itself from a TU nothing
# references, so it is present only where it is linked, and it is linked by the
# executable that intends to select one — never by lse_graph, which implements
# the seam a backend plugs into and must not depend on any implementation of it.
add_library(lse_backends INTERFACE)
add_library(lse::backends ALIAS lse_backends)
if(TARGET lse_backend_cpu)
  target_link_libraries(lse_backends INTERFACE lse_backend_cpu)
endif()
if(TARGET lse_backend_hrx)
  target_link_libraries(lse_backends INTERFACE lse_backend_hrx)
endif()
