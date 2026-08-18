# Build options and dependency discovery for the Lemon Seed Engine.

option(LSE_BUILD_TESTS    "Build the test suite"            ON)
option(LSE_BUILD_SERVER   "Build the OpenAI-compatible server" ON)
option(LSE_BUILD_CLI      "Build the lse CLI"               ON)
option(LSE_ENABLE_HRX     "Build the HRX backend"           ON)
option(LSE_ENABLE_CPU     "Build the CPU reference backend" ON)
option(LSE_WERROR         "Treat warnings as errors"        OFF)
option(LSE_ASAN           "Build with AddressSanitizer"     OFF)

# gfx1151 = Radeon 8060S (Strix Halo, this machine)
# gfx1201 = RDNA4 discrete
# gfx942  = MI300X (the lemonseed training target)
set(LSE_GPU_TARGETS "gfx1151;gfx1201;gfx942" CACHE STRING
    "AMDGPU targets to compile AOT kernels for")

set(LSE_ROCM_PATH "/opt/rocm" CACHE PATH "ROCm installation root")

# --- ROCm / HIP discovery ----------------------------------------------------
# We do not require the HIP *language* to be enabled at configure time; the
# core library must build on a machine with no GPU at all. HIP is probed and
# enabled only for the backends that need it.

# ROCm ships in two layouts: the classic flat one (<root>/include/hip/...) and
# TheRock-style component layout (<root>/core-<ver>/include/hip/...). Probe the
# component dirs as well so both work without the user pointing at internals.
# Component dirs come FIRST: the top-level lib/ and include/ aggregate an older
# component, and mixing them gives a comgr built against a different LLVM than
# the device bitcode it loads ("Unknown attribute kind").
file(GLOB _lse_rocm_component_dirs LIST_DIRECTORIES true "${LSE_ROCM_PATH}/core-*")
list(SORT _lse_rocm_component_dirs)
list(REVERSE _lse_rocm_component_dirs)
set(LSE_ROCM_INCLUDE_HINTS "")
set(LSE_ROCM_LIB_HINTS "")
foreach(_dir IN LISTS _lse_rocm_component_dirs)
  if(IS_DIRECTORY "${_dir}")
    list(APPEND LSE_ROCM_INCLUDE_HINTS "${_dir}/include")
    list(APPEND LSE_ROCM_LIB_HINTS "${_dir}/lib")
  endif()
endforeach()
list(APPEND LSE_ROCM_INCLUDE_HINTS "${LSE_ROCM_PATH}/include")
list(APPEND LSE_ROCM_LIB_HINTS "${LSE_ROCM_PATH}/lib")

find_path(LSE_HIP_INCLUDE_DIR
  NAMES hip/hip_runtime.h
  HINTS ${LSE_ROCM_INCLUDE_HINTS}
  NO_DEFAULT_PATH)
set(LSE_HAVE_HIP OFF)
if(LSE_HIP_INCLUDE_DIR)
  set(LSE_HAVE_HIP ON)
endif()

# amd_comgr is the *compiler* used by the kernel JIT: generated HIP source is
# turned into an AMDGPU code object without dragging in the HIP runtime, which
# matters because dispatch goes through the native HRX API, not libamdhip64.
find_path(LSE_COMGR_INCLUDE_DIR
  NAMES amd_comgr/amd_comgr.h
  HINTS ${LSE_ROCM_INCLUDE_HINTS}
  NO_DEFAULT_PATH)
find_library(LSE_COMGR_LIBRARY
  NAMES amd_comgr
  HINTS ${LSE_ROCM_LIB_HINTS}
  NO_DEFAULT_PATH)
set(LSE_HAVE_COMGR OFF)
if(LSE_COMGR_INCLUDE_DIR AND LSE_COMGR_LIBRARY)
  set(LSE_HAVE_COMGR ON)
endif()

# hipRTC is kept as a fallback compiler for environments without comgr.
find_path(LSE_HIPRTC_INCLUDE_DIR
  NAMES hip/hiprtc.h
  HINTS ${LSE_ROCM_INCLUDE_HINTS}
  NO_DEFAULT_PATH)
find_library(LSE_HIPRTC_LIBRARY
  NAMES hiprtc
  HINTS ${LSE_ROCM_LIB_HINTS}
  NO_DEFAULT_PATH)
set(LSE_HAVE_HIPRTC OFF)
if(LSE_HIPRTC_LIBRARY AND LSE_HIPRTC_INCLUDE_DIR)
  set(LSE_HAVE_HIPRTC ON)
endif()

# --- HRX discovery -----------------------------------------------------------
# HRX is an out-of-tree runtime; point LSE_HRX_ROOT at an install produced by
# `cmake --install ... --component HrxPublicDist`. When absent, the HRX backend
# is skipped and the engine falls back to the CPU backend.

set(LSE_HRX_ROOT "" CACHE PATH "HRX install root (libhrx.so + headers)")

# libhrx dlopens libhsa-runtime64 by soname, so the loader must reach the 7.13
# component's copy: the distro one in /lib is older and lacks symbols HRX needs.
# dlopen consults LD_LIBRARY_PATH as captured at process start, which nothing
# inside the process can change, so this is recorded and reported rather than
# baked into an RPATH.
# LSE_ROCM_LIB_HINTS puts the versioned components ahead of the aggregated
# top-level lib/, which is an older component.
find_path(LSE_ROCM_RUNTIME_DIR
  NAMES libhsa-runtime64.so.1
  HINTS ${LSE_ROCM_LIB_HINTS}
  NO_DEFAULT_PATH)

set(LSE_HAVE_HRX OFF)
if(LSE_ENABLE_HRX)
  find_path(LSE_HRX_INCLUDE_DIR
    NAMES hrx_runtime.h
    HINTS "${LSE_HRX_ROOT}/include"
          # The install component nests headers under include/hrx/.
          "${LSE_HRX_ROOT}/include/hrx"
          "${CMAKE_CURRENT_SOURCE_DIR}/reference/hrx-system/build/hrx-install/include/hrx"
          "${CMAKE_CURRENT_SOURCE_DIR}/reference/hrx-system/libhrx/include")
  find_library(LSE_HRX_LIBRARY
    NAMES hrx
    HINTS "${LSE_HRX_ROOT}/lib"
          "${CMAKE_CURRENT_SOURCE_DIR}/reference/hrx-system/build/hrx-install/lib")
  if(LSE_HRX_INCLUDE_DIR AND LSE_HRX_LIBRARY)
    set(LSE_HAVE_HRX ON)
  elseif(LSE_HRX_INCLUDE_DIR)
    # Headers only: we can compile the backend but not link it yet. Useful
    # while HRX is still being built out.
    message(STATUS "HRX headers found but libhrx not built yet — "
                   "HRX backend will be header-checked only")
  endif()
endif()

# --- loomc discovery ---------------------------------------------------------
# loomc is the second kernel generator inside the HRX runtime: it parses .loom
# text and emits an AMDGPU code object itself, with no LLVM in the graph and no
# ROCm or HSA dependency. It installs into the same prefix libhrx does, so
# nothing new has to be pointed at.

set(LSE_HAVE_LOOMC OFF)
if(LSE_ENABLE_HRX)
  find_path(LSE_LOOMC_INCLUDE_DIR
    NAMES loomc/loomc.h
    HINTS "${LSE_HRX_ROOT}/include"
          "${CMAKE_CURRENT_SOURCE_DIR}/reference/hrx-system/build/hrx-install/include")
  find_library(LSE_LOOMC_LIBRARY
    NAMES loomc
    HINTS "${LSE_HRX_ROOT}/lib"
          "${CMAKE_CURRENT_SOURCE_DIR}/reference/hrx-system/build/hrx-install/lib")
  find_file(LSE_LOOMC_VERSION_FILE
    NAMES loomc-config-version.cmake
    HINTS "${LSE_HRX_ROOT}/lib/cmake/loomc"
          "${CMAKE_CURRENT_SOURCE_DIR}/reference/hrx-system/build/hrx-install/lib/cmake/loomc")
  # Read out of the install rather than written here. The JIT cache key carries
  # it, and a version constant maintained on this side is one forgotten edit
  # away from serving an object a different loomc built.
  set(LSE_LOOMC_VERSION "unknown")
  if(LSE_LOOMC_VERSION_FILE)
    file(STRINGS "${LSE_LOOMC_VERSION_FILE}" _lse_loomc_version_line
         REGEX "^set\\(PACKAGE_VERSION ")
    if(_lse_loomc_version_line)
      string(REGEX REPLACE "^set\\(PACKAGE_VERSION \"([^\"]+)\".*$" "\\1"
             LSE_LOOMC_VERSION "${_lse_loomc_version_line}")
    endif()
  endif()
  if(LSE_LOOMC_INCLUDE_DIR AND LSE_LOOMC_LIBRARY)
    set(LSE_HAVE_LOOMC ON)
    message(STATUS "loomc ${LSE_LOOMC_VERSION}: ${LSE_LOOMC_LIBRARY}")
  elseif(LSE_LOOMC_INCLUDE_DIR)
    message(STATUS "loomc headers found but libloomc not built yet — "
                   "the loom dialect will report itself unavailable")
  endif()
endif()
