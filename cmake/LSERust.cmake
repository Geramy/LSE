# Drives cargo for the fastokens FFI shim and exposes it as an imported target.

find_program(LSE_CARGO cargo)

# fastokens is a path dependency of the shim and lives outside this repo, so a
# fresh clone has nothing to build against and cargo fails with a missing
# directory rather than anything naming the cause. Fetch it at the revision CI
# pins. Nothing here is a fallback: the checkout is the dependency, and a build
# that skipped it would silently drop the tokenizer.
set(LSE_FASTOKENS_REPO "https://github.com/crusoecloud/fastokens.git"
    CACHE STRING "Where to fetch fastokens from")
set(LSE_FASTOKENS_REF "7973014e4f3a6028ac48f305704eacd64d0b4ef6"
    CACHE STRING "fastokens revision to build against")

function(lse_provide_fastokens dir)
  if(EXISTS "${dir}/Cargo.toml")
    return()   # a working copy is already there; leave whatever is checked out
  endif()
  find_package(Git QUIET REQUIRED)
  message(STATUS "fetching fastokens ${LSE_FASTOKENS_REF} into ${dir}")
  get_filename_component(_parent "${dir}" DIRECTORY)
  file(MAKE_DIRECTORY "${_parent}")
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" clone --quiet "${LSE_FASTOKENS_REPO}" "${dir}"
    RESULT_VARIABLE _rc)
  if(NOT _rc EQUAL 0)
    message(FATAL_ERROR
      "could not clone ${LSE_FASTOKENS_REPO} into ${dir}. Clone it by hand, or "
      "point -DLSE_FASTOKENS_REPO at a mirror.")
  endif()
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" -C "${dir}" checkout --quiet "${LSE_FASTOKENS_REF}"
    RESULT_VARIABLE _rc)
  if(NOT _rc EQUAL 0)
    message(FATAL_ERROR "fastokens has no revision ${LSE_FASTOKENS_REF}")
  endif()
endfunction()

# A Rust staticlib does not carry the native C libraries its crates link
# against (pcre2, ring, ...), so the consumer must link them. Ask rustc for the
# list instead of hardcoding it — the set changes with fastokens' dependencies.
function(_lse_rust_native_libs crate_dir out_var)
  execute_process(
    COMMAND ${LSE_CARGO} rustc --release --lib
            --manifest-path "${crate_dir}/Cargo.toml"
            -- --print native-static-libs
    OUTPUT_VARIABLE _out ERROR_VARIABLE _err RESULT_VARIABLE _rc)

  set(_libs "")
  if(_rc EQUAL 0 AND "${_out}${_err}" MATCHES "native-static-libs: ([^\n]*)")
    string(REGEX MATCHALL "-l[^ ]+" _flags "${CMAKE_MATCH_1}")
    foreach(_flag IN LISTS _flags)
      string(REGEX REPLACE "^-l" "" _name "${_flag}")
      # The C runtime and unwinder come in via the C++ driver already.
      if(NOT _name MATCHES "^(c|gcc_s|m)$")
        list(APPEND _libs "${_name}")
      endif()
    endforeach()
  else()
    message(WARNING "could not query rustc for native libs; falling back")
    set(_libs pthread dl)
  endif()
  set(${out_var} "${_libs}" PARENT_SCOPE)
endfunction()

function(lse_add_rust_staticlib target crate_dir lib_name)
  if(NOT LSE_CARGO)
    message(STATUS "cargo not found — ${target} disabled")
    return()
  endif()

  set(_lib "${crate_dir}/target/release/lib${lib_name}.a")

  # Depend on the sources so cargo re-runs on change; cargo decides whether a
  # rebuild is actually needed.
  file(GLOB_RECURSE _srcs "${crate_dir}/src/*.rs" "${crate_dir}/Cargo.toml")

  add_custom_command(
    OUTPUT "${_lib}"
    COMMAND ${LSE_CARGO} build --release --manifest-path "${crate_dir}/Cargo.toml"
    DEPENDS ${_srcs}
    COMMENT "cargo build --release (${lib_name})"
    VERBATIM)

  add_custom_target(${target}_build DEPENDS "${_lib}")

  _lse_rust_native_libs("${crate_dir}" _native)
  message(STATUS "${lib_name} native deps: ${_native}")

  add_library(${target} INTERFACE)
  add_dependencies(${target} ${target}_build)
  target_include_directories(${target} INTERFACE "${crate_dir}/include")
  target_link_libraries(${target} INTERFACE "${_lib}" ${_native})
endfunction()
