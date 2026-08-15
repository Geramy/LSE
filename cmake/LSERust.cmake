# Drives cargo for the fastokens FFI shim and exposes it as an imported target.

find_program(LSE_CARGO cargo)

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
