# Shared compiler/linker flags. Applied via lse_apply_common_flags(<target>) so
# every target gets the same treatment without repeating the list.

function(lse_apply_common_flags target)
  target_compile_options(${target} PRIVATE
    -Wall
    -Wextra
    -Wpedantic
    -Wshadow
    -Wconversion
    -Wsign-conversion
    -Wnon-virtual-dtor
    -Wcast-align
    -Wunused
    -Woverloaded-virtual
    -Wnull-dereference
    -Wdouble-promotion
    -Wformat=2
  )

  if(LSE_WERROR)
    target_compile_options(${target} PRIVATE -Werror)
  endif()

  # P2996 is off until this flag is set; __cpp_impl_reflection follows it.
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU"
      AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 16)
    target_compile_options(${target} PRIVATE -freflection)
  endif()

  if(LSE_ASAN)
    target_compile_options(${target} PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_options(${target} PRIVATE -fsanitize=address,undefined)
  endif()

  # The engine is a numerics project: never let the compiler reassociate float
  # math behind our backs. Reproducibility across backends depends on it.
  target_compile_options(${target} PRIVATE -fno-fast-math)

  set_target_properties(${target} PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
  )
endfunction()
