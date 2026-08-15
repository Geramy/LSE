// Loads a .hip source file defining several primitives that share helpers.
//
//   __device__ inline float softplus(float x) { return __logf(1.0f + __expf(x)); }
//
//   // LSE_PRIMITIVE: mish arity=1
//   __device__ inline float lse_mish(float x) { return x * tanhf(softplus(x)); }
//
//   // LSE_PRIMITIVE: swiglu arity=2
//   __device__ inline float lse_swiglu(float a, float b) { ... }
//
// Every primitive in the file gets the whole file as its device preamble, so
// helpers are shared and emitted once per kernel. The function immediately
// following an annotation is the entry point.
#pragma once

#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "lse/core/status.hpp"
#include "lse/graph/primitive.hpp"

namespace lse::graph {

struct LibraryEntry {
  std::string primitive_name;
  std::string function_name;
  std::size_t arity = 1;
};

class PrimitiveLibrary {
 public:
  // Parses and registers every annotated primitive. The library owns the
  // primitives, so it must outlive any graph referencing them.
  static Result<std::shared_ptr<PrimitiveLibrary>> load_file(const std::string& path);
  static Result<std::shared_ptr<PrimitiveLibrary>> load_source(std::string source,
                                                               std::string origin);

  [[nodiscard]] const std::vector<LibraryEntry>& entries() const noexcept {
    return entries_;
  }
  [[nodiscard]] const std::string& source() const noexcept { return source_; }
  [[nodiscard]] const std::string& origin() const noexcept { return origin_; }

  ~PrimitiveLibrary();

 private:
  std::string source_;
  std::string origin_;
  std::vector<LibraryEntry> entries_;
  std::vector<std::unique_ptr<Primitive>> owned_;
};

// Parses the LSE_PRIMITIVE annotations without registering anything.
Result<std::vector<LibraryEntry>> parse_primitive_annotations(std::string_view source);

}  // namespace lse::graph
