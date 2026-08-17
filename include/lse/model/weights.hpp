// safetensors reader. mmaps the file and hands out zero-copy views; nothing is
// materialized until a tensor is uploaded to a device.
#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "lse/core/dtype.hpp"
#include "lse/core/shape.hpp"
#include "lse/core/status.hpp"

namespace lse::model {

struct TensorView {
  std::string name;
  Shape shape;
  DType dtype = DType::kBF16;
  std::span<const std::byte> data;

  [[nodiscard]] std::size_t element_count() const noexcept { return shape.elem_count(); }

  // Widens to f32 regardless of stored dtype.
  Status read_f32(float* dst, std::size_t count) const;

  // The stored bytes, unchanged. Half the copy and none of the widening loop
  // when the device holds the tensor in the format the checkpoint used.
  Status read_native(void* dst, std::size_t bytes) const;
};

class SafeTensors {
 public:
  SafeTensors() = default;
  ~SafeTensors();
  SafeTensors(SafeTensors&&) noexcept;
  SafeTensors& operator=(SafeTensors&&) noexcept;
  SafeTensors(const SafeTensors&) = delete;
  SafeTensors& operator=(const SafeTensors&) = delete;

  static Result<SafeTensors> open(const std::string& path);

  // A checkpoint split across shards, named by a safetensors index file. Every
  // shard the index references is mapped and the tensors are merged into one
  // namespace, so a caller cannot tell a sharded model from a single-file one.
  static Result<SafeTensors> open_sharded(const std::string& index_path);

  [[nodiscard]] const TensorView* find(std::string_view name) const;
  [[nodiscard]] const std::map<std::string, TensorView>& tensors() const noexcept {
    return tensors_;
  }
  [[nodiscard]] std::size_t total_parameters() const noexcept;
  [[nodiscard]] const std::string& path() const noexcept { return path_; }

 private:
  // Each shard stays mapped for the lifetime of the reader; TensorView::data
  // points into these, so they must not be unmapped while views are alive.
  struct Mapping {
    void* ptr = nullptr;
    std::size_t size = 0;
  };

  // Maps one safetensors file and merges its tensors into this reader.
  Status map_file(const std::string& path);
  void unmap_all() noexcept;

  std::string path_;
  std::vector<Mapping> mappings_;
  std::map<std::string, TensorView> tensors_;
};

// Old checkpoints store routed experts as `...moe.experts.{e}.w1|w2|w3.weight`;
// current ones stack them into `w_gate|w_up|w_down` of shape [E, out, in].
// Returns the stacked name plus expert index, or nullopt when `name` is
// already in the current layout.
struct LegacyExpertKey {
  std::string stacked_name;
  int expert_index = 0;
};
Result<LegacyExpertKey> migrate_legacy_expert_name(std::string_view name);

// Resolution order: existing path, then the HF cache, then a repo id.
// Mirrors what mlx_lm and llama.cpp do, so `lse run org/name` just works.
struct ModelPaths {
  std::string weights;  // .safetensors
  std::string config;   // sidecar .json
};
Result<ModelPaths> resolve_model(const std::string& name_or_path);
std::string hf_cache_root();

}  // namespace lse::model
