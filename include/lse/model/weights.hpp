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

  [[nodiscard]] const TensorView* find(std::string_view name) const;
  [[nodiscard]] const std::map<std::string, TensorView>& tensors() const noexcept {
    return tensors_;
  }
  [[nodiscard]] std::size_t total_parameters() const noexcept;
  [[nodiscard]] const std::string& path() const noexcept { return path_; }

 private:
  std::string path_;
  void* mapping_ = nullptr;
  std::size_t mapping_size_ = 0;
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
