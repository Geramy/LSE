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
#include "lse/quant/group_affine.hpp"

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

  // total_parameters() counts *stored* elements, which on a quantized
  // checkpoint is not the parameter count: a group-affine plane packs several
  // weights into each u32 lane, so a 0.8B model sums to 0.2B. This expands each
  // packed plane by 32/bits using the geometry `quant` gives for that tensor and
  // drops the scale/bias planes, which are not parameters. A plane whose
  // geometry does not resolve is counted as stored rather than guessed at.
  [[nodiscard]] std::size_t logical_parameters(
      const quant::GroupAffineMap* quant) const noexcept;

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

// The hub cache directory, resolved the way huggingface_hub resolves it:
//
//   1. $HF_HUB_CACHE
//   2. $HUGGINGFACE_HUB_CACHE   (the legacy name, honoured but outranked)
//   3. $HF_HOME/hub, where HF_HOME defaults to $XDG_CACHE_HOME/huggingface and
//      XDG_CACHE_HOME to ~/.cache
//
// XDG_CACHE_HOME is consulted only for HF_HOME's default, never as a fallback
// for either cache variable, and `~`/`$VAR` in a value are expanded because the
// Python library expands them. TRANSFORMERS_CACHE does not appear in
// huggingface_hub at all and is deliberately not read.
std::string hf_cache_root();

// $HF_HOME, or the default derived from XDG_CACHE_HOME / HOME.
std::string hf_home();

// Every shard a safetensors index names, and which of them are absent. A repo
// directory can exist with the download unfinished, so completeness has to be
// answerable without mapping tens of gigabytes to learn a yes or no.
struct ShardIndex {
  std::vector<std::string> shards;   // file names, sorted, deduplicated
  std::vector<std::string> missing;  // the subset not present beside the index
  std::size_t named_tensors = 0;

  [[nodiscard]] bool complete() const noexcept { return missing.empty(); }
};
Result<ShardIndex> read_shard_index(const std::string& index_path);

// Whether this build can load a checkpoint. kUnknown exists so that a repo
// whose loadability could not be established is never offered as loadable —
// claiming a model loads when it does not is the failure this answers.
enum class Loadable {
  kYes,
  kNo,
  kIncomplete,  // the download did not finish; not a property of the model
  kUnknown,
};
std::string_view to_string(Loadable l) noexcept;

// What one checkpoint directory is. Every field is either read from the
// checkpoint or left empty/zero to mean "not established"; nothing here is
// inferred from a repo's name.
struct CacheModel {
  std::string repo_id;
  std::string path;  // the snapshot directory, empty when there is none

  std::string architecture;     // config.json architectures[0], else model_type
  std::string engine_arch;      // what detect_architecture matched, if anything
  std::size_t parameters = 0;   // 0 when it could not be counted
  std::string quantization;     // "affine 4-bit g64", "none", or empty
  std::uintmax_t bytes = 0;     // on disk, hard links and symlinks counted once
  bool multimodal = false;
  bool multimodal_known = false;

  Loadable loadable = Loadable::kUnknown;
  std::string reason;  // why, for every verdict other than kYes
};

// Inspects a checkpoint directory. Reads config.json and the tensor headers,
// then runs the engine's own architecture detection and the group-affine
// preconditions over them, so the verdict comes from the code that loads the
// model rather than from a second opinion about it. Never returns an error:
// "could not tell" is a verdict, not a failure. `repo_id` is only a label.
CacheModel inspect_model_dir(const std::string& dir, std::string_view repo_id);

// Every model in the hub cache, sorted by repo id, complete or not.
Result<std::vector<CacheModel>> list_cached_models();

}  // namespace lse::model
