// safetensors reader. mmaps the file and hands out zero-copy views; nothing is
// materialized until a tensor is uploaded to a device.
#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <string_view>
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
  // Where `data` lives in its file: a descriptor the reader keeps open for
  // as long as it does, and the byte offset of the first element. Weight
  // loading reads through these rather than through the mapping (see
  // read_file below). -1 when the view is not backed by a file.
  int fd = -1;
  std::uint64_t file_offset = 0;

  [[nodiscard]] std::size_t element_count() const noexcept { return shape.elem_count(); }

  // Widens to f32 regardless of stored dtype.
  Status read_f32(float* dst, std::size_t count) const;

  // The stored bytes, unchanged. Half the copy and none of the widening loop
  // when the device holds the tensor in the format the checkpoint used.
  Status read_native(void* dst, std::size_t bytes) const;

  // `bytes` of the stored bytes starting `offset` bytes in, read from the
  // file with positioned reads instead of by faulting the mapping in. Large
  // reads run at the drive's rate (11.6 GB/s cold on an M5 Max) where the
  // mapping faults at 1.4 GB/s, and they leave the page cache alone.
  Status read_file(void* dst, std::size_t bytes, std::size_t offset = 0) const;
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
    // Open for positioned reads (TensorView::fd) until the reader is gone.
    int fd = -1;
  };

  // Maps one safetensors file and merges its tensors into this reader.
  Status map_file(const std::string& path);
 public:
  // Tells the kernel the mapped pages are not needed now. The mapping stays
  // valid (a later read faults the bytes back in from the file); its
  // resident pages are dropped. What a loaded engine calls once every
  // weight is on the device. Returns the bytes mapped.
  std::size_t release_pages() const noexcept;
 private:
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

// Resolution order: an existing path, then a repo id in the local HF cache —
// "org/name", "org/name@revision" (a ref such as main, a commit, or a unique
// commit prefix of 7+ hex digits), or a bare name that picks out one repo.
// Never touches the network: a repo that is not cached is an error that says
// how to pull it.
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

// "models--org--name": the directory a repo id occupies under the cache root.
std::string repo_cache_dir_name(std::string_view repo_id);

// The snapshot directory a cached repo resolves to at `revision` (empty: the
// commit refs/main names, or the only snapshot). Offline.
Result<std::string> cached_snapshot(const std::string& repo_id,
                                    const std::string& revision = {});

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
