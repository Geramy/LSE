// DFlash2 BF16 -> affine Q8/group64 conversion, run once when a draft
// checkpoint is opened.
//
// Output is byte-identical to scripts/convert_dflash2_q8.py:
// model.safetensors, config.json and source-repository.json all match.
// The conversion streams the source with pread. It holds one 128-row block
// plus one tensor's scales and biases at a time, so peak memory is a few MiB
// whatever the checkpoint size. The output is assembled in a private
// temporary directory and renamed into place, so a crash leaves no partial
// cache behind.
#pragma once

#include <cstdint>
#include <filesystem>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

#include "lse/core/status.hpp"
#include "lse/model/weights.hpp"

namespace lse::model {

// What a DFlash2 checkpoint holds, read from its config and safetensors
// header rather than from its path.
enum class DFlash2CheckpointKind : std::uint8_t {
  kNotDFlash2,   // the config is not a DFlash2DraftModel
  kQuantized,    // the config carries quantization fields; load it as is
  kBF16Source,   // unquantized, single file, BF16 matrices; convert first
  kUnsupported,  // unquantized but not convertible (sharded, non-BF16 matrices)
};

[[nodiscard]] std::string_view to_string(DFlash2CheckpointKind kind) noexcept;

Result<DFlash2CheckpointKind> inspect_dflash2_checkpoint(const ModelPaths& paths);

// One tensor of the checkpoint convert_dflash2_q8 would write, as its
// safetensors header would describe it.
struct DFlash2PlannedTensor {
  std::string name;
  std::string dtype;  // safetensors spelling: "U32", "BF16", ...
  std::vector<std::int64_t> shape;
};

// The tensors the Q8 conversion of `source` would hold, read from the source's
// config and header only: no payload is read and nothing is written. Fails for
// a checkpoint the converter would refuse.
Result<std::vector<DFlash2PlannedTensor>> dflash2_q8_layout(const ModelPaths& source);

struct DFlash2ConvertProgress {
  enum class Phase : std::uint8_t { kHashSource, kQuantize, kFinish };
  Phase phase = Phase::kHashSource;
  std::uint64_t done = 0;   // bytes for this phase
  std::uint64_t total = 0;
  std::string_view tensor;  // the tensor being quantized, if any
};
using DFlash2ProgressFn = std::function<void(const DFlash2ConvertProgress&)>;

[[nodiscard]] std::string_view to_string(DFlash2ConvertProgress::Phase phase) noexcept;

struct DFlash2ConvertOptions {
  std::string source_repository;  // recorded in source-repository.json
  std::string source_revision;
  DFlash2ProgressFn progress;     // null: log to stderr instead
};

// The Python script's convert(): the destination must not exist. It is
// created atomically from a temporary sibling directory.
Status convert_dflash2_q8(const ModelPaths& source,
                          const std::filesystem::path& destination,
                          const DFlash2ConvertOptions& options);

struct DFlash2PrepareOptions {
  // Where converted checkpoints are cached. Empty means $LSE_DFLASH2_CACHE_DIR
  // when that is set, and otherwise a `lse-q8g64` directory inside the source
  // checkpoint directory.
  std::filesystem::path cache_dir;
  // Override the recorded origin. Empty means: derived from an HF cache
  // snapshot path, else from `hf-origin.json` beside the weights, else the
  // directory name with revision "unknown".
  std::string source_repository;
  std::string source_revision;
  DFlash2ProgressFn progress;
};

// Resolves `name_or_path` (directory, file or HF repo id) and returns the
// paths to load. A quantized checkpoint is returned unchanged. A BF16 source
// is converted on first use and the cached Q8 result is returned after that.
// Set LSE_DFLASH2_AUTOCONVERT=0 to load BF16 sources directly.
Result<ModelPaths> prepare_dflash2_checkpoint(const std::string& name_or_path,
                                              const DFlash2PrepareOptions& options = {});

// Where prepare_dflash2_checkpoint caches the conversion of `source`.
Result<std::filesystem::path> dflash2_cache_path(const ModelPaths& source,
                                                 const DFlash2PrepareOptions& options = {});

// The text the Python script writes, exposed for tests: json.dumps(value,
// indent=2) with ensure_ascii, and the compact separators=(",", ":") form.
Result<std::string> dflash2_python_json(std::string_view json_text, bool indent);

}  // namespace lse::model
