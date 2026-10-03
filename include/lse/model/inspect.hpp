// What a checkpoint is, and what loading it would allocate, answered from its
// config and safetensors headers alone: no tensor payload is read, no device
// is opened and nothing is allocated on one.
//
// Both answers come from the loader's own rules rather than from a second
// opinion: the architecture is the registry's detection over the tensor
// names, the KV layers are Config::is_attention_layer, KV bytes follow
// kv/sizing.hpp (which the paged allocator itself uses), and weights are
// packed into slabs the way WeightBinder packs them. What cannot be stated
// exactly without running the model -- graph workspaces -- is modelled and
// labelled as such.
#pragma once

#include <cstdint>
#include <optional>
#include <string>

#include <nlohmann/json.hpp>

#include "lse/core/status.hpp"
#include "lse/kv/cache_dtype.hpp"

namespace lse::model {

// A model directory, .safetensors file or HF repo id, described as JSON. A
// target model, an MTP module and a DFlash2 draft are all accepted; "kind"
// says which. A checkpoint this build cannot load is still described, with
// "loadable": false and the reason. Errors are reserved for a path that does
// not resolve to a checkpoint at all.
Result<nlohmann::json> model_info(const std::string& name_or_path);

enum class DraftKind : std::uint8_t { kNone, kMtp, kDFlash2 };

struct MemoryPlanRequest {
  std::string model;
  DraftKind draft = DraftKind::kNone;
  // The MTP module or the DFlash2 checkpoint. For kMtp, empty means the
  // module found beside the model the way lse_open finds it.
  std::string draft_path;
  std::uint32_t mtp_depth = 3;
  // Unset: the format the model's config selects.
  std::optional<kv::CacheDType> kv_cache_dtype;
  // Tokens per sequence the KV pools may reach. 0: the model's default.
  std::int32_t kv_len = 0;
  // Tokens held at the moment being estimated. 0: kv_len, the full pool.
  std::int32_t context_tokens = 0;
  std::uint32_t batch_size = 1024;
  std::uint32_t ubatch_size = 1024;
  std::int32_t sequences = 1;
  // Loom keeps K/V in 256 KiB fragments; HIP keeps contiguous pools.
  bool fragmented_kv = true;
  // The device the plan is for. Empty: unknown, and the packed Q8 weight
  // copies a gfx1201-class device adds are assumed. Any other name that
  // runs no packed Q8 kernels (e.g. "cpu") leaves them out.
  std::string device_arch;
  // Nonzero: also report the largest kv_len whose estimate fits this many
  // bytes of device memory.
  std::uint64_t device_memory_bytes = 0;
};

// The bytes a load with these settings would allocate, by component.
Result<nlohmann::json> estimate_memory(const MemoryPlanRequest& request);

}  // namespace lse::model
