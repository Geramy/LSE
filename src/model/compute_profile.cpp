#include "lse/model/compute_profile.hpp"
#include "lse/model/config.hpp"
#include "lse/model/weights.hpp"
#include "lse/core/sha256.hpp"
#include "lse/kernels/int8_policy.hpp"
#include <array>
#include <filesystem>
#include <fstream>
#include <sstream>
namespace lse::model {
namespace {
constexpr std::string_view config_digest =
    "14b65a0ee06517060a6bbd979bb1a8ff54e7b304b1a1f01d54344b88b8285e85";
constexpr std::array<std::string_view, 3> shards = {
    "471a9a81b54dddbfef43e8a71960ca24ad3c727891c6b0750a9db9a214aab504",
    "e7474b8f968f4078dfd4889925961c5b0da9939219b614dba08eb63c46590de7",
    "1038989bed1346ecdb541a45c078df2aae99a9509f35bfd51e7a04802a5d8402"};
}
bool qualified_q4_digests(std::string_view config,
                          std::span<const std::string_view> actual) noexcept {
  if (config != config_digest || actual.size() != shards.size()) return false;
  for (std::size_t i = 0; i < shards.size(); ++i)
    if (actual[i] != shards[i]) return false;
  return true;
}
std::int32_t qualified_q4_compute(const Config& config, const SafeTensors& weights) {
  // Fail closed before streaming any shard for unknown architectures/geometry.
  if (config.hidden_size != 5120 || config.num_layers != 64 || config.vocab_size != 248320 ||
      config.mlp_intermediate != 17408 || !weights.path().ends_with(".index.json"))
    return 0;
  std::ifstream input(std::filesystem::path(weights.path()).parent_path() / "config.json",
                      std::ios::binary);
  if (!input) return 0;
  std::ostringstream buffer;
  buffer << input.rdbuf();
  const auto text = buffer.str();
  if (sha256(text) != config_digest) return 0;
  auto expected = Config::from_json_string(text);
  if (!expected.ok()) return 0;
  // Runtime KV reservation is allocation policy; other model changes invalidate qualification.
  auto actual = config;
  actual.kv_length = 0;
  expected->kv_length = 0;
  // This runtime flag is not serialized by Config::to_json().
  if (actual.moe_bias_balance != expected->moe_bias_balance) return 0;
  if (actual.to_json() != expected->to_json()) return 0;
  // Config::to_json omits quantization; compare every mapped plane's resolved geometry too.
  for (const auto& [name, view] : weights.tensors()) {
    if (view.dtype != DType::kU32) continue;
    if (actual.quantization.is_skipped(name) != expected->quantization.is_skipped(name)) return 0;
    const auto a = actual.quantization.resolve(name), b = expected->quantization.resolve(name);
    if (a.ok() != b.ok()) return 0;
    if (a.ok() && (a->bits != b->bits || a->group_size != b->group_size)) return 0;
  }
  if (!weights.content_sha256_matches(shards)) return 0;
  return kernels::kQwen27BQ4ComputeRevision;
}
}
