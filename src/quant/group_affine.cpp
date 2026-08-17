#include "lse/quant/group_affine.hpp"

#include <algorithm>

#include <nlohmann/json.hpp>

namespace lse::quant {

namespace {

bool legal_bits(int b) noexcept {
  return std::find(std::begin(kGroupAffineBits), std::end(kGroupAffineBits),
                   b) != std::end(kGroupAffineBits);
}

bool legal_group(int g) noexcept {
  return std::find(std::begin(kGroupAffineGroupSizes),
                   std::end(kGroupAffineGroupSizes),
                   g) != std::end(kGroupAffineGroupSizes);
}

std::string bits_list() {
  std::string out;
  for (int b : kGroupAffineBits) {
    if (!out.empty()) out += ", ";
    out += std::to_string(b);
  }
  return out;
}

constexpr std::string_view kAffine = "affine";

// mxfp4 / nvfp4 / mxfp8 exist upstream and encode their scales completely
// differently; decoding one of them as affine produces plausible garbage.
Status check_mode(const nlohmann::json& j, std::string_view where) {
  if (!j.contains("mode")) return OkStatus();
  if (!j["mode"].is_string()) {
    return LSE_ERROR(kInvalidArgument, "quantization mode of ", where,
                     " is not a string");
  }
  const auto mode = j["mode"].get<std::string>();
  if (mode != kAffine) {
    return LSE_ERROR(kUnimplemented, "quantization mode '", mode, "' of ",
                     where, " is not supported; only '", kAffine, "' is");
  }
  return OkStatus();
}

Status read_int(const nlohmann::json& j, const char* key, std::string_view where,
                int& out) {
  if (!j.contains(key)) return OkStatus();
  if (!j[key].is_number_integer()) {
    return LSE_ERROR(kInvalidArgument, "quantization field '", key, "' of ",
                     where, " is not an integer");
  }
  out = j[key].get<int>();
  return OkStatus();
}

}  // namespace

Result<GroupAffine> GroupAffine::make(int bits, int group_size) {
  if (!legal_bits(bits)) {
    return LSE_ERROR(kUnimplemented, "group-affine bit width ",
                     std::to_string(bits), " is not one of ", bits_list());
  }
  if (!legal_group(group_size)) {
    return LSE_ERROR(kUnimplemented, "group-affine group size ",
                     std::to_string(group_size), " is not one of 32, 64, 128");
  }
  return GroupAffine{bits, group_size};
}

Status GroupAffine::check_row(std::size_t k) const {
  if (k == 0 || k % static_cast<std::size_t>(group_size) != 0) {
    return LSE_ERROR(kInvalidArgument, "a row of ", std::to_string(k),
                     " weights is not a whole number of ",
                     std::to_string(group_size), "-element groups");
  }
  if ((k * static_cast<std::size_t>(bits)) % 32 != 0) {
    return LSE_ERROR(kInvalidArgument, "a row of ", std::to_string(k),
                     " weights at ", std::to_string(bits),
                     " bits does not fill a whole number of 32-bit lanes");
  }
  return OkStatus();
}

Result<int> GroupAffine::bits_from_shapes(std::int64_t packed_last,
                                          std::int64_t scales_last,
                                          int group_size) {
  if (packed_last <= 0 || scales_last <= 0) {
    return LSE_ERROR(kInvalidArgument,
                     "packed and scale planes must both have a positive last "
                     "dimension, got ",
                     std::to_string(packed_last), " and ",
                     std::to_string(scales_last));
  }
  if (!legal_group(group_size)) {
    return LSE_ERROR(kUnimplemented, "group-affine group size ",
                     std::to_string(group_size), " is not one of 32, 64, 128");
  }
  const std::int64_t num = packed_last * 32;
  const std::int64_t den = scales_last * static_cast<std::int64_t>(group_size);
  if (num % den != 0) {
    return LSE_ERROR(kInvalidArgument, "packed plane of ",
                     std::to_string(packed_last), " lanes and ",
                     std::to_string(scales_last), " groups of ",
                     std::to_string(group_size),
                     " do not solve to a whole bit width");
  }
  const auto bits = static_cast<int>(num / den);
  if (!legal_bits(bits)) {
    return LSE_ERROR(kUnimplemented, "shapes solve to ", std::to_string(bits),
                     " bits, which is not one of ", bits_list());
  }
  return bits;
}

std::string_view GroupAffineMap::module_path_of(
    std::string_view tensor_name) noexcept {
  for (std::string_view leaf : {".weight", ".scales", ".biases"}) {
    if (tensor_name.size() > leaf.size() &&
        tensor_name.substr(tensor_name.size() - leaf.size()) == leaf) {
      return tensor_name.substr(0, tensor_name.size() - leaf.size());
    }
  }
  return tensor_name;
}

Result<GroupAffineMap> GroupAffineMap::from_config_json(
    std::string_view config_json) {
  nlohmann::json j;
  try {
    j = nlohmann::json::parse(config_json);
  } catch (const std::exception& e) {
    return LSE_ERROR(kInvalidArgument, "model config is not valid JSON: ",
                     e.what());
  }

  const nlohmann::json* block = nullptr;
  if (j.contains("quantization") && j["quantization"].is_object()) {
    block = &j["quantization"];
  } else if (j.contains("quantization_config") &&
             j["quantization_config"].is_object()) {
    block = &j["quantization_config"];
  }

  GroupAffineMap out;
  if (block == nullptr) return out;

  LSE_RETURN_IF_ERROR(check_mode(*block, "the checkpoint"));

  // Two passes: nlohmann sorts object keys, so a module path can precede the
  // global scalars an override inherits from.
  int global_bits = 0;
  int global_group = 0;
  LSE_RETURN_IF_ERROR(read_int(*block, "bits", "the checkpoint", global_bits));
  LSE_RETURN_IF_ERROR(
      read_int(*block, "group_size", "the checkpoint", global_group));
  if ((global_bits != 0) != (global_group != 0)) {
    return LSE_ERROR(kInvalidArgument,
                     "the checkpoint's quantization block names only one of "
                     "'bits' and 'group_size'; both are needed or neither");
  }
  if (global_bits != 0) {
    LSE_ASSIGN_OR(out.global_, GroupAffine::make(global_bits, global_group));
    out.has_global_ = true;
  }

  for (const auto& [key, value] : block->items()) {
    if (key == "bits" || key == "group_size" || key == "mode") continue;
    if (value.is_boolean()) {
      // `true` means "quantized with the global geometry", which is what
      // resolve() already answers; only the refusal needs recording.
      if (!value.get<bool>()) out.skipped_.emplace_back(key);
      continue;
    }
    if (!value.is_object()) {
      return LSE_ERROR(kInvalidArgument, "quantization entry '", key,
                       "' is neither a per-module override object nor a "
                       "true/false skip flag");
    }
    LSE_RETURN_IF_ERROR(check_mode(value, key));
    int bits = global_bits;
    int group = global_group;
    LSE_RETURN_IF_ERROR(read_int(value, "bits", key, bits));
    LSE_RETURN_IF_ERROR(read_int(value, "group_size", key, group));
    if (bits == 0 || group == 0) {
      return LSE_ERROR(kNotFound, "quantization override '", key,
                       "' does not name both 'bits' and 'group_size' and the "
                       "checkpoint has no global pair to inherit from");
    }
    LSE_ASSIGN_OR(GroupAffine spec, GroupAffine::make(bits, group));
    out.overrides_.emplace_back(key, spec);
  }
  return out;
}

bool GroupAffineMap::is_skipped(std::string_view tensor_name) const noexcept {
  const std::string_view path = module_path_of(tensor_name);
  return std::find(skipped_.begin(), skipped_.end(), path) != skipped_.end();
}

Result<GroupAffine> GroupAffineMap::resolve(
    std::string_view tensor_name) const {
  const std::string_view path = module_path_of(tensor_name);
  for (const auto& [name, spec] : overrides_) {
    if (name == path) return spec;
  }
  if (is_skipped(tensor_name)) {
    return LSE_ERROR(kNotFound, "'", path,
                     "' is marked unquantized in the checkpoint's "
                     "quantization block");
  }
  if (!has_global_) {
    return LSE_ERROR(kNotFound, "'", path,
                     "' has no quantization geometry: the config names no "
                     "global bits/group_size and no override for it");
  }
  return global_;
}

Result<GroupAffine> GroupAffineMap::resolve_checked(
    std::string_view tensor_name, std::int64_t packed_last,
    std::int64_t scales_last) const {
  LSE_ASSIGN_OR(GroupAffine spec, resolve(tensor_name));
  LSE_ASSIGN_OR(const int from_shapes,
                GroupAffine::bits_from_shapes(packed_last, scales_last,
                                              spec.group_size));
  if (from_shapes != spec.bits) {
    return LSE_ERROR(kInvalidArgument, "'", tensor_name, "' is configured as ",
                     std::to_string(spec.bits), "-bit but its shapes (",
                     std::to_string(packed_last), " lanes, ",
                     std::to_string(scales_last), " groups of ",
                     std::to_string(spec.group_size), ") say ",
                     std::to_string(from_shapes), "-bit");
  }
  return spec;
}

}  // namespace lse::quant
