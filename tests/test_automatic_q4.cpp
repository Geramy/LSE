#include "harness.hpp"
#include "lse/core/sha256.hpp"
#include "lse/kernels/int8_policy.hpp"
#include "lse/kernels/quant_operand_cache.hpp"
#include "lse/model/compute_profile.hpp"
#include "lse/model/config.hpp"
#include "lse/model/weights.hpp"
#include "lse/backends/hrx/arch_database.hpp"
#include "lse/backends/hrx/loomc/loom_emitter.hpp"
#include "lse/backends/hrx/hipc/hip_emitter.hpp"
#include "lse/graph/ops.hpp"
#include <array>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
using namespace lse;
using namespace lse::graph;
namespace {
std::string incremental(std::string_view input, std::size_t stride) {
  Sha256 h;
  for (std::size_t i = 0; i < input.size(); i += stride)
    h.update(std::as_bytes(std::span(input.data() + i, std::min(stride, input.size() - i))));
  auto digest = h.finish();
  std::string out(64, '0');
  constexpr char hex[] = "0123456789abcdef";
  for (std::size_t i = 0; i < 32; ++i) {
    unsigned v = std::uint8_t(digest[i]);
    out[2 * i] = hex[v >> 4];
    out[2 * i + 1] = hex[v & 15];
  }
  return out;
}
Array leaf(Shape shape, DType type) {
  auto n = std::make_shared<Node>();
  n->shape = shape;
  n->dtype = type;
  return Array(n);
}
Array contraction(std::int32_t profile, int m = 1) {
  auto x = leaf({1, m, 5120}, DType::kF32), w = leaf({17408, 640}, DType::kU32),
       sc = leaf({17408, 80}, DType::kBF16), bi = leaf({17408, 80}, DType::kBF16);
  auto q = std::make_shared<QuantPlanes>();
  q->scales = sc.node();
  q->biases = bi.node();
  q->bits = 4;
  q->group_size = 64;
  q->in_features = 5120;
  q->compute_profile_revision = profile;
  w.node()->quant = q;
  return linear(x, w);
}
struct Request {
  backend::DeviceInfo device;
  backend::AmdDeviceInfo amd;
  std::array<Shape, 4> inputs{Shape{1, 1, 5120}, Shape{17408, 640}, Shape{17408, 80},
                              Shape{17408, 80}};
  std::array<DType, 4> dtypes{DType::kF32, DType::kU32, DType::kBF16, DType::kBF16};
  KernelShapes shapes;
  Request() {
    device.arch = "gfx1201";
    backend::apply_arch_defaults(device, amd);
    device.extension_id = backend::AmdDeviceInfo::kExtensionId;
    device.extension = &amd;
    shapes.inputs = inputs;
    shapes.input_dtypes = dtypes;
    shapes.output = {1, 1, 17408};
    shapes.device = &device;
    shapes.iattrs = {4, 64, kernels::kQwen27BQ4ComputeRevision, 0};
  }
};
}
LSE_TEST(sha256_known_vectors_padding_streaming_and_mutation) {
  const std::array<std::pair<std::string, std::string>, 4> vectors{
      {{"", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"},
       {"abc", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"},
       {"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
        "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"},
       {std::string(1000000, 'a'),
        "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"}}};
  for (const auto& [input, expected] : vectors) {
    LSE_EXPECT(sha256(input) == expected);
    for (auto stride : {1u, 7u, 64u, 4096u}) LSE_EXPECT(incremental(input, stride) == expected);
  }
  for (auto n : {55u, 56u, 63u, 64u, 65u, 119u, 120u, 128u}) {
    auto input = std::string(n, 'x');
    LSE_EXPECT(sha256(input) == incremental(input, 13));
    auto original = sha256(input);
    input[n / 2] = 'y';
    LSE_EXPECT(sha256(input) != original);
  }
}
LSE_TEST(qualification_requires_exact_config_and_all_shard_content) {
  const std::string_view cfg = "14b65a0ee06517060a6bbd979bb1a8ff54e7b304b1a1f01d54344b88b8285e85";
  std::array<std::string_view, 3> shards{
      "471a9a81b54dddbfef43e8a71960ca24ad3c727891c6b0750a9db9a214aab504",
      "e7474b8f968f4078dfd4889925961c5b0da9939219b614dba08eb63c46590de7",
      "1038989bed1346ecdb541a45c078df2aae99a9509f35bfd51e7a04802a5d8402"};
  LSE_EXPECT(model::qualified_q4_digests(cfg, shards));
  LSE_EXPECT(!model::qualified_q4_digests("same architecture different content", shards));
  LSE_EXPECT(!model::qualified_q4_digests(cfg, std::span(shards).first(2)));
  shards[1] = "mutated fine-tune";
  LSE_EXPECT(!model::qualified_q4_digests(cfg, shards));
}
LSE_TEST(mapped_content_digest_rejects_a_mutated_shard) {
  namespace fs = std::filesystem;
  const auto dir = fs::temp_directory_path() / "lse-q4-profile-mapped-test";
  fs::create_directories(dir);
  std::string header = R"({"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}})";
  const std::uint64_t n = header.size();
  std::string file(reinterpret_cast<const char*>(&n), sizeof(n));
  file += header;
  file.append(4, '\0');
  auto write = [&](const fs::path& path, const std::string& content) {
    std::ofstream o(path, std::ios::binary);
    o.write(content.data(), static_cast<std::streamsize>(content.size()));
  };
  write(dir / "a.safetensors", file);
  auto first = model::SafeTensors::open((dir / "a.safetensors").string());
  LSE_EXPECT(first.ok());
  const auto digest = sha256(file);
  const std::array<std::string_view, 1> expected{digest};
  if (first.ok()) {
    LSE_EXPECT(first->content_sha256_matches(expected));
    LSE_EXPECT(!first->content_sha256_matches({}));
  }
  file.back() = 1;
  write(dir / "b.safetensors", file);
  auto second = model::SafeTensors::open((dir / "b.safetensors").string());
  LSE_EXPECT(second.ok());
  if (second.ok()) LSE_EXPECT(!second->content_sha256_matches(expected));
  fs::remove_all(dir);
}
LSE_TEST(automatic_override_and_shape_scope_are_fail_closed) {
  LSE_EXPECT(model::qualified_q4_load_scope(1, false));
  LSE_EXPECT(!model::qualified_q4_load_scope(0, false));
  LSE_EXPECT(!model::qualified_q4_load_scope(2, false));
  LSE_EXPECT(!model::qualified_q4_load_scope(1, true));
  using P = kernels::ActivationInt8Policy;
  LSE_EXPECT(kernels::parse_activation_int8_policy(nullptr) == P::kAutomatic);
  for (auto text : {"0", "", "yes", "true", " 1", "2"})
    LSE_EXPECT(kernels::parse_activation_int8_policy(text) == P::kExact);
  LSE_EXPECT(kernels::parse_activation_int8_policy("1") == P::kEnabled);
  Request r;
  LSE_EXPECT(kernels::activation_int8_enabled(r.shapes, P::kAutomatic));
  r.inputs[0] = {1, 5120};
  LSE_EXPECT(kernels::activation_int8_enabled(r.shapes, P::kAutomatic));
  r.inputs[0] = {5120};
  LSE_EXPECT(!kernels::activation_int8_enabled(r.shapes, P::kAutomatic));
  r.inputs[0] = {1, 1, 1, 5120};
  LSE_EXPECT(!kernels::activation_int8_enabled(r.shapes, P::kAutomatic));
  r.inputs[0] = {1, 1, 5120};
  LSE_EXPECT(!kernels::activation_int8_enabled(r.shapes, P::kExact));
  r.shapes.iattrs[2] = 0;
  LSE_EXPECT(!kernels::activation_int8_enabled(r.shapes, P::kAutomatic));
  LSE_EXPECT(kernels::activation_int8_enabled(r.shapes, P::kEnabled));
  r.shapes.iattrs[2] = 1;
  for (int m : {2, 16, 64, 256, 1024}) {
    r.inputs[0] = {1, m, 5120};
    r.shapes.output = {1, m, 17408};
    LSE_EXPECT(!kernels::activation_int8_enabled(r.shapes, P::kAutomatic));
  }
  r.inputs[0] = {1, 512, 5120};
  r.shapes.output = {1, 512, 17408};
  LSE_EXPECT(kernels::activation_int8_enabled(r.shapes, P::kAutomatic));
  r.device.arch = "gfx1200";
  LSE_EXPECT(!kernels::activation_int8_enabled(r.shapes, P::kAutomatic));
  r.device.arch = "gfx1201";
  r.device.wavefront_size = 64;
  LSE_EXPECT(!kernels::activation_int8_enabled(r.shapes, P::kAutomatic));
  r.device.wavefront_size = 32;
  r.shapes.output_dtype = DType::kF16;
  LSE_EXPECT(!kernels::activation_int8_enabled(r.shapes, P::kAutomatic));
  r.shapes.output_dtype = DType::kF32;
  r.dtypes[3] = DType::kF16;
  LSE_EXPECT(!kernels::activation_int8_enabled(r.shapes, P::kAutomatic));
  r.dtypes[3] = DType::kBF16;
  r.dtypes[0] = DType::kF16;
  LSE_EXPECT(!kernels::activation_int8_enabled(r.shapes, P::kAutomatic));
  r.dtypes[0] = DType::kF32;
  r.dtypes[2] = DType::kF16;
  LSE_EXPECT(!kernels::activation_int8_enabled(r.shapes, P::kAutomatic));
  r.dtypes[2] = DType::kBF16;
  r.shapes.iattrs[0] = 6;
  LSE_EXPECT(!kernels::activation_int8_enabled(r.shapes, P::kAutomatic));
  r.shapes.iattrs[0] = 4;
  r.shapes.iattrs[1] = 128;
  LSE_EXPECT(!kernels::activation_int8_enabled(r.shapes, P::kAutomatic));
  r.shapes.iattrs[1] = 64;
  r.shapes.staged.name = "caller-owned";
  LSE_EXPECT(!kernels::activation_int8_enabled(r.shapes, P::kAutomatic));
  r.shapes.staged.name = {};
  r.inputs[0] = {2, 256, 5120};
  LSE_EXPECT(!kernels::activation_int8_enabled(r.shapes, P::kAutomatic));
}
LSE_TEST(metadata_propagation_mixed_models_and_cache_domains) {
  Request r;
  backend::LoomEmitter loom;
  backend::HipEmitter hip;
  auto qualified = contraction(1), unknown = contraction(0), another = contraction(1);
  LSE_EXPECT(qualified.node()->iattrs[2] == 1 && unknown.node()->iattrs[2] == 0);
  auto group = [](const Array& a) {
    const NodePtr roots[]{a.node()};
    auto groups = Partitioner::partition(roots);
    for (const auto& g : groups)
      if (g.anchor == OpKind::kQuantMatMul) return g;
    return groups.back();
  };
  auto a = group(qualified), b = group(unknown), c = group(another);
  auto la = loom.cache_key(a, r.device), lb = loom.cache_key(b, r.device),
       lc = loom.cache_key(c, r.device);
  auto ha = hip.cache_key(a, r.device), hb = hip.cache_key(b, r.device),
       hc = hip.cache_key(c, r.device);
  LSE_EXPECT(la != lb && la == lc && ha != hb && ha == hc);
  auto unknown_code = loom.emit(b, r.device);
  LSE_EXPECT(unknown_code.ok());
  if (unknown_code.ok() &&
      kernels::activation_int8_policy() != kernels::ActivationInt8Policy::kEnabled)
    LSE_EXPECT(unknown_code->source.find("vector.dot4i") == std::string::npos &&
               unknown_code->source.find("vector.mma") == std::string::npos);
  auto emitted = loom.emit(a, r.device);
  LSE_EXPECT(emitted.ok());
  if (!emitted.ok()) std::fprintf(stderr, "EMIT %s\n", emitted.status().to_string().c_str());
  bool expected_dot = kernels::activation_int8_policy() != kernels::ActivationInt8Policy::kExact &&
                      !kernels::q4_decode_exact_enabled();
  if (emitted.ok())
    LSE_EXPECT((emitted->source.find("vector.dot4i") != std::string::npos) == expected_dot);
  auto mid = contraction(1, 256);
  auto mid_group = group(mid);
  auto mid_code = loom.emit(mid_group, r.device);
  LSE_EXPECT(mid_code.ok());
  if (!mid_code.ok()) std::fprintf(stderr, "MID %s\n", mid_code.status().to_string().c_str());
  if (mid_code.ok() &&
      kernels::activation_int8_policy() == kernels::ActivationInt8Policy::kAutomatic)
    LSE_EXPECT(mid_code->source.find("vector.mma") == std::string::npos &&
               mid_code->source.find("vector.dot4i") == std::string::npos);
  auto pp = contraction(1, 512);
  auto pp_code = loom.emit(group(pp), r.device);
  LSE_EXPECT(pp_code.ok());
  if (!pp_code.ok()) std::fprintf(stderr, "PP %s\n", pp_code.status().to_string().c_str());
  if (pp_code.ok())
    LSE_EXPECT((pp_code->source.find("vector.mma") != std::string::npos) ==
               (kernels::activation_int8_policy() != kernels::ActivationInt8Policy::kExact));
  auto packed = Array(qualified.node()->inputs[1]);
  auto q = packed.node()->quant;
  auto replacement = leaf(q->scales->shape, DType::kBF16);
  auto altered = quant_linear(Array(qualified.node()->inputs[0]), packed, replacement,
                              Array(q->biases), 4, 64);
  LSE_EXPECT(altered.node()->iattrs[2] == 0);
  auto ids = leaf({1, 1}, DType::kI32);
  auto emb = embedding(packed, ids);
  LSE_EXPECT(emb.node()->kind == OpKind::kQuantEmbedding && emb.node()->iattrs[2] == 0);
  auto idx = leaf({1, 1}, DType::kF32), expert = leaf({2, 17408, 640}, DType::kU32);
  auto eq = std::make_shared<QuantPlanes>(*q);
  eq->scales = leaf({2, 17408, 80}, DType::kBF16).node();
  eq->biases = leaf({2, 17408, 80}, DType::kBF16).node();
  expert.node()->quant = eq;
  auto routed = linear_indexed(Array(qualified.node()->inputs[0]), expert, idx, 0);
  LSE_EXPECT(routed.node()->kind == OpKind::kMoEDispatch && routed.node()->iattrs[1] == 4 &&
             routed.node()->iattrs[2] == 64 && routed.node()->iattrs[3] == 0);
  std::printf("CACHE policy=%u loom=%llu hip=%llu unknown_loom=%llu exact_override=%u\n",
              unsigned(kernels::activation_int8_policy()), (unsigned long long)la,
              (unsigned long long)ha, (unsigned long long)lb,
              unsigned(kernels::q4_decode_exact_enabled()));
}
LSE_TEST_MAIN()
