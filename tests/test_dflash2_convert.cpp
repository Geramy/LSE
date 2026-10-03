// DFlash2 BF16 -> Q8/group64 conversion must reproduce
// scripts/convert_dflash2_q8.py byte for byte.
//
// The synthetic checkpoint below covers the awkward cases: all-zero, signed
// zero, constant, subnormal, huge and one-sided groups, codebook renames,
// passthrough tensors, row blocks over 128, and config JSON that needs
// Python's float repr and ensure_ascii. Its expected hashes come from running
// the Python script on the same bytes (LSE_DFLASH2_KEEP_FIXTURE=<dir> writes
// the fixture out for that).
//
// The full checkpoint test runs when LSE_DFLASH2_CONVERT_SOURCE names the
// incoai/Qwen3.8-27B-DFlash2 snapshot. It needs about 2 GB free.
// LSE_DFLASH2_CONVERT_REFERENCE may name a directory the Python script
// produced; its three files are then compared byte for byte too.
#include "lse/model/dflash2_convert.hpp"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <unistd.h>

#include "harness.hpp"
#include "lse/core/sha256.hpp"

using namespace lse;
using namespace lse::model;
namespace fs = std::filesystem;

namespace {

struct Scratch {
  fs::path path;
  Scratch() {
    std::string pattern = (fs::temp_directory_path() / "lse-dflash2-convert.XXXXXX").string();
    path = ::mkdtemp(pattern.data());
  }
  ~Scratch() {
    std::error_code ec;
    fs::remove_all(path, ec);
  }
};

std::string slurp(const fs::path& p) {
  std::ifstream in(p, std::ios::binary);
  std::ostringstream s;
  s << in.rdbuf();
  return s.str();
}

void spit(const fs::path& p, const std::string& text) {
  std::ofstream out(p, std::ios::binary);
  out << text;
}

std::string file_sha(const fs::path& p) {
  const std::string bytes = slurp(p);
  return sha256(std::string_view(bytes));
}

struct Rng {
  std::uint64_t state;
  std::uint32_t next() {
    state = state * 6364136223846793005ull + 1442695040888963407ull;
    return static_cast<std::uint32_t>(state >> 33);
  }
  // A finite BF16 with a random sign, exponent in [lo, hi] and mantissa.
  std::uint16_t bf16(std::uint32_t lo, std::uint32_t hi) {
    const std::uint32_t sign = next() & 1u;
    const std::uint32_t exponent = lo + next() % (hi - lo + 1);
    const std::uint32_t mantissa = next() & 0x7fu;
    return static_cast<std::uint16_t>(sign << 15 | exponent << 7 | mantissa);
  }
};

constexpr std::uint16_t kOne5 = 0x3fc0, kMinus225 = 0xc010, kHundred = 0x42c8,
                        kMinusHundred = 0xc2c8, kThree = 0x4040, kNegZero = 0x8000;

std::vector<std::uint16_t> edge_groups(Rng& rng) {
  std::vector<std::uint16_t> v;
  const auto group = [&](auto fill) {
    for (int i = 0; i < 64; ++i) v.push_back(fill(i));
  };
  group([](int) -> std::uint16_t { return 0; });
  group([](int i) -> std::uint16_t { return i % 2 ? kNegZero : 0; });
  group([](int) { return kOne5; });
  group([](int) { return kMinus225; });
  group([&](int) { return static_cast<std::uint16_t>((rng.next() & 1u) << 15 | (rng.next() & 0x7fu)); });
  group([&](int) { return rng.bf16(240, 254); });
  group([&](int i) { return i == 17 ? kMinusHundred : rng.bf16(100, 125); });
  group([&](int i) { return i == 40 ? kHundred : rng.bf16(100, 125); });
  group([](int i) {  // the integers 0..63
    const float f = static_cast<float>(i);
    std::uint32_t bits;
    std::memcpy(&bits, &f, 4);
    return static_cast<std::uint16_t>(bits >> 16);
  });
  group([&](int i) { return static_cast<std::uint16_t>((i % 2 ? 0x8000u : 0u) | (100u << 7) | (rng.next() & 0x7fu)); });
  group([](int i) -> std::uint16_t { return i == 5 ? kThree : 0; });
  group([&](int) { return rng.bf16(1, 254); });
  return v;
}

struct Tensor {
  std::string name, dtype;
  std::vector<std::int64_t> shape;
  std::vector<std::uint16_t> data;
};

std::string shape_json(const std::vector<std::int64_t>& shape) {
  std::string s = "[";
  for (std::size_t i = 0; i < shape.size(); ++i) s += (i ? ", " : "") + std::to_string(shape[i]);
  return s + "]";
}

void write_safetensors(const fs::path& path, const std::vector<Tensor>& tensors) {
  std::string header = "{\"__metadata__\": {\"format\": \"pt\"}";
  std::uint64_t offset = 0;
  for (const auto& t : tensors) {
    const auto bytes = t.data.size() * 2;
    header += ", \"" + t.name + "\": {\"dtype\": \"" + t.dtype + "\", \"shape\": " + shape_json(t.shape) +
              ", \"data_offsets\": [" + std::to_string(offset) + ", " + std::to_string(offset + bytes) + "]}";
    offset += bytes;
  }
  header += "}";
  header.append((8 - header.size() % 8) % 8, ' ');
  std::ofstream out(path, std::ios::binary);
  const std::uint64_t size = header.size();
  out.write(reinterpret_cast<const char*>(&size), 8);
  out << header;
  for (const auto& t : tensors)
    out.write(reinterpret_cast<const char*>(t.data.data()), static_cast<std::streamsize>(t.data.size() * 2));
}

const char* const kFixtureConfig = R"({
  "architectures": ["DFlash2DraftModel"],
  "attention_dropout": 0.0,
  "bos_token_id": null,
  "dflash_config": {"block_size": 8, "target_layer_ids": [1, 3], "empty_list": [], "empty_map": {}},
  "rms_norm_eps": 1e-6,
  "initializer_range": 0.02,
  "big": 1e16, "almost_big": 1e15, "small": 0.0001, "smaller": 0.00001, "neg": -3,
  "pi": 3.141592653589793, "third": 0.3333333333333333, "exp": 1.5E300, "half_ulp": 2.5e-308,
  "rope_parameters": {"rope_theta": 10000000, "rope_type": "default"},
  "name": "naïve – 😀 \"quoted\" back\\slash\ttab\u007f",
  "use_cache": true, "tie_word_embeddings": false,
  "quantization": null
})";

void write_fixture(const fs::path& dir) {
  fs::create_directories(dir);
  Rng rng{0x5eed};
  std::vector<Tensor> tensors;
  const auto random = [&](std::string name, std::vector<std::int64_t> shape, std::uint32_t lo, std::uint32_t hi) {
    Tensor t{std::move(name), "BF16", std::move(shape), {}};
    std::int64_t n = 1;
    for (auto d : t.shape) n *= d;
    for (std::int64_t i = 0; i < n; ++i) t.data.push_back(rng.bf16(lo, hi));
    tensors.push_back(std::move(t));
  };
  random("fc.weight", {8, 128}, 110, 130);
  random("norm.weight", {128}, 120, 128);
  random("layers.0.attention_conv.base_kernel", {2, 2, 64}, 115, 128);
  random("candidate_selector.predecessor_codebook", {130, 64}, 100, 135);
  tensors.push_back({"layers.0.mlp.edge.weight", "BF16", {4, 192}, edge_groups(rng)});
  random("some.bias", {4, 64}, 110, 125);
  random("candidate_selector.successor_codebook", {3, 128}, 90, 140);
  write_safetensors(dir / "model.safetensors", tensors);
  spit(dir / "config.json", kFixtureConfig);
}

// From scripts/convert_dflash2_q8.py on the fixture, with
// --source-repository test/fixture --source-revision 0123abcd.
constexpr const char* kFixtureWeightsSha = "b30b5547e8107643de122bdb63d0625fd3cece42c50b0aeae97e02e01d6db390";
constexpr const char* kFixtureConfigSha = "75328b6bf98cb0ace3652d09e55f51a6a13db5ed0e9371d74eba2118db2c9a2e";
constexpr const char* kFixtureManifestSha = "e42d7f9523fdb8520a024a9031b072c50103cbae3219c90306c3d522be9c2e4e";

}  // namespace

LSE_TEST(python_json_matches_json_dumps) {
  // Expected strings are Python 3's json.dumps output for the same input.
  auto compact = dflash2_python_json(R"({"b": [1, 2.5, -0.0, 1e-7, 1e16, 123456789012345678], "a": {"x": "é\n"}, "e": [], "o": {}})", false);
  LSE_EXPECT(compact.ok());
  LSE_EXPECT(*compact == R"({"b":[1,2.5,-0.0,1e-07,1e+16,123456789012345678],"a":{"x":"\u00e9\n"},"e":[],"o":{}})");
  auto indented = dflash2_python_json(R"({"k": [1, {"z": null}], "f": 100.0, "t": true, "e": []})", true);
  LSE_EXPECT(indented.ok());
  LSE_EXPECT(*indented == "{\n  \"k\": [\n    1,\n    {\n      \"z\": null\n    }\n  ],\n  \"f\": 100.0,\n  \"t\": true,\n  \"e\": []\n}");
}

LSE_TEST(fixture_converts_byte_identically) {
  Scratch scratch;
  const fs::path source = scratch.path / "source";
  write_fixture(source);
  if (const char* keep = std::getenv("LSE_DFLASH2_KEEP_FIXTURE")) write_fixture(keep);
  const ModelPaths paths{(source / "model.safetensors").string(), (source / "config.json").string()};
  auto kind = inspect_dflash2_checkpoint(paths);
  LSE_EXPECT(kind.ok() && *kind == DFlash2CheckpointKind::kBF16Source);

  const fs::path out = scratch.path / "q8";
  DFlash2ConvertOptions options{"test/fixture", "0123abcd", [](const DFlash2ConvertProgress&) {}};
  LSE_EXPECT_OK(convert_dflash2_q8(paths, out, options));
  const auto weights_sha = file_sha(out / "model.safetensors");
  const auto config_sha = file_sha(out / "config.json");
  const auto manifest_sha = file_sha(out / "source-repository.json");
  std::printf("    fixture model.safetensors %s\n    fixture config.json %s\n    fixture source-repository.json %s\n",
              weights_sha.c_str(), config_sha.c_str(), manifest_sha.c_str());
  LSE_EXPECT(weights_sha == kFixtureWeightsSha);
  LSE_EXPECT(config_sha == kFixtureConfigSha);
  LSE_EXPECT(manifest_sha == kFixtureManifestSha);

  const ModelPaths converted{(out / "model.safetensors").string(), (out / "config.json").string()};
  auto converted_kind = inspect_dflash2_checkpoint(converted);
  LSE_EXPECT(converted_kind.ok() && *converted_kind == DFlash2CheckpointKind::kQuantized);
  // Like the script, refuse to write over an existing destination.
  LSE_EXPECT(!convert_dflash2_q8(paths, out, options).ok());
}

LSE_TEST(non_dflash2_and_sharded_checkpoints_are_left_alone) {
  Scratch scratch;
  write_fixture(scratch.path);
  spit(scratch.path / "config.json", R"({"architectures": ["Qwen3ForCausalLM"]})");
  auto kind = inspect_dflash2_checkpoint({(scratch.path / "model.safetensors").string(),
                                          (scratch.path / "config.json").string()});
  LSE_EXPECT(kind.ok() && *kind == DFlash2CheckpointKind::kNotDFlash2);
  spit(scratch.path / "config.json", R"({"architectures": ["DFlash2DraftModel"]})");
  kind = inspect_dflash2_checkpoint({(scratch.path / "model.safetensors.index.json").string(),
                                     (scratch.path / "config.json").string()});
  LSE_EXPECT(kind.ok() && *kind == DFlash2CheckpointKind::kUnsupported);
}

LSE_TEST(prepare_converts_once_and_reuses_the_cache) {
  Scratch scratch;
  const fs::path source = scratch.path / "Qwen-DFlash2";
  write_fixture(source);
  spit(source / "hf-origin.json", R"({"repository": "test/fixture", "revision": "0123abcd"})");
  int quantize_calls = 0;
  DFlash2PrepareOptions options;
  options.progress = [&](const DFlash2ConvertProgress& p) {
    quantize_calls += p.phase == DFlash2ConvertProgress::Phase::kQuantize;
  };
  auto first = prepare_dflash2_checkpoint(source.string(), options);
  LSE_EXPECT(first.ok());
  if (!first.ok()) return;
  LSE_EXPECT(fs::path(first->weights) == source / "lse-q8g64" / "model.safetensors");
  LSE_EXPECT(file_sha(first->weights) == kFixtureWeightsSha);
  // hf-origin.json supplied the same origin as the fixture hash above.
  LSE_EXPECT(file_sha(source / "lse-q8g64" / "source-repository.json") == kFixtureManifestSha);
  LSE_EXPECT(quantize_calls > 0);

  quantize_calls = 0;
  auto second = prepare_dflash2_checkpoint(source.string(), options);
  LSE_EXPECT(second.ok() && second->weights == first->weights);
  LSE_EXPECT_EQ(quantize_calls, 0);

  // Pointing at the converted directory loads it directly.
  auto direct = prepare_dflash2_checkpoint((source / "lse-q8g64").string(), options);
  LSE_EXPECT(direct.ok() && fs::path(direct->weights) == source / "lse-q8g64" / "model.safetensors");
  LSE_EXPECT_EQ(quantize_calls, 0);

  // Changed source weights invalidate the cache.
  {
    std::fstream f(source / "model.safetensors", std::ios::in | std::ios::out | std::ios::binary);
    f.seekp(-2, std::ios::end);
    f.write("\x00\x47", 2);  // 32768.0 widens its group
  }
  auto third = prepare_dflash2_checkpoint(source.string(), options);
  LSE_EXPECT(third.ok());
  LSE_EXPECT(quantize_calls > 0);
  LSE_EXPECT(file_sha(source / "lse-q8g64" / "model.safetensors") != kFixtureWeightsSha);
}

LSE_TEST(prepare_honours_a_cache_directory) {
  Scratch scratch;
  const fs::path source = scratch.path / "src";
  write_fixture(source);
  DFlash2PrepareOptions options;
  options.cache_dir = scratch.path / "cache";
  options.source_repository = "test/fixture";
  options.source_revision = "0123abcd";
  options.progress = [](const DFlash2ConvertProgress&) {};
  auto path = dflash2_cache_path({(source / "model.safetensors").string(), (source / "config.json").string()}, options);
  LSE_EXPECT(path.ok() && *path == scratch.path / "cache" / "test-fixture@0123abcd-q8g64");
  auto prepared = prepare_dflash2_checkpoint(source.string(), options);
  LSE_EXPECT(prepared.ok());
  if (!prepared.ok()) return;
  LSE_EXPECT(fs::path(prepared->weights).parent_path() == *path);
  LSE_EXPECT(file_sha(prepared->weights) == kFixtureWeightsSha);
  LSE_EXPECT(!fs::exists(source / "lse-q8g64"));
}

LSE_TEST(full_checkpoint_matches_the_python_reference) {
  const char* source = std::getenv("LSE_DFLASH2_CONVERT_SOURCE");
  if (source == nullptr) LSE_SKIP("set LSE_DFLASH2_CONVERT_SOURCE to the incoai/Qwen3.8-27B-DFlash2 snapshot");
  Scratch scratch;
  const fs::path out = scratch.path / "qwen38-27b-dflash2-q8";
  const ModelPaths paths{(fs::path(source) / "model.safetensors").string(),
                         (fs::path(source) / "config.json").string()};
  DFlash2ConvertOptions options{"incoai/Qwen3.8-27B-DFlash2", "dedf8df68adfb1afeaf7b7480c0a0243108177b4", {}};
  LSE_EXPECT_OK(convert_dflash2_q8(paths, out, options));
  LSE_EXPECT_EQ(fs::file_size(out / "model.safetensors"), std::uintmax_t{2044950184});
  LSE_EXPECT(file_sha(out / "model.safetensors") ==
             "cc3b5742f8edf02c4edcc734ef66e7c4c5f660b43438641ee36c5a72f8414188");
  LSE_EXPECT(file_sha(out / "config.json") == "411ffd1c02c210843385cd9eb5784554d0c982c8878669083c5e58258073a65d");
  LSE_EXPECT(file_sha(out / "source-repository.json") == "e85ae6a22b772666a068cb5669ec409b1ffae3a51c73210e9d288b8e9f138959");
  if (const char* reference = std::getenv("LSE_DFLASH2_CONVERT_REFERENCE")) {
    for (const char* name : {"model.safetensors", "config.json", "source-repository.json"})
      LSE_EXPECT(file_sha(out / name) == file_sha(fs::path(reference) / name));
  }
  // The load-time path: the origin comes from the HF snapshot path, so the
  // cached manifest is the same bytes, and a second open reuses the cache.
  DFlash2PrepareOptions prepare;
  prepare.cache_dir = scratch.path / "cache";
  auto prepared = prepare_dflash2_checkpoint(source, prepare);
  LSE_EXPECT(prepared.ok());
  if (!prepared.ok()) return;
  const fs::path cached = fs::path(prepared->weights).parent_path();
  for (const char* name : {"model.safetensors", "config.json", "source-repository.json"})
    LSE_EXPECT(file_sha(cached / name) == file_sha(out / name));
  bool converted_again = false;
  prepare.progress = [&](const DFlash2ConvertProgress& p) {
    converted_again |= p.phase == DFlash2ConvertProgress::Phase::kQuantize;
  };
  auto again = prepare_dflash2_checkpoint(source, prepare);
  LSE_EXPECT(again.ok() && again->weights == prepared->weights && !converted_again);
}

LSE_TEST_MAIN()
