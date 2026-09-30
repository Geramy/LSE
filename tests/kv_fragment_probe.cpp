#include <array>
#include <bit>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>
#include "lse/backend/backend.hpp"
#include "lse/backends/hrx/loomc/loom_emitter.hpp"
#include "lse/backends/hrx/loomc/loomc_compiler.hpp"
#include "lse/graph/ops.hpp"
#include "lse/kv/memory.hpp"

using namespace lse;
using namespace lse::graph;

int main(int argc, char** argv) {
  const int capacity = argc > 1 ? std::atoi(argv[1]) : 512;
  if (capacity < 512 || capacity > 131072 || capacity % 16) return 2;
  const int live = argc > 2 ? std::atoi(argv[2]) : (capacity == 512 ? 270 : capacity - 17);
  const int query_filter = argc > 3 ? std::atoi(argv[3]) : 0;
  if (live < 128 || live > capacity ||
      (query_filter != 0 && query_filter != 1 && query_filter != 8 && query_filter != 128)) return 2;
  auto made = backend::create_backend("hrx");
  if (!made.ok()) { std::fprintf(stderr, "%s\n", made.status().message().c_str()); return 1; }
  auto backend = made.release();
  auto status = backend->init(0);
  if (!status.ok()) { std::fprintf(stderr, "%s\n", status.message().c_str()); return 1; }
  auto run = [&]() -> Status {
    auto manager = kv::MemoryManager::create();
    auto leaf = [&](Shape shape, DType dtype, const void* data) -> Result<Array> {
      const auto bytes = dtype_storage_bytes(dtype, shape.elem_count());
      LSE_ASSIGN_OR(auto buffer, backend->allocate(bytes, backend::MemoryClass::kDevice));
      if (data) LSE_RETURN_IF_ERROR(backend->copy_h2d(data, buffer, bytes, 0));
      return Array::from_buffer(std::move(buffer), shape, dtype);
    };
    const Shape pool{capacity / 16, 4, 16, 256};
    std::vector<std::uint16_t> k(pool.elem_count()), v(k.size());
    for (std::size_t i = 0; i < k.size(); ++i) {
      k[i] = bfloat16_t(static_cast<float>(static_cast<int>(i % 97) - 48) / 64.f).bits;
      v[i] = bfloat16_t(static_cast<float>(static_cast<int>(i % 83) - 41) / 64.f).bits;
    }
    LSE_ASSIGN_OR(auto kc, leaf(pool, DType::kBF16, k.data()));
    LSE_ASSIGN_OR(auto vc, leaf(pool, DType::kBF16, v.data()));
    auto ks = std::make_shared<kv::FragmentStorage>(manager, *backend, backend::kDefaultStream);
    auto vs = std::make_shared<kv::FragmentStorage>(manager, *backend, backend::kDefaultStream);
    // Interleave the first fragments so logical neighbours are not necessarily
    // physical neighbours, then grow without moving either first fragment.
    LSE_RETURN_IF_ERROR(ks->grow(kv::kFragmentBytes));
    LSE_RETURN_IF_ERROR(vs->grow(kv::kFragmentBytes));
    LSE_RETURN_IF_ERROR(ks->grow(k.size() * sizeof(k[0])));
    LSE_RETURN_IF_ERROR(vs->grow(v.size() * sizeof(v[0])));
    LSE_RETURN_IF_ERROR(ks->write(k.data(), k.size() * sizeof(k[0])));
    LSE_RETURN_IF_ERROR(vs->write(v.data(), v.size() * sizeof(v[0])));
    LSE_ASSIGN_OR(auto kb, ks->binding());
    LSE_ASSIGN_OR(auto vb, vs->binding());
    auto kf = Array::from_buffer(std::move(kb), pool, DType::kBF16);
    auto vf = Array::from_buffer(std::move(vb), pool, DType::kBF16);
    kf.node()->kv_fragments = ks;
    vf.node()->kv_fragments = vs;
    std::vector<float> table(static_cast<std::size_t>(capacity / 16));
    for (std::size_t i = 0; i < table.size(); ++i) table[i] = static_cast<float>((i * 13 + 7) % table.size());
    LSE_ASSIGN_OR(auto blocks, leaf({1, capacity / 16}, DType::kF32, table.data()));
    backend::LoomEmitter emitter;
    backend::LoomcCompiler compiler;
    std::vector<float> additions(1 * 4 * 3 * 256, 0.3125f);
    const float write_metadata[]{127.f, 130.f, 1.f, 127.f, 130.f};
    LSE_ASSIGN_OR(auto source, leaf({1, 4, 3, 256}, DType::kF32, additions.data()));
    LSE_ASSIGN_OR(auto write_meta, leaf({5}, DType::kF32, write_metadata));
    for (auto* destination : {&kc, &kf}) {
      auto written = kv_page_write(*destination, source, write_meta, blocks, 16, kv::CacheDType::kBF16);
      written.node()->buffer = destination->node()->buffer;
      const NodePtr roots[]{written.node()};
      for (const auto& group : Partitioner::partition(roots, &backend->device_info())) {
        LSE_ASSIGN_OR(auto emitted, emitter.emit(group, backend->device_info()));
        std::ofstream("/tmp/kv-fragment-last.loom") << emitted.source;
        LSE_ASSIGN_OR(auto code, compiler.compile(emitted.source, std::string(backend->device_info().arch)));
        LSE_ASSIGN_OR(auto handle, backend->load_executable(emitted.entry_name, code.code));
        std::vector<backend::BufferRef> refs;
        for (const auto& node : emitted.binding_order)
          refs.push_back({&node->buffer, 0, node->buffer.size_bytes});
        const auto elements = static_cast<std::uint32_t>(written.node()->element_count());
        std::array<std::byte, 4> constants;
        std::memcpy(constants.data(), &elements, sizeof(elements));
        LSE_RETURN_IF_ERROR(backend->launch(handle, emitted.dims, {refs, constants}));
        LSE_RETURN_IF_ERROR(backend->synchronize());
      }
    }
    auto expected = k;
    std::vector<std::uint16_t> actual(k.size());
    for (int pos = 127; pos < 130; ++pos)
      for (int h = 0; h < 4; ++h)
        for (int d = 0; d < 256; ++d)
          expected[((static_cast<std::size_t>(table[pos / 16]) * 4 + h) * 16 + pos % 16) * 256 + d] = bfloat16_t(0.3125f).bits;
    LSE_RETURN_IF_ERROR(backend->copy_d2h(kc.node()->buffer, actual.data(), actual.size() * 2, 0));
    if (expected != actual) return LSE_ERROR(kInternal, "contiguous K/V write mismatch");
    LSE_RETURN_IF_ERROR(ks->read(actual.data(), actual.size() * 2));
    if (expected != actual) return LSE_ERROR(kInternal, "fragment K/V write mismatch");
    std::puts("PASS BF16 fragmented K/V writes with permuted page IDs");
    for (int queries : {1, 8, 128}) {
      if (query_filter && queries != query_filter) continue;
      const Shape qshape{1, 24, queries, 256};
      std::vector<float> data(qshape.elem_count());
      for (std::size_t i = 0; i < data.size(); ++i)
        data[i] = static_cast<float>(static_cast<int>(i % 71) - 35) / 64.f;
      const float metadata[]{static_cast<float>(live - queries), static_cast<float>(live), 1.f,
                             static_cast<float>(live - queries), static_cast<float>(live)};
      LSE_ASSIGN_OR(auto q, leaf(qshape, DType::kF32, data.data()));
      LSE_ASSIGN_OR(auto meta, leaf({5}, DType::kF32, metadata));
      std::vector<float> reference;
      for (bool fragmented : {false, true}) {
        auto output = sdpa_paged(q, fragmented ? kf : kc, fragmented ? vf : vc,
                                 0.0625f, MaskKind::kCausal, 0, meta, blocks, 16,
                                 &backend->device_info(), kv::CacheDType::kBF16);
        const NodePtr roots[]{output.node()};
        const auto groups = Partitioner::partition(roots, &backend->device_info());
        struct Stage {
          backend::KernelHandle handle;
          backend::LaunchDims dims;
          std::vector<backend::BufferRef> refs;
          std::array<std::byte, 4> constants;
        };
        std::vector<Stage> stages;
        for (const auto& group : groups) {
          LSE_ASSIGN_OR(auto emitted, emitter.emit(group, backend->device_info()));
          std::ofstream("/tmp/kv-fragment-last.loom") << emitted.source;
          LSE_ASSIGN_OR(auto code, compiler.compile(emitted.source, std::string(backend->device_info().arch)));
          LSE_ASSIGN_OR(auto handle, backend->load_executable(emitted.entry_name, code.code));
          std::vector<backend::BufferRef> refs;
          for (const auto& node : emitted.binding_order) {
            if (!node->buffer.valid()) {
              LSE_ASSIGN_OR(node->buffer, backend->allocate(
                  dtype_storage_bytes(node->dtype, node->element_count()), backend::MemoryClass::kDevice));
            }
            refs.push_back({&node->buffer, 0, node->buffer.size_bytes});
          }
          const auto elements = static_cast<std::uint32_t>(emitted.binding_order.back()->element_count());
          std::array<std::byte, 4> constants;
          std::memcpy(constants.data(), &elements, sizeof(elements));
          LSE_RETURN_IF_ERROR(backend->launch(handle, emitted.dims, {refs, constants}));
          LSE_RETURN_IF_ERROR(backend->synchronize());
          stages.push_back({std::move(handle), emitted.dims, std::move(refs), constants});
        }
        const auto start = std::chrono::steady_clock::now();
        for (int repeat = 0; repeat < 64; ++repeat)
          for (const auto& stage : stages)
            LSE_RETURN_IF_ERROR(backend->launch(stage.handle, stage.dims, {stage.refs, stage.constants}));
        LSE_RETURN_IF_ERROR(backend->synchronize());
        const double ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count() / 64.;
        std::printf("TIMING queries=%d fragmented=%d ms=%.6f\n", queries, fragmented, ms);
        std::vector<float> result(qshape.elem_count());
        LSE_RETURN_IF_ERROR(backend->copy_d2h(output.node()->buffer, result.data(), result.size() * sizeof(float), 0));
        if (!fragmented) reference = std::move(result);
        else {
          for (std::size_t i = 0; i < result.size(); ++i)
            if (!std::isfinite(result[i]) || std::bit_cast<std::uint32_t>(result[i]) != std::bit_cast<std::uint32_t>(reference[i]))
              return LSE_ERROR(kInternal, "fragment attention mismatch at ", std::to_string(i));
          std::printf("PASS BF16 fragmented attention queries=%d live=%d capacity=%d\n", queries, live, capacity);
        }
      }
    }
    return OkStatus();
  };
  status = run();
  if (!status.ok()) { std::fprintf(stderr, "%s\n", status.message().c_str()); return 1; }
  return 0;
}
