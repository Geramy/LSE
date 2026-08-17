#include "lse/graph/kernel_args.hpp"
#include "lse/graph/kernel_env.hpp"
#include "lse/graph/kernel_primitive.hpp"
#include "lse/math.hpp"

#include <string>

namespace lse::backend::hrx_kernels {

using namespace lse::graph;
namespace math = lse::math;

namespace {

// How a kernel reads a value the dispatch supplies rather than one baked into
// its source. HRX separates buffer bindings from a push-constant block, and
// that block carries exactly one field today (`count`, filled in
// graph::Scheduler) — so the dispatch surface a kernel primitive can actually
// reach is a small device slot it binds like any other input. Spelling the read
// here is what lets the extent be *declared* as ExtentBinding::kRuntime and go
// through ir::verify's rules, instead of being an untyped buffer load the
// verifier never sees.
std::string dispatch_u32(const kir::KernelBody& k, const kir::TypeTable& types,
                         std::size_t input, unsigned element) {
  return "(" + std::string(types.scalar(kir::Scalar::kU32)) + ")(" +
         k.input_name(input) + "[" + std::to_string(element) + "u])";
}

constexpr bool is_pow2(std::uint32_t v) noexcept {
  return v >= 2 && (v & (v - 1)) == 0;
}

}  // namespace

template <class E>
struct SdpaArgs {
  env::In<kir::f32, E> q;
  env::In<kir::f32, E> k;
  env::In<kir::f32, E> v;
  // Optional 4th input. Contiguous form: [1], the live cache offset. Paged
  // form: [3] = {first query position, live KV length, real rows}.
  env::In<kir::f32, E> meta;
  // Optional 5th input, paged form only: [rows, stride] block ids.
  env::In<kir::f32, E> table;
  env::Out<kir::f32, E> out;
};

// One thread per output [B, Hq, Tq, Dv]. Recomputes the key dots for softmax;
// S is small on the generate path (prefill tokens or cached length).
//
// Two forms, chosen by the input count:
//
//   3 or 4 inputs — k/v are one contiguous [B, Hkv, S, Dh] span per sequence.
//   5 inputs      — k/v are pools [blocks, Hkv, block_size, Dh] and the block
//                   table says which block holds each position.
//
// The paged form is what takes the sequence length out of the address
// arithmetic: the row stride becomes `block_size`, a literal, and the varying
// part becomes a looked-up origin. The length survives only as the softmax trip
// count, which is an outermost loop bound — the one place ir::verify lets a
// runtime extent appear. The scan order over positions is unchanged, so a paged
// read and a contiguous read of the same logical KV produce the same bits.
struct SdpaKernel final : KernelPrimitive<SdpaKernel> {
  static constexpr std::string_view kName = "attention";
  static constexpr std::string_view kEntry = "lse_sdpa";
  static constexpr std::string_view kSource = {};

  std::size_t arity() const noexcept override { return 3; }

  std::string emit_kernel(const KernelShapes& s) const override {
    if (s.inputs.size() < 3 || s.inputs.size() > 5 ||
        s.types.scalar == nullptr || s.intrinsics == nullptr ||
        s.inputs[0].rank() != 4 || s.inputs[1].rank() != 4 ||
        s.inputs[2].rank() != 4) {
      return {};
    }
    const Shape& q = s.inputs[0];
    const Shape& ksh = s.inputs[1];
    const Shape& vsh = s.inputs[2];
    const auto bsz = static_cast<std::uint32_t>(q.dim(0));
    const auto qh = static_cast<std::uint32_t>(q.dim(1));
    const auto tq = static_cast<std::uint32_t>(q.dim(2));
    const auto dh = static_cast<std::uint32_t>(q.dim(3));
    const auto kvh = static_cast<std::uint32_t>(ksh.dim(1));
    // Contiguous: the allocated sequence length. Paged: the block size.
    const auto ts = static_cast<std::uint32_t>(ksh.dim(2));
    const auto dv = static_cast<std::uint32_t>(vsh.dim(3));
    if (qh == 0 || kvh == 0 || qh % kvh != 0 || dh == 0 || ts == 0 || dv == 0) {
      return {};
    }
    const auto group = qh / kvh;
    const float scale = s.attrs[0];
    const int mask = s.iattrs[0];
    const auto window = static_cast<std::uint32_t>(s.iattrs[1]);
    const auto baked_off = static_cast<std::uint32_t>(s.iattrs[2]);
    const bool live_off = s.inputs.size() >= 4;
    const bool paged = s.inputs.size() == 5;
    (void)bsz;

    std::uint32_t stride = 0;
    if (paged) {
      if (s.inputs[4].rank() < 2) return {};
      stride = static_cast<std::uint32_t>(s.inputs[4].dim(s.inputs[4].rank() - 1));
      const auto want = static_cast<std::uint32_t>(s.iattrs[3]);
      // The pool's own third dimension is the authority on the block size; a
      // node that disagrees with the buffer it points at is a mismatch, not a
      // reason to pick one.
      if (stride == 0 || want != ts) return {};
      // Power of two required: `pos / block_size` and `pos % block_size` are
      // then a shift and a mask rather than an integer divide in the address
      // chain. See kv/block.hpp.
      if (!is_pow2(ts)) return {};
    }

    kir::KernelBody k(s.types, *s.intrinsics);
    SdpaArgs<env::Emit> a;
    if (!env::bind(k, a, s)) return {};
    env::Emit e{&k};
    const auto i = e.thread_id();
    const auto d = e.let(i % dv);
    const auto qi = e.let((i / dv) % tq);
    const auto h = e.let((i / (dv * tq)) % qh);
    const auto b = e.let(i / (dv * tq * qh));
    const auto kh = e.let(h / group);
    kir::Val<kir::u32> offset = e.u32(baked_off);
    if (live_off) {
      offset = e.let(kir::cast<kir::u32>(a.meta[0u]));
    }
    const auto abs_i = e.let(offset + qi);
    const auto qb0 = e.let(((b * qh + h) * tq + qi) * dh);

    if (!paged) {
      const auto used = e.let(offset + tq);
      const auto hi = e.let(select(used < e.u32(ts), used, e.u32(ts)));

      const auto m = e.var(math::neg_inf());
      for (auto j : e.range(live_off ? hi : e.u32(ts))) {
        auto score = e.var(0.0f);
        for (auto dd : e.range(dh)) {
          const auto kb = ((b * kvh + kh) * ts + j) * dh + dd;
          score = math::fma(a.q[qb0 + dd], a.k[kb], score.read());
        }
        score = score.read() * scale;
        auto take = [&] { m = math::max(m.read(), score.read()); };
        if (mask == 0) {
          take();
        } else if (mask == 1) {
          if (auto g = e.when(j <= abs_i)) take();
        } else {
          if (auto g = e.when(j <= abs_i && (abs_i - j) < window)) take();
        }
      }

      auto denom = e.var(0.0f);
      auto acc = e.var(0.0f);
      for (auto j : e.range(live_off ? hi : e.u32(ts))) {
        auto score = e.var(0.0f);
        for (auto dd : e.range(dh)) {
          const auto kb = ((b * kvh + kh) * ts + j) * dh + dd;
          score = math::fma(a.q[qb0 + dd], a.k[kb], score.read());
        }
        score = score.read() * scale;
        auto w = e.var(0.0f);
        auto apply = [&] { w = math::exp(score.read() - m.read()); };
        if (mask == 0) {
          apply();
        } else if (mask == 1) {
          if (auto g = e.when(j <= abs_i)) apply();
        } else {
          if (auto g = e.when(j <= abs_i && (abs_i - j) < window)) apply();
        }
        denom = denom.read() + w.read();
        const auto vb = ((b * kvh + kh) * ts + j) * dv + d;
        acc = math::fma(w.read(), a.v[vb], acc.read());
      }
      e.ret(acc.read() /
            select(denom.read() == 0.0f, e.f32(1.0f), denom.read()));
      return k.str();
    }

    // Padded batch rows. A pass is widened to a bucket so the bucket, not the
    // true row count, is what reaches the JIT key; the rows past `rows` read
    // real blocks through their own table row and answer zero. Returning here
    // rather than masking inside the address arithmetic is the width-invariance
    // rule: no real row's result may depend on how many rows shared the pass.
    const auto rows = e.runtime_extent("rows", dispatch_u32(k, s.types, 3, 2));
    if (auto pad = e.when(b >= rows)) e.ret(e.f32(0.0f));

    // The live KV length. One code object serves every sequence length: this is
    // the softmax trip count and nothing else, and an outermost trip count is
    // exactly where ExtentBinding::kRuntime is legal.
    const auto kv_len = e.runtime_extent("kv_len", dispatch_u32(k, s.types, 3, 1));
    const auto nblk = e.let((kv_len + e.u32(ts - 1)) / e.u32(ts));
    const auto tb = e.let(b * stride);

    const auto m = e.var(math::neg_inf());
    for (auto bi : e.range(nblk)) {
      // One table read per block, not per key: the whole reason a block is 16
      // positions wide is that this load amortizes over them.
      const auto blk = e.let(kir::cast<kir::u32>(a.table[tb + bi]));
      const auto kb0 = e.let(((blk * kvh + kh) * ts) * dh);
      const auto j0 = e.let(bi * e.u32(ts));
      for (auto jj : e.range(ts)) {
        const auto j = e.let(j0 + jj);
        if (auto live = e.when(j < kv_len)) {
          auto score = e.var(0.0f);
          for (auto dd : e.range(dh)) {
            score = math::fma(a.q[qb0 + dd], a.k[kb0 + jj * dh + dd],
                              score.read());
          }
          score = score.read() * scale;
          auto take = [&] { m = math::max(m.read(), score.read()); };
          if (mask == 0) {
            take();
          } else if (mask == 1) {
            if (auto g = e.when(j <= abs_i)) take();
          } else {
            if (auto g = e.when(j <= abs_i && (abs_i - j) < window)) take();
          }
        }
      }
    }

    auto denom = e.var(0.0f);
    auto acc = e.var(0.0f);
    for (auto bi : e.range(nblk)) {
      const auto blk = e.let(kir::cast<kir::u32>(a.table[tb + bi]));
      const auto kb0 = e.let(((blk * kvh + kh) * ts) * dh);
      const auto vb0 = e.let(((blk * kvh + kh) * ts) * dv + d);
      const auto j0 = e.let(bi * e.u32(ts));
      for (auto jj : e.range(ts)) {
        const auto j = e.let(j0 + jj);
        if (auto live = e.when(j < kv_len)) {
          auto score = e.var(0.0f);
          for (auto dd : e.range(dh)) {
            score = math::fma(a.q[qb0 + dd], a.k[kb0 + jj * dh + dd],
                              score.read());
          }
          score = score.read() * scale;
          auto w = e.var(0.0f);
          auto apply = [&] { w = math::exp(score.read() - m.read()); };
          if (mask == 0) {
            apply();
          } else if (mask == 1) {
            if (auto g = e.when(j <= abs_i)) apply();
          } else {
            if (auto g = e.when(j <= abs_i && (abs_i - j) < window)) apply();
          }
          denom = denom.read() + w.read();
          acc = math::fma(w.read(), a.v[vb0 + jj * dv], acc.read());
        }
      }
    }
    e.ret(acc.read() / select(denom.read() == 0.0f, e.f32(1.0f), denom.read()));
    return k.str();
  }

  Result<Shape> infer_shape(std::span<const Shape> in) const override {
    if (in.size() < 3 || in.size() > 5) {
      return LSE_ERROR(kInvalidArgument, "sdpa takes 3, 4 or 5 inputs");
    }
    return Shape{in[0].dim(0), in[0].dim(1), in[0].dim(2), in[2].dim(3)};
  }
  DType infer_dtype(std::span<const DType> in) const override {
    return in.empty() ? DType::kF32 : in[0];
  }
  static ThreadPlan plan_impl(const KernelShapes& s) {
    ThreadPlan tp;
    const std::uint32_t threads =
        s.device && s.device->max_threads_per_workgroup >= 256 ? 256u : 64u;
    const auto elems = static_cast<std::uint32_t>(s.output.elem_count());
    tp.workgroup_size[0] = threads;
    tp.workgroup_count[0] = elems == 0 ? 1u : (elems + threads - 1) / threads;
    return tp;
  }
};
LSE_REGISTER_PRIMITIVE(SdpaKernel);

template <class E>
struct KvPageWriteArgs {
  env::In<kir::f32, E> dst;  // inplace pool; written through the store hook
  env::In<kir::f32, E> src;
  env::In<kir::f32, E> meta;
  env::In<kir::f32, E> table;
  env::Out<kir::f32, E> out;
};

// Writes src [rows, Hkv, T, W] into the pool dst [blocks, Hkv, block_size, W]
// at absolute position meta[0], following the block table. Threads cover src
// only; pool bytes outside the positions written stay put.
//
// The paged counterpart of overwrite_slice: there the destination offset was
// `pos` with coefficient 1 against a baked capacity stride; here the capacity
// stride is gone and `pos` selects both a block and a slot inside it.
struct KvPageWriteKernel final : KernelPrimitive<KvPageWriteKernel> {
  static constexpr std::string_view kName = "kv_page_write";
  static constexpr std::string_view kEntry = "lse_kv_page_write";
  static constexpr std::string_view kSource = {};

  std::size_t arity() const noexcept override { return 4; }
  bool owns_indexing() const noexcept override { return true; }
  bool supports_epilogue() const noexcept override { return false; }
  int inplace_input() const noexcept override { return 0; }

  std::string emit_kernel(const KernelShapes& s) const override {
    if (s.inputs.size() != 4 || s.types.scalar == nullptr ||
        s.intrinsics == nullptr || !s.store || s.inputs[0].rank() != 4 ||
        s.inputs[1].rank() != 4 || s.inputs[3].rank() < 2) {
      return {};
    }
    const Shape& dst = s.inputs[0];
    const Shape& src = s.inputs[1];
    const auto kvh = static_cast<std::uint32_t>(dst.dim(1));
    const auto bs = static_cast<std::uint32_t>(dst.dim(2));
    const auto width = static_cast<std::uint32_t>(dst.dim(3));
    const auto t = static_cast<std::uint32_t>(src.dim(2));
    const auto stride =
        static_cast<std::uint32_t>(s.inputs[3].dim(s.inputs[3].rank() - 1));
    if (kvh == 0 || bs == 0 || width == 0 || t == 0 || stride == 0) return {};
    if (static_cast<std::uint32_t>(src.dim(1)) != kvh ||
        static_cast<std::uint32_t>(src.dim(3)) != width) {
      return {};
    }
    if (static_cast<std::uint32_t>(s.iattrs[0]) != bs) return {};
    if (!is_pow2(bs)) return {};
    const auto src_n = static_cast<std::uint32_t>(src.elem_count());
    if (src_n == 0) return {};

    kir::KernelBody k(s.types, *s.intrinsics);
    k.set_store(s.store);
    KvPageWriteArgs<env::Emit> a;
    if (!env::bind(k, a, s)) return {};
    env::Emit e{&k};
    const auto i = e.thread_id();
    (void)e.ret_if(i >= src_n);

    const auto w = e.let(i % width);
    const auto tt = e.let((i / width) % t);
    const auto h = e.let((i / (width * t)) % kvh);
    const auto r = e.let(i / (width * t * kvh));
    // The same padded-bucket rule the attention kernel follows: pad rows do no
    // work rather than writing somewhere harmless, so a real row's blocks
    // cannot be reached by a row that is not in the batch.
    const auto rows = e.runtime_extent("rows", dispatch_u32(k, s.types, 2, 2));
    (void)e.ret_if(r >= rows);

    const auto pos = e.let(kir::cast<kir::u32>(a.meta[0u]));
    const auto abs = e.let(pos + tt);
    const auto blk =
        e.let(kir::cast<kir::u32>(a.table[r * stride + abs / bs]));
    const auto slot = e.let(abs % bs);
    const auto dest = e.let(((blk * kvh + h) * bs + slot) * width + w);
    e.store(dest, a.src[i]);
    return k.str();
  }

  Result<Shape> infer_shape(std::span<const Shape> in) const override {
    if (in.size() != 4) {
      return LSE_ERROR(kInvalidArgument, "kv_page_write takes 4 inputs");
    }
    return in[0];
  }
  DType infer_dtype(std::span<const DType> in) const override {
    return in.empty() ? DType::kF32 : in[0];
  }

  static ThreadPlan plan_impl(const KernelShapes& s) {
    ThreadPlan tp;
    const std::uint32_t threads =
        s.device && s.device->max_threads_per_workgroup >= 256 ? 256u : 64u;
    const auto elems = s.inputs.size() > 1
                           ? static_cast<std::uint32_t>(s.inputs[1].elem_count())
                           : 1u;
    tp.workgroup_size[0] = threads;
    tp.workgroup_count[0] = elems == 0 ? 1u : (elems + threads - 1) / threads;
    return tp;
  }
};
LSE_REGISTER_PRIMITIVE(KvPageWriteKernel);

}  // namespace lse::backend::hrx_kernels
