// Shared HIP/Loom scaled residual feeder. Only format conversion and matrix
// instruction spellings are supplied by the backend intrinsic table.
constexpr std::uint32_t kResidualLdsBytes = 16384 + 512 + 80;
template <math::MatrixElem T>
std::string emit_residual(const KernelShapes &s, const Dims &d) {
  using Tile = Base<T>;
  using Op = typename Tile::Op;
  constexpr auto row = Tile::kRow;
  constexpr auto geo = geometry_of(row);
  constexpr float limit = T == math::MatrixElem::kFp8 ? 448.0f : 57344.0f;
  constexpr float minNormal = std::numeric_limits<float>::min();
  constexpr float maxFloat = std::numeric_limits<float>::max();
  static_assert(row.a_len == 2 && row.b_len == 2 && geo.wave == 32 &&
                geo.split_k && geo.lane_k == 8);
  kir::KernelBody body(s.types, *s.intrinsics, workgroup_lds_bytes(s.device));
  body.set_store(s.store);
  Args a;
  if (!env::bind(body, a, s))
    return {};
  env::Emit e{&body};
  const auto packed = e.lds<kir::i32>(4096u);
  const auto scales = e.lds<kir::f32>(128u);
  // Per-wave invalid, maxA, maxB, minA and minB. Every entry has one writer.
  const auto summary = e.lds<kir::f32>(20u);
  const auto lid = e.let(math::local_id()), lane = e.let(lid % 32u);
  const auto wave = e.let(lid / 32u), lo = e.let(lane % 16u),
             hi = e.let(lane / 16u);
  const auto wg = e.let(math::workgroup_id_x());
  const auto nblocks = (d.n + 63) / 64;
  const auto mbase = e.let((wg / nblocks) * 64u),
             nbase = e.let((wg % nblocks) * 64u);
  const auto am = e.let((wave % 2u) * 32u), bn = e.let((wave / 2u) * 32u);
  // Four-byte packs rotate in pairs, preserving eight-byte matrix fragments.
  const auto address = [&](const kir::Val<kir::u32> &r,
                           const kir::Val<kir::u32> &k) {
    return r * 16u + ((k / 8u + r % 8u) % 8u) * 2u + (k % 8u) / 4u;
  };
  std::vector<kir::Local<kir::f32, 8>> acc;
  for (unsigned i = 0; i < 4; ++i) {
    acc.push_back(e.local<kir::f32, 8>());
    for (auto j : e.unroll(8u))
      acc.back()[j] = e.f32(0);
  }
  for (auto kb : e.range(0u, d.k, 64u)) {
    auto bad = e.var(0.0f), maxA = e.var(0.0f), maxB = e.var(0.0f);
    auto minA = e.var(maxFloat), minB = e.var(maxFloat);
    for (auto block : e.unroll(2u)) {
      const auto r = e.let(lid / 4u + block * 32u),
                 kc = e.let((lid % 4u) * 16u);
      const auto ar = e.let(mbase + r), bc = e.let(nbase + r);
      const auto av = e.local<kir::f32, 16>(), bv = e.local<kir::f32, 16>();
      for (auto j : e.unroll(16u)) {
        av[j] = e.f32(0);
        bv[j] = e.f32(0);
      }
      if (auto active = e.when(ar < d.m)) {
        for (unsigned v = 0; v < 4; ++v) {
          const auto loaded = e.load(a.x, ar * d.k + kb + kc + v * 4u, 16u);
          for (unsigned j = 0; j < 4; ++j)
            av[v * 4 + j] = loaded[j];
        }
      }
      if (auto active = e.when(bc < d.n)) {
        const auto gi = e.let(bc * d.groups + (kb + kc) / d.group);
        const auto scale = e.let(math::widen(a.scales[gi])),
                   bias = e.let(math::widen(a.biases[gi]));
        const auto wb = e.let(bc * d.words + ((kb + kc) / 16u) * 3u);
        const std::array<kir::Val<kir::u32>, 3> words{e.let(a.packed[wb]),
                                                      e.let(a.packed[wb + 1u]),
                                                      e.let(a.packed[wb + 2u])};
        for (unsigned j = 0; j < 16; ++j) {
          const unsigned off = (j * 6) % 32, wi = j * 6 / 32;
          auto code = words[wi] / (1u << off);
          if (off > 26)
            code = code +
                   (words[wi + 1] % (1u << (off - 26))) * (1u << (32 - off));
          else
            code = code % 64u;
          bv[j] = math::fma(math::cast<kir::f32>(e.let(code)), scale, bias);
        }
      }
      auto ax = e.var(0.0f), bw = e.var(0.0f), rowBad = e.var(0.0f);
      for (auto j : e.unroll(16u)) {
        const auto x = av[j].read(), w = bv[j].read();
        // Ordered floating comparisons make NaN != NaN false in this IR.
        // Start invalid and clear only after a positive finite-range proof.
        auto invalid=e.var(1.0f);
        if(auto finite=e.when(math::abs(x)<=maxFloat && math::abs(w)<=maxFloat))
          invalid=e.f32(0);
        rowBad=math::max(rowBad.read(),invalid.read());
        ax = math::max(ax.read(), math::abs(x));
        bw = math::max(bw.read(), math::abs(w));
      }
      for (unsigned bit : {1u, 2u}) {
        ax = math::max(ax.read(), math::shfl_xor(ax.read(), e.u32(bit)));
        bw = math::max(bw.read(), math::shfl_xor(bw.read(), e.u32(bit)));
        rowBad =
            math::max(rowBad.read(), math::shfl_xor(rowBad.read(), e.u32(bit)));
      }
      auto sx = e.var(1.0f), sw = e.var(1.0f);
      if (auto live = e.when(ax.read() > 0.0f))
        sx = ax.read() * (1.0f / limit);
      if (auto live = e.when(bw.read() > 0.0f))
        sw = bw.read() * (1.0f / limit);
      // Tiny/invalid scale blocks use the original FP32 path, not a silent
      // quantization-to-zero. Zero rows retain a benign unit scale.
      if (auto invalid =
              e.when(sx.read() < minNormal || sw.read() < minNormal ||
                     sx.read() > maxFloat || sw.read() > maxFloat))
        rowBad = e.f32(1);
      bad = math::max(bad.read(), rowBad.read());
      maxA = math::max(maxA.read(), sx.read());
      maxB = math::max(maxB.read(), sw.read());
      minA = math::min(minA.read(), sx.read());
      minB = math::min(minB.read(), sw.read());
      if (auto leader = e.when(lid % 4u == 0u)) {
        scales[r] = sx.read();
        scales[64u + r] = sw.read();
      }
      for (unsigned pack = 0; pack < 4; ++pack) {
        const auto at = e.let(address(r, kc + pack * 4u));
        packed[at] = math::cast<kir::i32>(e.u32(0));
        packed[1024u + at] = math::cast<kir::i32>(e.u32(0));
        packed[2048u + at] = math::cast<kir::i32>(e.u32(0));
        packed[3072u + at] = math::cast<kir::i32>(e.u32(0));
        if (auto valid = e.when(rowBad.read() == 0.0f)) {
          std::array<kir::Val<kir::f32>, 4> x{
              av[pack * 4].read() / sx.read(),
              av[pack * 4 + 1].read() / sx.read(),
              av[pack * 4 + 2].read() / sx.read(),
              av[pack * 4 + 3].read() / sx.read()};
          std::array<kir::Val<kir::f32>, 4> w{
              bv[pack * 4].read() / sw.read(),
              bv[pack * 4 + 1].read() / sw.read(),
              bv[pack * 4 + 2].read() / sw.read(),
              bv[pack * 4 + 3].read() / sw.read()};
          const auto xh = e.let(math::pack_fp8<T>(x[0], x[1], x[2], x[3]));
          const auto wh = e.let(math::pack_fp8<T>(w[0], w[1], w[2], w[3]));
          packed[at] = math::cast<kir::i32>(xh);
          packed[2048u + at] = math::cast<kir::i32>(wh);
          packed[1024u + at] = math::cast<kir::i32>(
              math::pack_fp8<T>(x[0] - math::unpack_fp8<T, 0>(xh),
                                x[1] - math::unpack_fp8<T, 1>(xh),
                                x[2] - math::unpack_fp8<T, 2>(xh),
                                x[3] - math::unpack_fp8<T, 3>(xh)));
          packed[3072u + at] = math::cast<kir::i32>(
              math::pack_fp8<T>(w[0] - math::unpack_fp8<T, 0>(wh),
                                w[1] - math::unpack_fp8<T, 1>(wh),
                                w[2] - math::unpack_fp8<T, 2>(wh),
                                w[3] - math::unpack_fp8<T, 3>(wh)));
        }
      }
    }
    for (unsigned bit : {1u, 2u, 4u, 8u, 16u}) {
      bad = math::max(bad.read(), math::shfl_xor(bad.read(), e.u32(bit)));
      maxA = math::max(maxA.read(), math::shfl_xor(maxA.read(), e.u32(bit)));
      maxB = math::max(maxB.read(), math::shfl_xor(maxB.read(), e.u32(bit)));
      minA = math::min(minA.read(), math::shfl_xor(minA.read(), e.u32(bit)));
      minB = math::min(minB.read(), math::shfl_xor(minB.read(), e.u32(bit)));
    }
    if (auto leader = e.when(lane == 0u)) {
      summary[wave] = bad.read();
      summary[4u + wave] = maxA.read();
      summary[8u + wave] = maxB.read();
      summary[12u + wave] = minA.read();
      summary[16u + wave] = minB.read();
    }
    e.barrier();
    auto blockBad = e.var(0.0f), maxSx = e.var(0.0f), maxSw = e.var(0.0f);
    auto minSx = e.var(maxFloat), minSw = e.var(maxFloat);
    for (auto w : e.unroll(4u)) {
      blockBad = math::max(blockBad.read(), summary[w]);
      maxSx = math::max(maxSx.read(), summary[4u + w]);
      maxSw = math::max(maxSw.read(), summary[8u + w]);
      minSx = math::min(minSx.read(), summary[12u + w]);
      minSw = math::min(minSw.read(), summary[16u + w]);
    }
    // Conservative uniform scale-product bounds prevent intermediate overflow
    // or underflow even when the final exact contraction would cancel.
    if (auto unsafe =
            e.when(maxSx.read() >
                       (maxFloat / (2.0f * limit * limit * 64.0f)) / maxSw.read() ||
                   minSx.read() < minNormal / minSw.read()))
      blockBad = e.f32(1);
    if (auto normal = e.when(blockBad.read() == 0.0f)) {
      std::vector<kir::Local<kir::f32, 8>> partial;
      for (unsigned i = 0; i < 4; ++i) {
        partial.push_back(e.local<kir::f32, 8>());
        for (auto j : e.unroll(8u))
          partial.back()[j] = e.f32(0);
      }
      for (auto slice : e.unroll(4u)) {
        std::vector<kir::Local<kir::i32, 2>> ah, al, bh, bl;
        for (unsigned i = 0; i < 2; ++i) {
          ah.push_back(e.local<kir::i32, 2>());
          al.push_back(e.local<kir::i32, 2>());
          bh.push_back(e.local<kir::i32, 2>());
          bl.push_back(e.local<kir::i32, 2>());
          const auto ar = e.let(am + i * 16u + lo),
                     bc = e.let(bn + i * 16u + lo);
          for (auto j : e.unroll(2u)) {
            const auto k = e.let(slice * 16u + hi * 8u + j * 4u);
            const auto ai = e.let(address(ar, k)), bi = e.let(address(bc, k));
            ah.back()[j] = math::cast<kir::i32>(packed[ai].read());
            al.back()[j] = math::cast<kir::i32>(packed[1024u + ai].read());
            bh.back()[j] = math::cast<kir::i32>(packed[2048u + bi].read());
            bl.back()[j] = math::cast<kir::i32>(packed[3072u + bi].read());
          }
        }
        for (unsigned m = 0; m < 2; ++m)
          for (unsigned n = 0; n < 2; ++n) {
            partial[m * 2 + n] = math::mma<Op>(ah[m].value(), bh[n].value(),
                                               partial[m * 2 + n].value());
            partial[m * 2 + n] = math::mma<Op>(al[m].value(), bh[n].value(),
                                               partial[m * 2 + n].value());
            partial[m * 2 + n] = math::mma<Op>(ah[m].value(), bl[n].value(),
                                               partial[m * 2 + n].value());
          }
      }
      for (unsigned m = 0; m < 2; ++m)
        for (unsigned n = 0; n < 2; ++n) {
          const auto col = e.let(bn + n * 16u + lo);
          for (auto j : e.unroll(8u)) {
            const auto rr =
                e.let(am + m * 16u + j * geo.slot_step + hi * geo.half_rows);
            const auto scale =
                e.let(scales[rr].read() * scales[64u + col].read());
            acc[m * 2 + n][j] = math::fma(partial[m * 2 + n][j].read(), scale,
                                          acc[m * 2 + n][j].read());
          }
        }
    }
    if (auto exceptional = e.when(blockBad.read() != 0.0f)) {
      // Reload only exceptional blocks. No original FP32 panel needs to remain
      // live in LDS during the fast path.
      for (unsigned m = 0; m < 2; ++m)
        for (unsigned n = 0; n < 2; ++n) {
          const auto col = e.let(nbase + bn + n * 16u + lo);
          for (auto j : e.unroll(8u)) {
            const auto rr = e.let(mbase + am + m * 16u + j * geo.slot_step +
                                  hi * geo.half_rows);
            if (auto active = e.when(rr < d.m && col < d.n)) {
              for (auto chunk : e.range(kb / 16u, kb / 16u + 4u, 1u)) {
                const auto gi = e.let(col * d.groups + (chunk * 16u) / d.group);
                const auto scale = e.let(math::widen(a.scales[gi])),
                           bias = e.let(math::widen(a.biases[gi]));
                quant::dequant_chunk(
                    e, a.packed, quant::GroupAffine{6, 64},
                    e.let(col * d.words + chunk * 3u), scale, bias,
                    [&](int q, const kir::Val<kir::f32> &weight) {
                      const auto x = a.x[rr * d.k + chunk * 16u +
                                         static_cast<unsigned>(q)];
                      acc[m * 2 + n][j] =
                          math::fma(x, weight, acc[m * 2 + n][j].read());
                    });
              }
            }
          }
        }
    }
    e.barrier();
  }
  for (unsigned m = 0; m < 2; ++m)
    for (unsigned n = 0; n < 2; ++n) {
      const auto col = e.let(nbase + bn + n * 16u + lo);
      for (auto j : e.unroll(8u)) {
        const auto rr = e.let(mbase + am + m * 16u + j * geo.slot_step +
                              hi * geo.half_rows);
        if (auto active = e.when(rr < d.m && col < d.n))
          e.store(rr * d.n + col, acc[m * 2 + n][j].read());
      }
    }
  if (!body.lds().ok())
    return {};
  return body.str();
}
