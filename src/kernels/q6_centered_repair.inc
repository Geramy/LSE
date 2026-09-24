Dims stage_dims(const graph::KernelShapes &s) {
  if (s.inputs.size() != 4 || s.inputs[0].rank() == 0)
    return {};
  auto normal = s;
  normal.output = Shape{static_cast<int64_t>(s.inputs[0].elem_count()) /
                            s.inputs[0].dim(s.inputs[0].rank() - 1),
                        s.inputs[1].dim(0)};
  return dims_of(normal);
}
struct CenteredStage final : graph::KernelPrimitive<CenteredStage> {
  static constexpr std::string_view kName =
                                        "quant_linear.centered_repair_stage_v1",
                                    kEntry = "lse_centered_repair_stage_v1",
                                    kSource = "";
  std::size_t arity() const noexcept override { return 4; }
  bool owns_indexing() const noexcept override { return true; }
  bool supports_epilogue() const noexcept override { return false; }
  Result<Shape> infer_shape(std::span<const Shape> in) const override {
    if (in.size() != 4)
      return LSE_ERROR(kInvalidArgument, "four operands required");
    return Shape{2 * static_cast<int64_t>(in[0].elem_count()) /
                     in[0].dim(in[0].rank() - 1),
                 in[1].dim(0)};
  }
  DType infer_dtype(std::span<const DType>) const override {
    return DType::kF32;
  }
  static graph::ThreadPlan plan_impl(const graph::KernelShapes &s) {
    auto d = stage_dims(s);
    graph::ThreadPlan p;
    p.workgroup_size[0] = 128;
    p.workgroup_count[0] = ((d.m + 63) / 64) * ((d.n + 63) / 64);
    p.lds_bytes = 26160;
    return p;
  }
  std::string emit_kernel(const graph::KernelShapes &s) const override {
    auto d = stage_dims(s);
    if (!d.valid || s.output.elem_count() != 2ull * d.m * d.n)
      return {};
    return emit_staged_bf16_centered(s, d, true);
  }
};
struct RepairArgs {
  env::In<kir::f32, env::Emit> staged, x;
  env::In<uint32_t, env::Emit> packed;
  env::In<lse::bf16, env::Emit> scales, biases;
  env::Out<kir::f32, env::Emit> out;
};
struct CenteredRepair final : graph::KernelPrimitive<CenteredRepair> {
  static constexpr std::string_view
      kName = "quant_linear.centered_repair_finish_v1",
      kEntry = "lse_centered_repair_finish_v1", kSource = "";
  std::size_t arity() const noexcept override { return 5; }
  bool owns_indexing() const noexcept override { return true; }
  bool supports_epilogue() const noexcept override { return false; }
  Result<Shape> infer_shape(std::span<const Shape> in) const override {
    if (in.size() != 5)
      return LSE_ERROR(kInvalidArgument, "five operands required");
    Shape o;
    for (size_t i = 0; i + 1 < in[1].rank(); ++i)
      o.push_back(in[1].dim(i));
    o.push_back(in[2].dim(0));
    return o;
  }
  DType infer_dtype(std::span<const DType>) const override {
    return DType::kF32;
  }
  static graph::ThreadPlan plan_impl(const graph::KernelShapes &s) {
    graph::ThreadPlan p;
    p.workgroup_size[0] = 256;
    p.workgroup_count[0] = (s.output.elem_count() + 255) / 256;
    return p;
  }
  std::string emit_kernel(const graph::KernelShapes &s) const override {
    if (s.inputs.size() != 5 || !s.store || !s.intrinsics ||
        s.inputs[1].rank() == 0)
      return {};
    const uint32_t count = s.output.elem_count(), n = s.inputs[2].dim(0),
                   k = s.inputs[1].dim(s.inputs[1].rank() - 1), groups = k / 64,
                   words = k * 6 / 32;
    if (s.inputs[0].elem_count() != 2ull * count || k % 64)
      return {};
    kir::KernelBody body(s.types, *s.intrinsics);
    body.set_store(s.store);
    RepairArgs a;
    if (!env::bind(body, a, s))
      return {};
    env::Emit e{&body};
    const auto i = e.let(e.thread_id());
    (void)e.ret_if(i >= count);
    auto value = e.var(a.staged[i]);
    if (auto repair = e.when(a.staged[count + i] != 0.0f)) {
      value = e.f32(0.0f);
      const auto row = e.let(i / n), col = e.let(i % n);
      for (auto chunk : e.range(0u, k / 16u, 1u)) {
        const auto gi = e.let(col * groups + chunk / 4u);
        const auto scale = e.let(math::widen(a.scales[gi])),
                   bias = e.let(math::widen(a.biases[gi]));
        quant::dequant_chunk(
            e, a.packed, quant::GroupAffine{6, 64},
            e.let(col * words + chunk * 3u), scale, bias,
            [&](int q, const kir::Val<kir::f32> &w) {
              value = math::fma(
                  a.x[row * k + chunk * 16u + static_cast<unsigned>(q)], w,
                  value.read());
            });
      }
    }
    e.store(i, value.read());
    return body.str();
  }
};
