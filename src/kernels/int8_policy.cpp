#include "lse/kernels/int8_policy.hpp"
#include "lse/graph/kernel_primitive.hpp"
namespace lse::kernels {
bool activation_int8_enabled(const graph::KernelShapes& s, ActivationInt8Policy policy) noexcept {
  if (policy == ActivationInt8Policy::kEnabled) return true;
  if (policy == ActivationInt8Policy::kExact) return false;
  if (s.iattrs[2] != kQwen27BQ4ComputeRevision || s.iattrs[0] != 4 || s.iattrs[1] != 64 ||
      !s.device || s.device->arch != "gfx1201" || s.device->wavefront_size != 32 ||
      s.inputs.size() != 4 || s.input_dtypes.size() != 4 || s.inputs[1].rank() != 2 ||
      s.input_dtypes[0] != DType::kF32 || s.input_dtypes[1] != DType::kU32 ||
      s.input_dtypes[2] != DType::kBF16 || s.input_dtypes[3] != DType::kBF16 ||
      s.output_dtype != DType::kF32 || !s.staged.name.empty() || !s.staged_quant.codes.empty())
    return false;
  auto n = s.inputs[1].dim(0);
  if (n <= 0 || s.output.elem_count() % static_cast<std::size_t>(n)) return false;
  const auto m = s.output.elem_count() / static_cast<std::size_t>(n);
  if (m != 1 && m != 512) return false;
  if (s.inputs[0].rank() != 2 && s.inputs[0].rank() != 3) return false;
  if (s.inputs[0].rank() == 3 && s.inputs[0].dim(0) != 1) return false;
  const auto k = s.inputs[1].dim(1) * 8;
  return k > 0 && k % 64 == 0 && s.inputs[0].rank() > 0 &&
         s.inputs[0].dim(s.inputs[0].rank() - 1) == k;
}
}
