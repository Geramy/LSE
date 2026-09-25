#pragma once
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <vector>
#include "lse/core/dtype.hpp"
#include "lse/core/status.hpp"

namespace lse::backend::hrx_kernels {
inline Result<std::vector<std::byte>> matrix_probe_ones(
    DType storage, bool activation, std::size_t bytes) {
  std::uint32_t bits = 0;
  std::size_t width = 0;
  if (storage == DType::kI32) { bits=0x01010101u; width=4; }
  else if (activation && (storage==DType::kF16 || storage==DType::kBF16)) {
    bits=0x3f800000u; width=4;
  } else if (storage==DType::kF16) { bits=0x3c00u; width=2; }
  else if (storage==DType::kBF16) { bits=0x3f80u; width=2; }
  else return LSE_ERROR(kUnimplemented,"no exact matrix-probe operand seed");
  if (bytes==0 || bytes%width) return LSE_ERROR(kInvalidArgument,"partial matrix-probe operand");
  std::vector<std::byte> result(bytes);
  for(std::size_t i=0;i<bytes;i+=width)std::memcpy(result.data()+i,&bits,width);
  return result;
}
inline Status check_matrix_probe_output(std::span<const float> values,
    std::size_t guard, std::size_t payload, float expected) {
  if(!guard || !payload || values.size()!=payload+2*guard)
    return LSE_ERROR(kInvalidArgument,"invalid guarded matrix output");
  for(std::size_t i=0;i<values.size();++i) {
    const float want=i>=guard && i<guard+payload?expected:-1234.0f;
    if(values[i]!=want)return LSE_ERROR(kDeviceError,"matrix probe output mismatch at ",std::to_string(i));
  }
  return OkStatus();
}
} // namespace lse::backend::hrx_kernels
