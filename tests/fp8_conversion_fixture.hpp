#pragma once
#include "loom_dot_fixture.hpp"
#include "lse/math/fp8.hpp"
namespace fp8_fixture {
using namespace lse;
template<math::MatrixElem E> Result<backend::LoomBody> body(unsigned byte) {
  auto types = backend::loom_types();
  auto sources = backend::loom_sources();
  ir::KernelBody b(types, sources);
  ir::env::Emit e{&b};
  ir::Buffer<ir::f32> values(&b, &types, "values");
  auto i = e.thread_id();
  auto x = e.let(values[i].read());
  auto packed = e.let(math::pack_fp8<E>(x, e.f32(0.0f) - x, x + e.f32(1.0f), x - e.f32(1.0f)));
  if (byte == 0) e.ret(math::unpack_fp8<E, 0>(packed));
  if (byte == 1) e.ret(math::unpack_fp8<E, 1>(packed));
  if (byte == 2) e.ret(math::unpack_fp8<E, 2>(packed));
  if (byte == 3) e.ret(math::unpack_fp8<E, 3>(packed));
  backend::LoomPrintOptions options;
  options.buffers.emplace("values", backend::LoomBufferView{ir::Scalar::kF32, 128, "%values_view"});
  return backend::loom_print(b.ir(), options);
}
}
