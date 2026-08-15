// Loads the golden activations produced by scripts/dump_reference.py.
//
// Regenerate with:
//   source /tmp/env.sh   # lemonseed kernel kill-switches
//   PYTHONPATH=reference/lemonseed .venv/bin/python scripts/dump_reference.py
#pragma once

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#include "lse/model/weights.hpp"

namespace lse::test {

inline std::string reference_path() {
  if (const char* p = std::getenv("LSE_REFERENCE")) return p;
  return "tests/data/reference.safetensors";
}

inline bool have_reference() {
  std::error_code ec;
  return std::filesystem::exists(reference_path(), ec);
}

// Worst absolute and relative error between a computed tensor and the golden
// one. Relative error is only meaningful where the reference is not ~0, so it
// is measured against max(|ref|) over the tensor rather than per element.
struct Deviation {
  double max_abs = 0.0;
  double max_rel = 0.0;
  double ref_absmax = 0.0;
  std::size_t count = 0;
  bool all_finite = true;
};

inline Deviation compare(const std::vector<float>& got,
                         const std::vector<float>& want) {
  Deviation d;
  d.count = std::min(got.size(), want.size());
  for (std::size_t i = 0; i < d.count; ++i) {
    d.ref_absmax = std::max(d.ref_absmax, std::fabs(static_cast<double>(want[i])));
    if (!std::isfinite(got[i])) d.all_finite = false;
  }
  for (std::size_t i = 0; i < d.count; ++i) {
    const double diff = std::fabs(static_cast<double>(got[i]) -
                                  static_cast<double>(want[i]));
    d.max_abs = std::max(d.max_abs, diff);
  }
  d.max_rel = d.ref_absmax > 0 ? d.max_abs / d.ref_absmax : d.max_abs;
  return d;
}

inline std::vector<float> tensor_f32(const model::SafeTensors& st,
                                     const std::string& name) {
  const model::TensorView* v = st.find(name);
  if (v == nullptr) return {};
  std::vector<float> out(v->element_count());
  if (!v->read_f32(out.data(), out.size()).ok()) return {};
  return out;
}

}  // namespace lse::test
