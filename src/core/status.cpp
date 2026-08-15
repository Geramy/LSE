// Status code names and formatting. Kept separate from dtype.cpp so the status
// vocabulary does not depend on the numeric type layer.
#include "lse/core/status.hpp"

namespace lse {


std::string Status::to_string() const {
  std::string out(lse::to_string(code_));
  if (!message_.empty()) {
    out += ": ";
    out += message_;
  }
  return out;
}

}  // namespace lse
