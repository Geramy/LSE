#include "lse/ir/spell.hpp"

#include <iomanip>
#include <sstream>

namespace lse::ir {

std::string float_literal(float v) {
  std::ostringstream os;
  os.setf(std::ios::showpoint);
  os << std::setprecision(9) << v << "f";
  return os.str();
}

std::string literal_u32(unsigned int v) { return std::to_string(v) + "u"; }
std::string literal_i32(int v) { return std::to_string(v); }

std::string substitute(std::string_view tmpl, std::span<const std::string> args) {
  return substitute(tmpl, args, {});
}

std::string substitute(std::string_view tmpl, std::span<const std::string> args,
                       std::span<const float> attrs) {
  std::string out;
  out.reserve(tmpl.size() + args.size() * 8);
  for (std::size_t i = 0; i < tmpl.size(); ++i) {
    if (tmpl[i] != '$' || i + 1 >= tmpl.size()) {
      out += tmpl[i];
      continue;
    }
    // "$aN" splices attrs[N] as a literal; "$N" splices input N.
    const bool is_attr = tmpl[i + 1] == 'a';
    std::size_t j = i + 1 + (is_attr ? 1 : 0);
    std::size_t index = 0;
    bool digits = false;
    while (j < tmpl.size() && tmpl[j] >= '0' && tmpl[j] <= '9') {
      index = index * 10 + static_cast<std::size_t>(tmpl[j] - '0');
      ++j;
      digits = true;
    }
    if (!digits) {
      out += tmpl[i];
      continue;
    }
    if (is_attr) {
      if (index < attrs.size()) out += float_literal(attrs[index]);
    } else if (index < args.size()) {
      // Parenthesize: a template like "$0 * $0" must not re-associate when the
      // argument is itself an expression.
      out += '(';
      out += args[index];
      out += ')';
    }
    i = j - 1;
  }
  return out;
}

}  // namespace lse::ir
