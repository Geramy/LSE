#include "lse/core/debug.hpp"

#include <cstdlib>

namespace lse {

namespace {
int g_debug = -1;

int from_env() noexcept {
  const char* e = std::getenv("LSE_DEBUG");
  if (e == nullptr || e[0] == '\0' || (e[0] == '0' && e[1] == '\0')) return 0;
  return 1;
}
}  // namespace

void set_debug(bool on) noexcept { g_debug = on ? 1 : 0; }

bool debug() noexcept {
  if (g_debug < 0) g_debug = from_env();
  return g_debug != 0;
}

}  // namespace lse
