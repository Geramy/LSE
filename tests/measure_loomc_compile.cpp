// Compiles one cached Loom kernel source with loomc and reports what the
// compile cost the host: peak resident memory (this process does nothing
// else), and from the compiler's accounting the bytes it asked for and its
// largest single allocation, which an on-device compile has to fit (macOS
// maps a large request lazily; iOS refuses it). Never opens a device.
//   measure_loomc_compile FILE.source [ARCH=gfx1201]
#include "lse/backends/hrx/loomc/loomc_compiler.hpp"

#include <sys/resource.h>

#include <chrono>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>


int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: measure_loomc_compile FILE.source [ARCH]\n");
    return 2;
  }
  std::ifstream in(argv[1]);
  std::stringstream text;
  text << in.rdbuf();
  const std::string source = text.str();
  const std::string arch = argc > 2 ? argv[2] : "gfx1201";
  lse::backend::LoomcCompiler compiler;
  rusage ru{};
  getrusage(RUSAGE_SELF, &ru);
  const double before = double(ru.ru_maxrss) / 1048576.0;
  // The compiler logs its own peak and largest allocation (loomc_compiler.cpp).
  setenv("LSE_TRACE_COMPILE_MEMORY", "1", 1);
  const auto t0 = std::chrono::steady_clock::now();
  auto compiled = compiler.compile(source, arch);
  const auto t1 = std::chrono::steady_clock::now();
  getrusage(RUSAGE_SELF, &ru);
  std::printf("%s: %zu bytes of Loom, %s in %.2f s; peak RSS %.1f MiB (%.1f before)\n",
              argv[1], source.size(), compiled.ok() ? "compiled" : compiled.status().to_string().c_str(),
              std::chrono::duration<double>(t1 - t0).count(), double(ru.ru_maxrss) / 1048576.0, before);
  return compiled.ok() ? 0 : 1;
}
