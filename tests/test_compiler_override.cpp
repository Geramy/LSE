#include "lse/backends/hrx/compiler_image_identity.hpp"
#include <cstdlib>
#include <iostream>
extern "C" int lse_compiler_identity_anchor();
int main(int argc, char **argv) {
  if (argc != 3)
    return 2;
  if (lse_compiler_identity_anchor() != std::atoi(argv[1]))
    return 3;
  const auto id = lse::backend::detail::compiler_image_identity(
      reinterpret_cast<const void *>(&lse_compiler_identity_anchor));
  std::error_code ec;
  const auto expected = std::filesystem::canonical(argv[2], ec);
  if (ec || id.find("file=" + expected.string() + " bytes=") != 0)
    return 4;
  if (id.find("process=") != std::string::npos)
    return 5;
#if defined(__APPLE__)
  if (id.find("resident_uuid=") == std::string::npos)
    return 6;
#endif
  std::cout << id << '\n';
  return 0;
}
