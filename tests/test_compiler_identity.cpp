#include "harness.hpp"
#include "lse/backends/hrx/compiler_image_identity.hpp"
#include <fcntl.h>
#include <fstream>
namespace d = lse::backend::detail;
LSE_TEST(compiler_identity_hashes_equal_size_equal_mtime_rebuilds) {
  const auto path = std::filesystem::temp_directory_path() /
                    ("lse-compiler-identity-" + std::to_string(getpid()));
  {
    std::ofstream out(path);
    out << "compiler version A";
  }
  const timespec times[2] = {{123456789, 123456789}, {123456789, 123456789}};
  LSE_EXPECT(utimensat(AT_FDCWD, path.c_str(), times, 0) == 0);
  const auto first = d::compiler_file_identity(path.c_str());
  LSE_EXPECT(first.has_value());
  LSE_EXPECT(first == d::compiler_file_identity(path.c_str()));
  {
    std::ofstream out(path);
    out << "compiler version B";
  }
  LSE_EXPECT(utimensat(AT_FDCWD, path.c_str(), times, 0) == 0);
  const auto second = d::compiler_file_identity(path.c_str());
  LSE_EXPECT(second.has_value());
  LSE_EXPECT(first != second);
  const timespec nanos[2] = {{123456789, 123456790}, {123456789, 123456790}};
  LSE_EXPECT(utimensat(AT_FDCWD, path.c_str(), nanos, 0) == 0);
  LSE_EXPECT(second != d::compiler_file_identity(path.c_str()));
  std::filesystem::remove(path);
  LSE_EXPECT(!d::compiler_file_identity(path.c_str()));
}
LSE_TEST(unknown_compiler_identity_is_process_scoped_and_stable) {
  const auto first = d::compiler_image_identity(nullptr);
  LSE_EXPECT(first.find("process=") != std::string::npos);
  LSE_EXPECT(first == d::compiler_image_identity(nullptr));
  LSE_EXPECT(!d::compiler_file_identity(nullptr));
  LSE_EXPECT(!d::compiler_file_identity("/"));
}
int main() { return lse::test::run_all(); }
