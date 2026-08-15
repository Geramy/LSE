// No exceptions on any path hit per-token: fallible operations return Status
// or Result<T>.
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include "lse/core/enum_names.hpp"

namespace lse {

#define LSE_STATUS_CODE_LIST(X)          \
  X(kOk,              "ok")              \
  X(kInvalidArgument, "invalid_argument") \
  X(kOutOfRange,      "out_of_range")    \
  X(kNotFound,        "not_found")       \
  X(kAlreadyExists,   "already_exists")  \
  X(kUnimplemented,   "unimplemented")   \
  X(kInternal,        "internal")        \
  X(kOutOfMemory,     "out_of_memory")   \
  X(kDeviceError,     "device_error")    \
  X(kCompileError,    "compile_error")   \
  X(kIoError,         "io_error")        \
  X(kCancelled,       "cancelled")

LSE_DECLARE_ENUM(StatusCode, std::uint8_t, LSE_STATUS_CODE_LIST)


// The OK status never allocates.
class [[nodiscard]] Status {
 public:
  Status() noexcept = default;

  Status(StatusCode code, std::string message)
      : code_(code), message_(std::move(message)) {}

  explicit Status(StatusCode code) noexcept : code_(code) {}

  [[nodiscard]] bool ok() const noexcept { return code_ == StatusCode::kOk; }
  [[nodiscard]] StatusCode code() const noexcept { return code_; }
  [[nodiscard]] const std::string& message() const noexcept { return message_; }

  [[nodiscard]] std::string to_string() const;

  explicit operator bool() const noexcept { return ok(); }

 private:
  StatusCode code_ = StatusCode::kOk;
  std::string message_;
};

inline Status OkStatus() noexcept { return Status{}; }

#define LSE_ERROR(code, ...) \
  ::lse::Status(::lse::StatusCode::code, ::lse::detail::concat(__VA_ARGS__))

namespace detail {
template <typename... Args>
std::string concat(Args&&... args) {
  std::string out;
  ((out += std::forward<Args>(args)), ...);
  return out;
}
}  // namespace detail

template <typename T>
class [[nodiscard]] Result {
 public:
  Result(T value) : value_(std::move(value)) {}          // NOLINT(*-explicit-*)
  Result(Status status) : status_(std::move(status)) {}  // NOLINT(*-explicit-*)

  [[nodiscard]] bool ok() const noexcept {
    return status_.ok() && value_.has_value();
  }
  [[nodiscard]] const Status& status() const noexcept { return status_; }

  T& operator*() noexcept { return *value_; }
  const T& operator*() const noexcept { return *value_; }
  T* operator->() noexcept { return &*value_; }
  const T* operator->() const noexcept { return &*value_; }

  T value_or(T fallback) const { return value_.value_or(std::move(fallback)); }

  // Only call after ok().
  T release() { return std::move(*value_); }

  explicit operator bool() const noexcept { return ok(); }

 private:
  Status status_;
  std::optional<T> value_;
};

#define LSE_RETURN_IF_ERROR(expr)                    \
  do {                                               \
    ::lse::Status _lse_status = (expr);              \
    if (!_lse_status.ok()) return _lse_status;       \
  } while (0)

#define LSE_CONCAT_INNER(a, b) a##b
#define LSE_CONCAT(a, b) LSE_CONCAT_INNER(a, b)

#define LSE_ASSIGN_OR(lhs, expr)                              \
  auto LSE_CONCAT(_lse_res_, __LINE__) = (expr);              \
  if (!LSE_CONCAT(_lse_res_, __LINE__).ok())                  \
    return LSE_CONCAT(_lse_res_, __LINE__).status();          \
  lhs = LSE_CONCAT(_lse_res_, __LINE__).release()

}  // namespace lse
