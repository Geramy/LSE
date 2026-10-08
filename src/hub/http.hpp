// The hub client's HTTP layer: one request at a time over the Rust shim's
// ureq/rustls transport, with redirects followed here so the headers of the
// first answer (where the hub puts a file's identity) stay visible.
#pragma once

#include <cstddef>
#include <functional>
#include <map>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "lse/core/status.hpp"

namespace lse::hub::http {

using Headers = std::vector<std::pair<std::string, std::string>>;

struct Response {
  int status = 0;
  std::map<std::string, std::string> headers;  // lowercase names
  std::string url;        // the URL that produced this answer
  // The first answer of a redirect chain, whose headers name the file.
  std::map<std::string, std::string> first_headers;

  [[nodiscard]] std::string header(const std::string& name) const;
  [[nodiscard]] std::string first(const std::string& name) const;
};

// Body bytes in order; returning false aborts the transfer (the request then
// fails with whatever status the sink recorded, or kCancelled).
using Sink = std::function<bool(std::span<const std::byte>)>;
// Sees the final answer's status and headers before its body; false aborts.
using HeadCheck = std::function<bool(const Response&)>;

// Sends `method` to `url`, following up to ten redirects (relative ones too).
// The Authorization header is dropped when a redirect leaves the origin's
// host, as a browser and huggingface_hub do: the CDN URL is pre-signed.
Result<Response> request(const std::string& method, const std::string& url,
                         const Headers& headers, const HeadCheck& head,
                         const Sink& sink);

// GET into memory, refusing a body larger than `limit` bytes.
Result<Response> get(const std::string& url, const Headers& headers,
                     std::string* body, std::size_t limit = 64u << 20);

// "scheme://host[:port]" of a URL.
std::string origin(const std::string& url);

// Percent-encodes one path segment (or several, keeping '/').
std::string encode_path(std::string_view path, bool keep_slash);

}  // namespace lse::hub::http
