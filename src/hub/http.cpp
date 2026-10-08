#include "http.hpp"

#include <cctype>
#include <cstdint>
#include <cstdio>
#include <sstream>

#include "fastokens_ffi.h"

namespace lse::hub::http {

namespace {

std::map<std::string, std::string> parse_headers(const char* text) {
  std::map<std::string, std::string> out;
  std::istringstream in(text != nullptr ? text : "");
  std::string line;
  while (std::getline(in, line)) {
    const std::size_t colon = line.find(':');
    if (colon == std::string::npos) continue;
    std::string name = line.substr(0, colon);
    std::string value = line.substr(colon + 1);
    while (!value.empty() && (value.front() == ' ' || value.front() == '\t')) value.erase(0, 1);
    while (!value.empty() && (value.back() == '\r' || value.back() == ' ')) value.pop_back();
    for (char& c : name) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    auto [it, fresh] = out.emplace(name, value);
    if (!fresh) it->second += ", " + value;
  }
  return out;
}

struct Call {
  const HeadCheck* head = nullptr;
  const Sink* sink = nullptr;
  Response* response = nullptr;
  bool first_hop = false;
  bool redirect = false;  // a 3xx: its body is not the caller's
  bool aborted_by_caller = false;
};

int on_head(void* ctx, int status, const char* headers) {
  auto* call = static_cast<Call*>(ctx);
  call->response->status = status;
  call->response->headers = parse_headers(headers);
  if (call->first_hop) call->response->first_headers = call->response->headers;
  call->redirect = status >= 300 && status < 400 && call->response->headers.contains("location");
  if (call->redirect) return 0;
  if (*call->head && !(*call->head)(*call->response)) {
    call->aborted_by_caller = true;
    return 1;
  }
  return 0;
}

int on_body(void* ctx, const std::uint8_t* data, std::size_t len) {
  auto* call = static_cast<Call*>(ctx);
  if (call->redirect) return 0;
  if (*call->sink && !(*call->sink)(std::as_bytes(std::span(data, len)))) {
    call->aborted_by_caller = true;
    return 1;
  }
  return 0;
}

std::string host_of(const std::string& url) {
  const std::size_t scheme = url.find("://");
  const std::size_t start = scheme == std::string::npos ? 0 : scheme + 3;
  const std::size_t end = url.find_first_of("/?#", start);
  return url.substr(start, end == std::string::npos ? std::string::npos : end - start);
}

std::string resolve_location(const std::string& base, const std::string& location) {
  if (location.find("://") != std::string::npos) return location;
  if (location.starts_with("//")) return base.substr(0, base.find("://") + 1) + location;
  if (location.starts_with("/")) return origin(base) + location;
  // Relative to the base path's directory.
  const std::size_t query = base.find_first_of("?#");
  std::string path = base.substr(0, query);
  path.resize(path.rfind('/') + 1);
  return path + location;
}

}  // namespace

std::string Response::header(const std::string& name) const {
  const auto it = headers.find(name);
  return it == headers.end() ? std::string() : it->second;
}

std::string Response::first(const std::string& name) const {
  const auto it = first_headers.find(name);
  return it == first_headers.end() ? std::string() : it->second;
}

std::string origin(const std::string& url) {
  const std::size_t scheme = url.find("://");
  if (scheme == std::string::npos) return {};
  const std::size_t end = url.find_first_of("/?#", scheme + 3);
  return url.substr(0, end);
}

std::string encode_path(std::string_view path, bool keep_slash) {
  static const char* hex = "0123456789ABCDEF";
  std::string out;
  for (const char ch : path) {
    const auto c = static_cast<unsigned char>(ch);
    if (std::isalnum(c) != 0 || c == '-' || c == '_' || c == '.' || c == '~' ||
        (keep_slash && c == '/')) {
      out += static_cast<char>(c);
    } else {
      out += '%';
      out += hex[c >> 4];
      out += hex[c & 15];
    }
  }
  return out;
}

Result<Response> request(const std::string& method, const std::string& url,
                         const Headers& headers, const HeadCheck& head,
                         const Sink& sink) {
  const std::string start_host = host_of(url);
  std::string current = url;
  Response response;
  for (int hop = 0; hop <= 10; ++hop) {
    std::string lines;
    const bool same_host = host_of(current) == start_host;
    for (const auto& [name, value] : headers) {
      if (!same_host && (name == "Authorization" || name == "authorization")) continue;
      lines += name + ": " + value + "\n";
    }
    Response hop_response;
    hop_response.first_headers = response.first_headers;
    Call call{&head, &sink, &hop_response, hop == 0};
    const int rc = fk_http_request(method.c_str(), current.c_str(), lines.c_str(), 0, 0,
                                   on_head, on_body, &call);
    hop_response.url = current;
    if (rc == -1) {
      return LSE_ERROR(kIoError, method, " ", current, " failed: ", fk_last_error());
    }
    response = std::move(hop_response);
    if (rc == -2 || call.aborted_by_caller) {
      return LSE_ERROR(kCancelled, method, " ", current, " was stopped");
    }
    if (!call.redirect) return response;
    current = resolve_location(current, response.header("location"));
  }
  return LSE_ERROR(kIoError, method, " ", url, " redirected more than ten times");
}

Result<Response> get(const std::string& url, const Headers& headers, std::string* body,
                     std::size_t limit) {
  body->clear();
  bool too_large = false;
  auto got = request("GET", url, headers, {}, [&](std::span<const std::byte> bytes) {
    if (body->size() + bytes.size() > limit) {
      too_large = true;
      return false;
    }
    body->append(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    return true;
  });
  if (too_large) {
    return LSE_ERROR(kOutOfRange, "GET ", url, " returned more than ",
                     std::to_string(limit), " bytes");
  }
  return got;
}

}  // namespace lse::hub::http
