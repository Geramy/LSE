// Downloads from the Hugging Face hub into the cache layout huggingface_hub
// writes. The HTTP API is used directly (no Python): the revision's metadata
// names every file with its size and hash, the config and safetensors headers
// decide whether LSE loads the repo before any weights move, and each file is
// streamed into blobs/<etag>.incomplete, resumed with a Range request when one
// is already there, hashed while it arrives and renamed into place only when
// the hash matches.
#include <fcntl.h>
#include <fnmatch.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>

#include <nlohmann/json.hpp>

#include "http.hpp"
#include "sha1.hpp"
#include "lse/core/sha256.hpp"
#include "lse/hub/hub.hpp"
#include "lse/model/weights.hpp"

#ifndef LSE_ENGINE_VERSION
#define LSE_ENGINE_VERSION "dev"
#endif

namespace lse::hub {

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

// One hash or the other, fed the same bytes.
class Hasher {
 public:
  Hasher(bool lfs, std::uint64_t size) : lfs_(lfs) {
    if (!lfs_) {
      const std::string prefix = "blob " + std::to_string(size) + '\0';
      sha1_.update(std::as_bytes(std::span(prefix.data(), prefix.size())));
    }
  }
  void update(std::span<const std::byte> b) noexcept { lfs_ ? sha256_.update(b) : sha1_.update(b); }
  std::string hex() noexcept {
    if (!lfs_) return sha1_.hex();
    const auto d = sha256_.finish();
    static const char* digits = "0123456789abcdef";
    std::string out;
    for (std::byte b : d) {
      out += digits[std::to_integer<int>(b) >> 4];
      out += digits[std::to_integer<int>(b) & 15];
    }
    return out;
  }
  [[nodiscard]] const char* name() const { return lfs_ ? "sha256" : "git blob sha1"; }

 private:
  bool lfs_;
  Sha256 sha256_;
  Sha1 sha1_;
};

struct File {
  std::string path;  // repo-relative, '/'-separated
  std::uint64_t size = 0;
  std::string blob_id;
  std::string sha256;  // LFS only
  [[nodiscard]] bool lfs() const { return !sha256.empty(); }
  [[nodiscard]] std::string etag() const { return lfs() ? sha256 : blob_id; }
};

struct RepoInfo {
  std::string id, commit;
  bool gated = false;
  std::vector<File> files;
};

std::string human(std::uint64_t n) {
  const char* unit[] = {"B", "KiB", "MiB", "GiB", "TiB"};
  double v = static_cast<double>(n);
  int u = 0;
  while (v >= 1024.0 && u < 4) {
    v /= 1024.0;
    ++u;
  }
  char buf[32];
  std::snprintf(buf, sizeof buf, u == 0 ? "%.0f %s" : "%.1f %s", v, unit[u]);
  return buf;
}

http::Headers headers(bool auth) {
  http::Headers h{{"User-Agent", std::string("lse/") + LSE_ENGINE_VERSION},
                  {"Accept-Encoding", "identity"}};
  if (auth) {
    if (auto t = token()) h.emplace_back("Authorization", "Bearer " + t->value);
  }
  return h;
}

std::string strip_etag(std::string e) {
  if (e.starts_with("W/")) e.erase(0, 2);
  if (e.size() >= 2 && e.front() == '"' && e.back() == '"') e = e.substr(1, e.size() - 2);
  return e;
}

// What a hub error answer means, worded for the person who has to fix it.
Status hub_error(const http::Response& r, const RepoSpec& spec, const std::string& what) {
  const std::string code = r.header("x-error-code");
  const std::string msg = r.header("x-error-message");
  const auto tok = token();
  const std::string page = endpoint() + "/" + spec.repo_id;
  if (code == "GatedRepo") {
    if (tok) {
      return LSE_ERROR(kInvalidArgument, spec.repo_id, " is gated and the token from ", tok->source,
                       " has no access to it; request access at ", page);
    }
    return LSE_ERROR(kInvalidArgument, spec.repo_id, " is gated: accept its terms at ", page,
                     ", then set HF_TOKEN or save a token to ", model::hf_home(),
                     "/token (huggingface-cli login)");
  }
  if (code == "RevisionNotFound") {
    return LSE_ERROR(kNotFound, spec.repo_id, " has no revision '",
                     spec.revision.empty() ? std::string("main") : spec.revision, "' on ", endpoint());
  }
  if (code == "EntryNotFound") {
    return LSE_ERROR(kNotFound, what, ": the hub has no such file (", msg, ")");
  }
  if (code == "RepoNotFound" || r.status == 401 || r.status == 404) {
    return LSE_ERROR(kNotFound, "no repository ", spec.repo_id, " on ", endpoint(),
                     tok ? " (or it is private and the token from " + tok->source + " cannot see it)"
                         : std::string(" (or it is private; no token is set: HF_TOKEN or ") +
                               model::hf_home() + "/token)");
  }
  if (r.status == 429) {
    return LSE_ERROR(kIoError, what, ": the hub is rate-limiting this client (HTTP 429)",
                     tok ? std::string() : std::string("; a token (HF_TOKEN) raises the limit"));
  }
  return LSE_ERROR(kIoError, what, ": HTTP ", std::to_string(r.status), msg.empty() ? "" : " (" + msg + ")");
}

Result<RepoInfo> fetch_info(const RepoSpec& spec) {
  const std::string revision = spec.revision.empty() ? "main" : spec.revision;
  const std::string url = endpoint() + "/api/models/" + spec.repo_id + "/revision/" +
                          http::encode_path(revision, false) + "?blobs=true";
  std::string body;
  LSE_ASSIGN_OR(const http::Response r, http::get(url, headers(true), &body));
  if (r.status != 200) return hub_error(r, spec, "reading " + spec.to_string() + "'s file list");
  RepoInfo info;
  try {
    const json j = json::parse(body);
    info.id = j.value("id", spec.repo_id);
    info.commit = j.value("sha", std::string());
    const auto g = j.find("gated");
    info.gated = g != j.end() && (g->is_string() || (g->is_boolean() && g->get<bool>()));
    for (const json& s : j.at("siblings")) {
      File f;
      f.path = s.at("rfilename").get<std::string>();
      f.blob_id = s.value("blobId", std::string());
      if (const auto lfs = s.find("lfs"); lfs != s.end() && lfs->is_object()) {
        f.sha256 = lfs->value("sha256", lfs->value("oid", std::string()));
        f.size = lfs->value("size", std::uint64_t{0});
      }
      if (s.contains("size")) f.size = s["size"].get<std::uint64_t>();
      if (!s.contains("size") && !f.lfs()) {
        return LSE_ERROR(kIoError, "the hub's file list for ", spec.to_string(), " gives no size for ", f.path);
      }
      if (f.etag().empty()) {
        return LSE_ERROR(kIoError, "the hub's file list for ", spec.to_string(), " gives no hash for ", f.path);
      }
      if (f.path.find("..") != std::string::npos || f.path.starts_with("/")) {
        return LSE_ERROR(kIoError, spec.to_string(), " lists an unsafe path '", f.path, "'");
      }
      info.files.push_back(std::move(f));
    }
  } catch (const std::exception& e) {
    return LSE_ERROR(kIoError, "the hub's answer for ", spec.to_string(), " is not the expected JSON: ", e.what());
  }
  if (info.commit.size() != 40) {
    return LSE_ERROR(kIoError, "the hub's answer for ", spec.to_string(), " names no commit");
  }
  return info;
}

std::string resolve_url(const RepoInfo& info, const std::string& path) {
  return endpoint() + "/" + info.id + "/resolve/" + info.commit + "/" + http::encode_path(path, true);
}

bool matches(const std::string& path, const std::vector<std::string>& globs) {
  for (std::string g : globs) {
    if (g.ends_with("/")) g += "*";
    if (::fnmatch(g.c_str(), path.c_str(), 0) == 0) return true;
  }
  return false;
}

// Up to `bytes` from the start of a file, by Range request.
Result<std::string> fetch_prefix(const RepoInfo& info, const RepoSpec& spec, const File& f,
                                 std::uint64_t from, std::uint64_t bytes) {
  auto h = headers(true);
  h.emplace_back("Range", "bytes=" + std::to_string(from) + "-" + std::to_string(from + bytes - 1));
  std::string body;
  Status failed;
  auto r = http::request(
      "GET", resolve_url(info, f.path), h,
      [&](const http::Response& resp) {
        if (resp.status == 206 || resp.status == 200) return true;
        failed = hub_error(resp, spec, "reading " + f.path);
        return false;
      },
      [&](std::span<const std::byte> b) {
        body.append(reinterpret_cast<const char*>(b.data()), b.size());
        return body.size() < bytes;  // a server that ignores Range is cut off here
      });
  if (!failed.ok()) return failed;
  if (!r.ok() && r.status().code() != StatusCode::kCancelled) return r.status();
  if (r.ok() && r->status == 200 && from != 0) {
    return LSE_ERROR(kIoError, "the server ignored a Range request for ", f.path);
  }
  if (body.size() < bytes) {
    return LSE_ERROR(kIoError, "reading ", f.path, ": got ", std::to_string(body.size()), " of ",
                     std::to_string(bytes), " bytes");
  }
  body.resize(bytes);
  return body;
}

struct TempDir {
  fs::path path;
  TempDir() {
    static std::atomic<int> n{0};
    path = fs::temp_directory_path() /
           ("lse-hub-probe-" + std::to_string(::getpid()) + "-" + std::to_string(n++));
    std::error_code ec;
    fs::remove_all(path, ec);
    fs::create_directories(path, ec);
  }
  ~TempDir() {
    std::error_code ec;
    fs::remove_all(path, ec);
  }
};

// Whether LSE loads the repo, decided before its weights are fetched: the
// config and index are downloaded, and each top-level safetensors file is
// recreated sparsely — its real header, then a hole to its real size — so the
// engine's own inspection (architecture detection, quantization rules, the
// MTP/DFlash2 checks) runs on exactly what the download would produce.
Result<Checkpoint> probe(const RepoInfo& info, const RepoSpec& spec) {
  const auto has = [&](std::string_view ext) {
    return std::any_of(info.files.begin(), info.files.end(), [&](const File& f) {
      return f.path.find('/') == std::string::npos && f.path.ends_with(ext);
    });
  };
  const auto find = [&](const std::string& path) -> const File* {
    for (const File& f : info.files)
      if (f.path == path) return &f;
    return nullptr;
  };
  if (!has(".safetensors")) {
    if (has(".gguf")) {
      return LSE_ERROR(kUnimplemented, spec.repo_id,
                       " is a GGUF repository; LSE loads MLX safetensors checkpoints (an "
                       "mlx-community build of the model is the usual source)");
    }
    return LSE_ERROR(kUnimplemented, spec.repo_id,
                     " has no .safetensors weights at its top level; LSE loads MLX safetensors checkpoints");
  }
  if (find("config.json") == nullptr) {
    return LSE_ERROR(kUnimplemented, spec.repo_id, " has no config.json; LSE cannot tell what model it is");
  }
  TempDir tmp;
  for (const File& f : info.files) {
    if (f.path.find('/') != std::string::npos) continue;
    std::string bytes;
    if (f.path == "config.json" || f.path.ends_with(".safetensors.index.json") ||
        f.path == "generation_config.json") {
      if (f.size > (64u << 20)) return LSE_ERROR(kUnimplemented, spec.repo_id, "'s ", f.path, " is implausibly large");
      std::string body;
      LSE_ASSIGN_OR(const http::Response r, http::get(resolve_url(info, f.path), headers(true), &body));
      if (r.status != 200) return hub_error(r, spec, "reading " + f.path);
      bytes = std::move(body);
      std::ofstream(tmp.path / f.path, std::ios::binary) << bytes;
      continue;
    }
    if (!f.path.ends_with(".safetensors")) continue;
    if (f.size < 8) return LSE_ERROR(kUnimplemented, spec.repo_id, "'s ", f.path, " is too short to be safetensors");
    const std::uint64_t first = std::min<std::uint64_t>(f.size, 1u << 20);
    LSE_ASSIGN_OR(std::string head, fetch_prefix(info, spec, f, 0, first));
    std::uint64_t n = 0;
    std::memcpy(&n, head.data(), 8);
    if (n > (256u << 20) || 8 + n > f.size) {
      return LSE_ERROR(kUnimplemented, spec.repo_id, "'s ", f.path, " does not start with a safetensors header");
    }
    if (8 + n > head.size()) {
      LSE_ASSIGN_OR(const std::string rest, fetch_prefix(info, spec, f, head.size(), 8 + n - head.size()));
      head += rest;
    }
    head.resize(8 + n);
    const fs::path out = tmp.path / f.path;
    {
      std::ofstream o(out, std::ios::binary);
      o << head;
      if (!o) return LSE_ERROR(kIoError, "cannot write ", out.string());
    }
    if (::truncate(out.c_str(), static_cast<off_t>(f.size)) != 0) {
      return LSE_ERROR(kIoError, "cannot size ", out.string(), ": ", std::strerror(errno));
    }
  }
  auto c = classify(tmp.path.string());
  if (!c.ok()) return Status(c.status().code(), spec.repo_id + ": " + c.status().message());
  return c;
}

// An exclusive flock on .locks/<repo>/<etag>.lock, the file huggingface_hub's
// FileLock takes, so the two never write one blob at once.
class BlobLock {
 public:
  static Result<BlobLock> take(const fs::path& path, const std::string& label) {
    std::error_code ec;
    fs::create_directories(path.parent_path(), ec);
    const int fd = ::open(path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0644);
    if (fd < 0) return LSE_ERROR(kIoError, "cannot open the lock ", path.string(), ": ", std::strerror(errno));
    if (::flock(fd, LOCK_EX | LOCK_NB) != 0) {
      std::fprintf(stderr, "lse: waiting for another download of %s\n", label.c_str());
      if (::flock(fd, LOCK_EX) != 0) {
        ::close(fd);
        return LSE_ERROR(kIoError, "cannot lock ", path.string(), ": ", std::strerror(errno));
      }
    }
    BlobLock l;
    l.fd_ = fd;
    return l;
  }
  BlobLock() = default;
  BlobLock(BlobLock&& o) noexcept : fd_(std::exchange(o.fd_, -1)) {}
  BlobLock& operator=(BlobLock&&) = delete;
  ~BlobLock() {
    if (fd_ >= 0) {
      ::flock(fd_, LOCK_UN);
      ::close(fd_);
    }
  }

 private:
  int fd_ = -1;
};

struct Tally {
  std::uint64_t done = 0, total = 0;
  std::uint64_t fresh = 0;  // bytes this run fetched, not resumed ones
  std::chrono::steady_clock::time_point started = std::chrono::steady_clock::now();
};

Status download_blob(const RepoInfo& info, const RepoSpec& spec, const File& f, const fs::path& blob,
                     Tally& tally, const ProgressFn& progress) {
  const fs::path part = fs::path(blob.string() + ".incomplete");
  std::error_code ec;
  std::uint64_t offset = fs::exists(part, ec) ? fs::file_size(part, ec) : 0;
  if (offset > f.size) {
    fs::remove(part, ec);
    offset = 0;
  }
  const int fd = ::open(part.c_str(), O_WRONLY | O_CREAT | O_CLOEXEC, 0644);
  if (fd < 0) return LSE_ERROR(kIoError, "cannot open ", part.string(), ": ", std::strerror(errno));
  struct Closer {
    int fd;
    ~Closer() { ::close(fd); }
  } closer{fd};

  Hasher hash(f.lfs(), f.size);
  if (offset > 0) {
    // What is already there counts toward the hash; reading it back is the
    // price of resuming without trusting a partial file blindly.
    std::ifstream in(part, std::ios::binary);
    std::vector<char> buf(1 << 20);
    std::uint64_t left = offset;
    while (left > 0 && in) {
      const auto n = static_cast<std::streamsize>(std::min<std::uint64_t>(left, buf.size()));
      in.read(buf.data(), n);
      const auto got = in.gcount();
      hash.update(std::as_bytes(std::span(buf.data(), static_cast<std::size_t>(got))));
      left -= static_cast<std::uint64_t>(got);
    }
    if (left != 0) return LSE_ERROR(kIoError, "cannot read back ", part.string());
    std::fprintf(stderr, "lse: resuming %s at %s of %s\n", f.path.c_str(), human(offset).c_str(),
                 human(f.size).c_str());
  }
  if (::lseek(fd, static_cast<off_t>(offset), SEEK_SET) < 0 || ::ftruncate(fd, static_cast<off_t>(offset)) != 0)
    return LSE_ERROR(kIoError, "cannot position ", part.string(), ": ", std::strerror(errno));
  tally.done += offset;

  std::uint64_t written = offset;
  Status failed;
  Progress p;
  p.repo_id = info.id;
  p.file = f.path;
  p.file_total = f.size;
  p.total = tally.total;
  const auto report = [&](bool finished) {
    if (!progress) return;
    p.file_done = written;
    p.done = tally.done;
    const double secs =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - tally.started).count();
    p.bytes_per_second = secs > 0 ? static_cast<double>(tally.fresh) / secs : 0;
    p.file_finished = finished;
    progress(p);
  };

  if (offset < f.size) {
    auto h = headers(true);
    if (offset > 0) h.emplace_back("Range", "bytes=" + std::to_string(offset) + "-");
    auto r = http::request(
        "GET", resolve_url(info, f.path), h,
        [&](const http::Response& resp) {
          // The file's identity is on the hub's own answer, before any CDN
          // redirect: X-Linked-Etag for an LFS file, ETag otherwise.
          std::string served = strip_etag(resp.first("x-linked-etag"));
          if (served.empty()) served = strip_etag(resp.first("etag"));
          if (!served.empty() && served != f.etag()) {
            failed = LSE_ERROR(kIoError, "etag mismatch for ", f.path, ": the hub now serves ", served,
                               " where the file list for ", info.commit.substr(0, 12), " named ", f.etag(),
                               "; the repo changed during the download, run the pull again");
            return false;
          }
          if (resp.status == 206) {
            const std::string range = resp.header("content-range");
            if (!range.starts_with("bytes " + std::to_string(offset) + "-")) {
              failed = LSE_ERROR(kIoError, "the server answered a resume of ", f.path, " at byte ",
                                 std::to_string(offset), " with Content-Range '", range, "'");
              return false;
            }
            return true;
          }
          if (resp.status == 200) {
            if (offset > 0) {
              // The whole file instead of the rest of it: start over.
              std::fprintf(stderr, "lse: the server does not resume %s; restarting it\n", f.path.c_str());
              tally.done -= offset;
              offset = written = 0;
              hash = Hasher(f.lfs(), f.size);
              if (::lseek(fd, 0, SEEK_SET) < 0 || ::ftruncate(fd, 0) != 0) {
                failed = LSE_ERROR(kIoError, "cannot truncate ", part.string());
                return false;
              }
            }
            return true;
          }
          failed = hub_error(resp, spec, "downloading " + f.path);
          return false;
        },
        [&](std::span<const std::byte> b) {
          if (written + b.size() > f.size) {
            failed = LSE_ERROR(kIoError, "the server sent more than the ", std::to_string(f.size),
                               " bytes the hub lists for ", f.path);
            return false;
          }
          const auto* data = reinterpret_cast<const char*>(b.data());
          std::size_t left = b.size();
          while (left > 0) {
            const ssize_t n = ::write(fd, data, left);
            if (n < 0) {
              if (errno == EINTR) continue;
              failed = LSE_ERROR(kIoError, "writing ", part.string(), ": ", std::strerror(errno));
              return false;
            }
            data += n;
            left -= static_cast<std::size_t>(n);
          }
          hash.update(b);
          written += b.size();
          tally.done += b.size();
          tally.fresh += b.size();
          report(false);
          return true;
        });
    if (!failed.ok()) return failed;
    if (!r.ok()) {
      return Status(r.status().code(), r.status().message() + "; " + human(written) + " of " + f.path +
                                           " kept in " + part.string() + ", run the pull again to resume");
    }
  }
  if (written != f.size) {
    return LSE_ERROR(kIoError, "the download of ", f.path, " ended at ", std::to_string(written), " of ",
                     std::to_string(f.size), " bytes; the partial file is kept in ", part.string(),
                     ", run the pull again to resume");
  }
  const std::string got = hash.hex();
  if (got != f.etag()) {
    fs::remove(part, ec);
    return LSE_ERROR(kIoError, "checksum mismatch for ", f.path, ": ", hash.name(), " ", got, ", expected ",
                     f.etag(), "; the corrupt download was deleted, run the pull again");
  }
  if (::fsync(fd) != 0) return LSE_ERROR(kIoError, "fsync ", part.string(), ": ", std::strerror(errno));
  fs::rename(part, blob, ec);
  if (ec) return LSE_ERROR(kIoError, "cannot move ", part.string(), " into place: ", ec.message());
  report(true);
  return OkStatus();
}

Status link_pointer(const fs::path& pointer, const fs::path& blob) {
  std::error_code ec;
  fs::create_directories(pointer.parent_path(), ec);
  if (fs::is_symlink(pointer, ec) || fs::exists(pointer, ec)) fs::remove(pointer, ec);
  const fs::path target = blob.lexically_relative(pointer.parent_path());
  fs::create_symlink(target, pointer, ec);
  if (ec) return LSE_ERROR(kIoError, "cannot link ", pointer.string(), " -> ", target.string(), ": ", ec.message());
  return OkStatus();
}

Status write_ref(const fs::path& repo, const std::string& name, const std::string& commit) {
  std::error_code ec;
  const fs::path ref = repo / "refs" / name;
  fs::create_directories(ref.parent_path(), ec);
  const fs::path tmp = fs::path(ref.string() + ".tmp");
  {
    std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
    out << commit;
    if (!out) return LSE_ERROR(kIoError, "cannot write ", tmp.string());
  }
  fs::rename(tmp, ref, ec);
  if (ec) return LSE_ERROR(kIoError, "cannot write ", ref.string(), ": ", ec.message());
  return OkStatus();
}

// Free bytes on the filesystem holding `p` (or its nearest existing parent).
std::uint64_t free_bytes(fs::path p) {
  std::error_code ec;
  while (!p.empty() && !fs::exists(p, ec)) p = p.parent_path();
  struct statvfs st {};
  if (p.empty() || ::statvfs(p.c_str(), &st) != 0) return UINT64_MAX;
  return static_cast<std::uint64_t>(st.f_bavail) * static_cast<std::uint64_t>(st.f_frsize);
}

}  // namespace

ProgressFn stderr_progress(std::string prefix) {
  const bool tty = ::isatty(STDERR_FILENO) != 0;
  auto last = std::make_shared<std::chrono::steady_clock::time_point>();
  return [prefix = std::move(prefix), tty, last](const Progress& p) {
    const auto now = std::chrono::steady_clock::now();
    const auto gap = tty ? std::chrono::milliseconds(200) : std::chrono::milliseconds(5000);
    if (!p.file_finished && now - *last < gap) return;
    *last = now;
    const double pct = p.total > 0 ? 100.0 * static_cast<double>(p.done) / static_cast<double>(p.total) : 100.0;
    char line[512];
    std::snprintf(line, sizeof line, "%s%s %s/%s (%.0f%%), %s/s", prefix.c_str(), p.file.c_str(),
                  human(p.done).c_str(), human(p.total).c_str(), pct,
                  human(static_cast<std::uint64_t>(p.bytes_per_second)).c_str());
    if (tty) {
      std::fprintf(stderr, "\r\033[K%s%s", line, p.file_finished && p.done >= p.total ? "\n" : "");
    } else {
      std::fprintf(stderr, "%s\n", line);
    }
  };
}

Result<PullResult> pull(const RepoSpec& spec, const PullOptions& options) {
  if (offline()) {
    return LSE_ERROR(kInvalidArgument, "HF_HUB_OFFLINE is set (or --offline was given); pulling ",
                     spec.to_string(), " needs the network");
  }
  LSE_ASSIGN_OR(const RepoInfo info, fetch_info(spec));
  if (info.gated && !token()) {
    return LSE_ERROR(kInvalidArgument, spec.repo_id, " is gated: accept its terms at ", endpoint(), "/",
                     spec.repo_id, ", then set HF_TOKEN or save a token to ", model::hf_home(),
                     "/token (huggingface-cli login)");
  }
  if (!options.pinned_file.empty()) {
    const auto it = std::find_if(info.files.begin(), info.files.end(),
                                 [&](const File& f) { return f.path == options.pinned_file; });
    if (it == info.files.end() || it->sha256 != options.pinned_sha256) {
      return LSE_ERROR(kInvalidArgument, spec.to_string(), "'s ", options.pinned_file, " is ",
                       it == info.files.end() ? std::string("missing") : "sha256 " + it->sha256,
                       ", not the ", options.pinned_sha256, " LSE was validated against");
    }
  }

  LSE_ASSIGN_OR(Checkpoint probed, probe(info, spec));
  if (std::find(options.accept.begin(), options.accept.end(), probed.kind) == options.accept.end()) {
    std::string want;
    for (Kind k : options.accept) want += (want.empty() ? "" : " or ") + std::string(to_string(k));
    return LSE_ERROR(kInvalidArgument, spec.repo_id, " is a ", std::string(to_string(probed.kind)),
                     " checkpoint where a ", want, " was asked for");
  }
  if (options.on_checked) options.on_checked(probed);

  std::vector<const File*> selected;
  for (const File& f : info.files) {
    if (!options.include.empty() && !matches(f.path, options.include)) continue;
    if (matches(f.path, options.exclude)) continue;
    selected.push_back(&f);
  }
  if (selected.empty()) {
    return LSE_ERROR(kInvalidArgument, "--include/--exclude leave none of ", spec.to_string(), "'s ",
                     std::to_string(info.files.size()), " files");
  }

  const fs::path root(model::hf_cache_root());
  const fs::path repo = repo_dir(info.id);
  const fs::path blobs = repo / "blobs";
  const fs::path snap = repo / "snapshots" / info.commit;
  const fs::path locks = root / ".locks" / repo.filename();
  std::error_code ec;

  PullResult out;
  out.repo_id = info.id;
  out.commit = info.commit;
  out.snapshot = snap.string();
  Tally tally;
  std::vector<const File*> fetch;
  std::uint64_t need = 0;
  for (const File* f : selected) {
    out.bytes += f->size;
    const fs::path pointer = snap / f->path;
    const fs::path blob = blobs / f->etag();
    if (fs::exists(pointer, ec) && fs::file_size(pointer, ec) == f->size) {
      ++out.reused;
      continue;
    }
    if (fs::exists(blob, ec) && fs::file_size(blob, ec) == f->size) {
      LSE_RETURN_IF_ERROR(link_pointer(pointer, blob));
      ++out.reused;
      continue;
    }
    const fs::path part(blob.string() + ".incomplete");
    const std::uint64_t have = fs::exists(part, ec) ? std::min<std::uint64_t>(fs::file_size(part, ec), f->size) : 0;
    need += f->size - have;
    tally.total += f->size;
    fetch.push_back(f);
  }
  const std::uint64_t room = free_bytes(blobs);
  if (need > room) {
    return LSE_ERROR(kOutOfMemory, "not enough disk space for ", spec.to_string(), ": ", human(need),
                     " still to download, ", human(room), " free on the filesystem holding ", root.string());
  }
  std::sort(fetch.begin(), fetch.end(), [](const File* a, const File* b) {
    return std::pair(a->size, a->path) < std::pair(b->size, b->path);
  });
  for (const File* f : fetch) {
    const fs::path blob = blobs / f->etag();
    fs::create_directories(blobs, ec);
    LSE_ASSIGN_OR(BlobLock lock, BlobLock::take(locks / (f->etag() + ".lock"), f->path));
    if (!(fs::exists(blob, ec) && fs::file_size(blob, ec) == f->size)) {
      const std::uint64_t before = tally.fresh;
      LSE_RETURN_IF_ERROR(download_blob(info, spec, *f, blob, tally, options.progress));
      out.downloaded += tally.fresh - before;
    } else {
      tally.done += f->size;
    }
    LSE_RETURN_IF_ERROR(link_pointer(snap / f->path, blob));
  }
  out.files = selected.size();
  // A ref is written for a named branch or tag, never for a commit hash: what
  // huggingface_hub does, so `main` keeps meaning what it last meant.
  const std::string revision = spec.revision.empty() ? "main" : spec.revision;
  if (revision != info.commit) LSE_RETURN_IF_ERROR(write_ref(repo, revision, info.commit));

  auto c = classify(snap.string());
  if (!c.ok()) {
    if (options.include.empty() && options.exclude.empty()) {
      return Status(c.status().code(), "downloaded " + spec.to_string() + " but LSE cannot load it: " +
                                           c.status().message());
    }
    out.checkpoint = probed;
    out.checkpoint.path = snap.string();
    out.checkpoint.repo_id = info.id;
    out.checkpoint.commit = info.commit;
    return out;
  }
  out.checkpoint = c.release();
  return out;
}

Status pull_if_missing(const std::string& name, std::span<const Kind> accept, const ProgressFn& progress) {
  if (!names_a_repo(name)) return OkStatus();
  if (model::resolve_model(name).ok()) return OkStatus();
  LSE_ASSIGN_OR(const RepoSpec spec, parse_repo_spec(name));
  PullOptions o;
  o.accept.assign(accept.begin(), accept.end());
  o.progress = progress;
  std::fprintf(stderr, "lse: %s is not in the HF cache; pulling it (--pull)\n", name.c_str());
  LSE_ASSIGN_OR(const PullResult r, pull(spec, o));
  std::fprintf(stderr, "lse: pulled %s@%.12s: %zu files, %s\n", r.repo_id.c_str(), r.commit.c_str(), r.files,
               human(r.bytes).c_str());
  return OkStatus();
}

}  // namespace lse::hub
