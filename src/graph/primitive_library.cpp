#include "lse/graph/primitive_library.hpp"

#include <cctype>
#include <fstream>
#include <sstream>

namespace lse::graph {

namespace {

constexpr std::string_view kMarker = "LSE_PRIMITIVE:";

std::string_view trim(std::string_view s) {
  while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front()))) {
    s.remove_prefix(1);
  }
  while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back()))) {
    s.remove_suffix(1);
  }
  return s;
}

// The identifier immediately before the '(' of the next function definition.
std::string function_after(std::string_view source, std::size_t from) {
  const std::size_t paren = source.find('(', from);
  if (paren == std::string_view::npos) return {};
  std::size_t end = paren;
  while (end > from && std::isspace(static_cast<unsigned char>(source[end - 1]))) {
    --end;
  }
  std::size_t begin = end;
  while (begin > from) {
    const char c = source[begin - 1];
    if (std::isalnum(static_cast<unsigned char>(c)) || c == '_') {
      --begin;
    } else {
      break;
    }
  }
  if (begin == end) return {};
  return std::string(source.substr(begin, end - begin));
}

// A primitive whose body calls a function defined in the owning source file.
class LibraryPrimitive final : public Primitive {
 public:
  LibraryPrimitive(LibraryEntry entry, const std::string* source,
                   const std::string* origin)
      : entry_(std::move(entry)), source_(source), origin_(origin) {}

  std::string_view name() const noexcept override { return entry_.primitive_name; }
  std::size_t arity() const noexcept override { return entry_.arity; }
  FusionClass fusion_class() const noexcept override {
    return FusionClass::kElementwise;
  }

  Result<Shape> infer_shape(std::span<const Shape> inputs) const override {
    if (inputs.size() != entry_.arity) {
      return LSE_ERROR(kInvalidArgument, entry_.primitive_name, " takes ",
                       std::to_string(entry_.arity), " inputs, got ",
                       std::to_string(inputs.size()));
    }
    Shape out = inputs[0];
    for (std::size_t i = 1; i < inputs.size(); ++i) {
      out = Shape::broadcast(out, inputs[i]);
      if (out.rank() == 0) {
        return LSE_ERROR(kInvalidArgument, entry_.primitive_name,
                         ": input shapes do not broadcast");
      }
    }
    return out;
  }

  DType infer_dtype(std::span<const DType> inputs) const override {
    return inputs.empty() ? DType::kF32 : inputs[0];
  }

  std::string emit_device(const EmitContext& ctx) const override {
    if (ctx.dialect != Dialect::kHip) return {};
    std::string body(ctx.out);
    body += " = ";
    body += entry_.function_name;
    body += "(";
    for (std::size_t i = 0; i < ctx.inputs.size(); ++i) {
      if (i) body += ", ";
      body += ctx.inputs[i];
    }
    body += ");";
    return body;
  }

  std::string_view device_preamble() const noexcept override { return *source_; }
  std::string_view preamble_id() const noexcept override { return *origin_; }

  bool has_host_impl() const noexcept override { return false; }

  void eval_cpu(std::span<const float* const>, float*, std::size_t,
                const std::array<float, 4>&) const override {}

 private:
  LibraryEntry entry_;
  const std::string* source_;
  const std::string* origin_;
};

}  // namespace

Result<std::vector<LibraryEntry>> parse_primitive_annotations(
    std::string_view source) {
  std::vector<LibraryEntry> out;
  std::size_t pos = 0;
  while ((pos = source.find(kMarker, pos)) != std::string_view::npos) {
    const std::size_t line_end = source.find('\n', pos);
    const std::string_view line =
        trim(source.substr(pos + kMarker.size(),
                           (line_end == std::string_view::npos
                                ? source.size()
                                : line_end) -
                               pos - kMarker.size()));

    LibraryEntry entry;
    std::istringstream fields{std::string(line)};
    std::string token;
    while (fields >> token) {
      if (token.rfind("arity=", 0) == 0) {
        entry.arity = static_cast<std::size_t>(std::atoi(token.c_str() + 6));
      } else if (entry.primitive_name.empty()) {
        entry.primitive_name = token;
      }
    }

    if (entry.primitive_name.empty()) {
      return LSE_ERROR(kInvalidArgument,
                       "LSE_PRIMITIVE annotation is missing a name");
    }
    if (entry.arity == 0 || entry.arity > 4) {
      return LSE_ERROR(kInvalidArgument, "primitive '", entry.primitive_name,
                       "' has arity ", std::to_string(entry.arity),
                       "; supported range is 1..4");
    }

    entry.function_name = function_after(source, line_end);
    if (entry.function_name.empty()) {
      return LSE_ERROR(kInvalidArgument, "no function definition follows the ",
                       "annotation for '", entry.primitive_name, "'");
    }

    out.push_back(std::move(entry));
    pos = line_end == std::string_view::npos ? source.size() : line_end;
  }
  return out;
}

Result<std::shared_ptr<PrimitiveLibrary>> PrimitiveLibrary::load_source(
    std::string source, std::string origin) {
  auto parsed = parse_primitive_annotations(source);
  if (!parsed.ok()) return parsed.status();
  if (parsed->empty()) {
    return LSE_ERROR(kNotFound, "'", origin,
                     "' contains no LSE_PRIMITIVE annotations");
  }

  auto lib = std::make_shared<PrimitiveLibrary>();
  lib->source_ = std::move(source);
  lib->origin_ = std::move(origin);
  lib->entries_ = parsed.release();

  for (const LibraryEntry& e : lib->entries_) {
    auto prim = std::make_unique<LibraryPrimitive>(e, &lib->source_, &lib->origin_);
    // The registry pins the library: primitives outlive the caller's handle.
    LSE_RETURN_IF_ERROR(register_primitive(prim.get(), lib));
    lib->owned_.push_back(std::move(prim));
  }
  return lib;
}

Result<std::shared_ptr<PrimitiveLibrary>> PrimitiveLibrary::load_file(
    const std::string& path) {
  std::ifstream in(path);
  if (!in) return LSE_ERROR(kIoError, "cannot open primitive library '", path, "'");
  std::ostringstream buf;
  buf << in.rdbuf();
  return load_source(buf.str(), path);
}

PrimitiveLibrary::~PrimitiveLibrary() = default;

}  // namespace lse::graph
