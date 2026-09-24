// Counting what an AMD code object was told to do, by disassembling it.
//
// The disassembler is comgr's, driven one instruction at a time over the
// object's own executable bytes. Nothing here decodes an encoding: writing an
// instruction decoder would be a second, worse copy of the one the toolchain
// already ships, and it would rot on the next ISA.
//
// The ELF walk below exists only to answer the two questions comgr's
// disassembly entry point does not: which byte ranges are code, and which
// kernel each range belongs to. Program addresses are mapped back to file
// offsets through the allocatable sections, because the disassembler asks for
// bytes by address.
#include "lse/backends/hrx/code_object.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#if LSE_HAVE_COMGR

#include <elf.h>

#include <amd_comgr/amd_comgr.h>

namespace lse::backend {

namespace {

// A width the classifier could not name. Such an instruction is counted as
// unclassified rather than as zero bytes: a zero would silently shrink the
// traffic this object performs, which is the one direction that must never
// happen quietly.
constexpr std::uint32_t kWidthUnknown = 0;

// The `n`th underscore-separated token counting back from the end.
std::string_view tail_token(std::string_view mnemonic, std::size_t from_end) {
  std::size_t end = mnemonic.size();
  for (std::size_t i = 0; i <= from_end; ++i) {
    if (end == 0) return {};
    const std::size_t sep = mnemonic.rfind('_', end - 1);
    if (sep == std::string_view::npos) return {};
    if (i == from_end) return mnemonic.substr(sep + 1, end - sep - 1);
    end = sep;
  }
  return {};
}

std::uint32_t width_of_token(std::string_view t) {
  if (t == "b8" || t == "u8" || t == "i8" || t == "d8" || t == "sbyte" ||
      t == "ubyte") {
    return 1;
  }
  if (t == "b16" || t == "u16" || t == "i16" || t == "f16" || t == "d16" ||
      t == "short" || t == "ushort" || t == "sshort") {
    return 2;
  }
  if (t == "b32" || t == "u32" || t == "i32" || t == "f32" || t == "dword") {
    return 4;
  }
  if (t == "b64" || t == "u64" || t == "f64" || t == "dwordx2") return 8;
  if (t == "b96" || t == "dwordx3") return 12;
  if (t == "b128" || t == "dwordx4") return 16;
  if (t == "b256") return 32;
  if (t == "b512") return 64;
  return kWidthUnknown;
}

// The width one issue of `mnemonic` moves, per lane for a vector access and per
// wave for a scalar one.
//
// The SMALLEST width named by the tail wins, because a narrowing suffix always
// follows the register form it narrows: ds_read_u8_d16_hi moves one byte, and
// reading only the last token would call it two.
std::uint32_t access_width(std::string_view mnemonic) {
  std::uint32_t width = kWidthUnknown;
  for (std::size_t i = 0; i < 3; ++i) {
    const std::uint32_t w = width_of_token(tail_token(mnemonic, i));
    if (w != kWidthUnknown && (width == kWidthUnknown || w < width)) width = w;
  }
  if (width == kWidthUnknown) return kWidthUnknown;
  // Two-address LDS forms name one width and read it twice.
  if (mnemonic.find("2addr") != std::string_view::npos) width *= 2;
  return width;
}

bool contains(std::string_view s, std::string_view needle) {
  return s.find(needle) != std::string_view::npos;
}

bool starts_with(std::string_view s, std::string_view p) {
  return s.size() >= p.size() && s.compare(0, p.size(), p) == 0;
}

struct Instruction {
  std::uint64_t address = 0;
  std::string text;
};

std::string_view first_word(std::string_view text) {
  const std::size_t end = text.find_first_of(" \t");
  return end == std::string_view::npos ? text : text.substr(0, end);
}

// Branch displacements are in instruction words and relative to the following
// instruction; llvm prints them as an unsigned 16-bit literal, so a backward
// branch arrives as a large positive number and must be read as signed.
bool branch_target(std::string_view text, std::uint64_t address,
                   std::uint64_t* target) {
  const std::size_t sp = text.find(' ');
  if (sp == std::string_view::npos) return false;
  const std::string operand(text.substr(sp + 1));
  char* end = nullptr;
  const long long parsed = std::strtoll(operand.c_str(), &end, 10);
  if (end == operand.c_str()) return false;
  const auto simm = static_cast<std::int16_t>(parsed);
  *target = address + 4 +
            static_cast<std::uint64_t>(static_cast<std::int64_t>(simm) * 4);
  return true;
}

// Loads left outstanding after this wait, or -1 when the instruction does not
// wait on the vector-memory counter at all.
int vector_memory_wait(std::string_view text) {
  const std::string_view mnemonic = first_word(text);
  std::size_t at = std::string_view::npos;
  if (mnemonic == "s_waitcnt") {
    at = text.find("vmcnt(");
    if (at == std::string_view::npos) return -1;
    at += 6;
  } else if (mnemonic == "s_wait_loadcnt") {
    at = text.find(' ');
    if (at == std::string_view::npos) return -1;
    ++at;
  } else {
    return -1;
  }
  const std::string rest(text.substr(at));
  char* end = nullptr;
  const long parsed = std::strtol(rest.c_str(), &end, 0);
  if (end == rest.c_str()) return -1;
  return static_cast<int>(parsed);
}

enum class Space { kNone, kGlobal, kShared, kPrivate, kScalar };

struct Access {
  Space space = Space::kNone;
  bool reads = false;
  bool writes = false;
};

Access classify_access(std::string_view m) {
  if (starts_with(m, "ds_")) {
    if (contains(m, "permute") || contains(m, "swizzle")) return {};
    if (starts_with(m, "ds_read") || starts_with(m, "ds_load")) {
      return {Space::kShared, true, false};
    }
    if (starts_with(m, "ds_write") || starts_with(m, "ds_store")) {
      return {Space::kShared, false, true};
    }
    // Everything else that addresses scratch is a read-modify-write; only the
    // returning forms hand the old value back.
    if (access_width(m) != kWidthUnknown) {
      return {Space::kShared, contains(m, "_rtn"), true};
    }
    return {};
  }
  if (starts_with(m, "scratch_load")) return {Space::kPrivate, true, false};
  if (starts_with(m, "scratch_store")) return {Space::kPrivate, false, true};
  if (starts_with(m, "s_load") || starts_with(m, "s_buffer_load")) {
    return {Space::kScalar, true, false};
  }
  const bool global = starts_with(m, "global_") || starts_with(m, "flat_") ||
                      starts_with(m, "buffer_") || starts_with(m, "tbuffer_");
  if (!global) return {};
  if (contains(m, "atomic")) return {Space::kGlobal, contains(m, "_rtn"), true};
  if (contains(m, "_load")) return {Space::kGlobal, true, false};
  if (contains(m, "_store")) return {Space::kGlobal, false, true};
  return {};  // cache maintenance: buffer_gl0_inv and its family move nothing
}

bool is_cache_maintenance(std::string_view m) {
  return contains(m, "_inv") || contains(m, "_wb") || contains(m, "gl0") ||
         contains(m, "gl1");
}

// Products one dot instruction contracts, from the width in its name:
// v_dot4_i32_iu8 contracts four. Zero when the name states no width.
std::uint32_t dot_terms(std::string_view m) {
  const std::size_t at = m.find("_dot");
  if (at == std::string_view::npos) return 0;
  std::uint32_t n = 0;
  for (std::size_t i = at + 4; i < m.size() && m[i] >= '0' && m[i] <= '9';
       ++i) {
    n = n * 10 + static_cast<std::uint32_t>(m[i] - '0');
  }
  return n;
}

void classify_valu(std::string_view m, KernelCensus* out) {
  out->vector_alu.value += 1;
  if (contains(m, "wmma") || contains(m, "mfma")) {
    out->matrix_ops.value += 1;
    return;
  }
  if (contains(m, "_dot")) {
    out->dot_products.value += 1;
    out->multiply_accumulates.value += dot_terms(m);
    return;
  }
  if (contains(m, "fma") || contains(m, "_mad_")) {
    out->fused_multiply_adds.value += 1;
    // A packed form carries two of them in one issue.
    out->multiply_accumulates.value += starts_with(m, "v_pk_") ? 2u : 1u;
  }
}

void count_instruction(const Instruction& in, bool in_loop, KernelCensus* out) {
  const std::string_view text = in.text;
  const std::string_view mnemonic = first_word(text);
  if (mnemonic.empty()) return;

  const Access access = classify_access(mnemonic);
  if (access.space != Space::kNone) {
    const std::uint32_t width = access_width(mnemonic);
    if (width == kWidthUnknown) {
      out->unclassified.value += 1;
      return;
    }
    switch (access.space) {
      case Space::kGlobal:
        if (access.reads) out->global_loads.add(width, in_loop);
        if (access.writes) out->global_stores.add(width, in_loop);
        // A load straight into workgroup scratch is both halves at once.
        if (contains(mnemonic, "_lds")) out->shared_stores.add(width, in_loop);
        break;
      case Space::kShared:
        if (access.reads) out->shared_loads.add(width, in_loop);
        if (access.writes) out->shared_stores.add(width, in_loop);
        break;
      case Space::kPrivate:
        if (access.reads) out->private_loads.add(width, in_loop);
        if (access.writes) out->private_stores.add(width, in_loop);
        break;
      case Space::kScalar:
        out->scalar_loads.add(width, in_loop);
        break;
      case Space::kNone:
        break;
    }
    return;
  }

  if (contains(mnemonic, "permute") || contains(mnemonic, "swizzle") ||
      contains(mnemonic, "permlane")) {
    out->lane_exchanges.value += 1;
    // The vector-unit forms still spend an issue slot; the scratch-unit ones
    // (ds_permute and its family) do not, which is why the two counts are not
    // one count.
    if (starts_with(mnemonic, "v_")) out->vector_alu.value += 1;
    return;
  }
  if (starts_with(mnemonic, "v_")) {
    // A dual-issue instruction carries two operations in one encoding, printed
    // as "op_a :: op_b". Counting it once would halve the arithmetic term.
    std::size_t at = 0;
    while (at <= text.size()) {
      const std::size_t sep = text.find(" :: ", at);
      classify_valu(first_word(text.substr(at, sep - at)), out);
      if (sep == std::string_view::npos) break;
      at = sep + 4;
    }
    return;
  }
  if (starts_with(mnemonic, "s_branch") || starts_with(mnemonic, "s_cbranch")) {
    out->branches.value += 1;
    return;
  }
  if (starts_with(mnemonic, "s_")) {
    if (starts_with(mnemonic, "s_wait") || starts_with(mnemonic, "s_nop") ||
        starts_with(mnemonic, "s_delay") ||
        starts_with(mnemonic, "s_barrier") ||
        starts_with(mnemonic, "s_endpgm") || starts_with(mnemonic, "s_sleep") ||
        starts_with(mnemonic, "s_setprio") ||
        starts_with(mnemonic, "s_sendmsg") || starts_with(mnemonic, "s_code")) {
      return;
    }
    out->scalar_alu.value += 1;
    return;
  }
  if (is_cache_maintenance(mnemonic)) return;
  out->unclassified.value += 1;
}

KernelCensus count_body(const std::vector<Instruction>& body,
                        std::string entry) {
  KernelCensus out;
  out.entry = std::move(entry);
  out.instructions = DeviceFact<std::uint32_t>::queried(
      static_cast<std::uint32_t>(body.size()));
  for (auto* f : {&out.vector_alu, &out.scalar_alu, &out.dot_products,
                  &out.fused_multiply_adds, &out.matrix_ops,
                  &out.lane_exchanges, &out.multiply_accumulates,
                  &out.branches, &out.backward_branches, &out.memory_waits,
                  &out.deepest_load_batch, &out.serializing_waits,
                  &out.unclassified}) {
    *f = DeviceFact<std::uint32_t>::queried(0);
  }

  // Every backward branch closes a region that repeats, so an instruction
  // inside one executes an unknown number of times.
  std::vector<std::pair<std::uint64_t, std::uint64_t>> loops;
  for (const Instruction& in : body) {
    const std::string_view mnemonic = first_word(in.text);
    if (!starts_with(mnemonic, "s_branch") &&
        !starts_with(mnemonic, "s_cbranch")) {
      continue;
    }
    std::uint64_t target = 0;
    if (!branch_target(in.text, in.address, &target)) continue;
    if (target <= in.address) {
      out.backward_branches.value += 1;
      loops.emplace_back(target, in.address);
    }
  }

  std::uint32_t outstanding = 0;
  for (const Instruction& in : body) {
    const bool in_loop =
        std::any_of(loops.begin(), loops.end(), [&](const auto& l) {
          return in.address >= l.first && in.address <= l.second;
        });
    count_instruction(in, in_loop, &out);

    const Access access = classify_access(first_word(in.text));
    if (access.reads &&
        (access.space == Space::kGlobal || access.space == Space::kPrivate)) {
      ++outstanding;
      continue;
    }
    const int remaining = vector_memory_wait(in.text);
    if (remaining < 0) continue;
    const auto keep = static_cast<std::uint32_t>(remaining);
    if (keep >= outstanding) continue;
    const std::uint32_t retired = outstanding - keep;
    outstanding = keep;
    out.memory_waits.value += 1;
    out.deepest_load_batch.value =
        std::max(out.deepest_load_batch.value, retired);
    if (retired == 1) out.serializing_waits.value += 1;
  }
  // A matrix instruction's contraction belongs to the wave rather than to the
  // lane, so a per-lane mac count that included it would be wrong by the
  // wavefront size. Saying nothing is the only honest answer here.
  if (out.matrix_ops.value != 0) out.multiply_accumulates = {};
  return out;
}

struct Segment {
  std::uint64_t address = 0;
  std::uint64_t size = 0;
  std::size_t offset = 0;
};

struct CodeRange {
  std::string name;
  std::uint64_t address = 0;
  std::uint64_t size = 0;
};

struct Image {
  std::span<const std::byte> bytes;
  std::vector<Segment> segments;
  std::vector<CodeRange> kernels;
};

template <typename T>
bool read_at(std::span<const std::byte> b, std::uint64_t offset, T* out) {
  if (offset + sizeof(T) > b.size()) return false;
  std::memcpy(out, b.data() + offset, sizeof(T));
  return true;
}

std::string name_at(std::span<const std::byte> bytes, std::uint64_t offset) {
  if (offset >= bytes.size()) return {};
  const auto* text = reinterpret_cast<const char*>(bytes.data() + offset);
  std::size_t n = 0;
  while (offset + n < bytes.size() && text[n] != '\0') ++n;
  return std::string(text, n);
}

// A minimal ELF64 walk: the allocatable sections give an address-to-offset map,
// and the function symbols give one code range per kernel the object defines.
bool map_image(std::span<const std::byte> bytes, Image* out) {
  Elf64_Ehdr eh{};
  if (!read_at(bytes, 0, &eh)) return false;
  if (std::memcmp(eh.e_ident, ELFMAG, SELFMAG) != 0) return false;
  if (eh.e_ident[EI_CLASS] != ELFCLASS64) return false;
  if (eh.e_shentsize != sizeof(Elf64_Shdr) || eh.e_shnum == 0) return false;

  std::vector<Elf64_Shdr> sections(eh.e_shnum);
  for (std::size_t i = 0; i < sections.size(); ++i) {
    if (!read_at(bytes, eh.e_shoff + i * sizeof(Elf64_Shdr), &sections[i])) {
      return false;
    }
  }

  std::vector<bool> executable(sections.size(), false);
  for (std::size_t i = 0; i < sections.size(); ++i) {
    const Elf64_Shdr& sh = sections[i];
    if ((sh.sh_flags & SHF_ALLOC) == 0 || sh.sh_type == SHT_NOBITS) continue;
    if (sh.sh_offset + sh.sh_size > bytes.size()) continue;
    out->segments.push_back(
        Segment{sh.sh_addr, sh.sh_size, static_cast<std::size_t>(sh.sh_offset)});
    executable[i] = (sh.sh_flags & SHF_EXECINSTR) != 0;
  }

  // .symtab first because it names every kernel; .dynsym is the fallback for an
  // object that keeps only what the loader needs.
  for (const auto want : {std::uint32_t{SHT_SYMTAB}, std::uint32_t{SHT_DYNSYM}}) {
    for (const Elf64_Shdr& sh : sections) {
      if (sh.sh_type != want || sh.sh_entsize != sizeof(Elf64_Sym)) continue;
      if (sh.sh_link >= sections.size()) continue;
      const Elf64_Shdr& strtab = sections[sh.sh_link];
      const std::size_t count = sh.sh_size / sizeof(Elf64_Sym);
      for (std::size_t i = 0; i < count; ++i) {
        Elf64_Sym sym{};
        if (!read_at(bytes, sh.sh_offset + i * sizeof(Elf64_Sym), &sym)) break;
        if (ELF64_ST_TYPE(sym.st_info) != STT_FUNC || sym.st_size == 0) continue;
        if (sym.st_shndx >= executable.size() || !executable[sym.st_shndx]) {
          continue;
        }
        std::string name = name_at(bytes, strtab.sh_offset + sym.st_name);
        if (name.empty()) continue;
        out->kernels.push_back(
            CodeRange{std::move(name), sym.st_value, sym.st_size});
      }
      if (!out->kernels.empty()) break;
    }
    if (!out->kernels.empty()) break;
  }
  out->bytes = bytes;
  return !out->kernels.empty() && !out->segments.empty();
}

struct Walk {
  const Image* image = nullptr;
  std::vector<Instruction>* body = nullptr;
  std::uint64_t address = 0;
};

std::uint64_t read_memory(std::uint64_t from, char* to, std::uint64_t size,
                          void* user) {
  const auto* w = static_cast<const Walk*>(user);
  for (const Segment& s : w->image->segments) {
    if (from < s.address || from >= s.address + s.size) continue;
    const std::uint64_t avail = std::min(size, s.address + s.size - from);
    std::memcpy(to, w->image->bytes.data() + s.offset + (from - s.address),
                avail);
    return avail;
  }
  return 0;
}

void print_instruction(const char* text, void* user) {
  auto* w = static_cast<Walk*>(user);
  std::string_view s(text == nullptr ? "" : text);
  while (!s.empty() && (s.front() == ' ' || s.front() == '\t')) {
    s.remove_prefix(1);
  }
  w->body->push_back(Instruction{w->address, std::string(s)});
}

void print_annotation(std::uint64_t, void*) {}

}  // namespace

std::vector<KernelCensus> read_code_object_census(
    std::span<const std::byte> object) {
  std::vector<KernelCensus> out;
  if (object.empty()) return out;
  std::string target = read_code_object_target(object);
  if (target.empty()) return out;

  // Newer comgr writes the environment slot of the stored target as "unknown"
  // ("amdgcn-amd-amdhsa-unknown-gfx1151"), and create_disassembly_info
  // rejects that triple while accepting the empty-environment form older ROCm
  // emitted ("amdgcn-amd-amdhsa--gfx1151"). The disassembler only needs the
  // arch and its feature suffix, so fold the environment token away. The
  // search resumes one character past the match so a second, overlapping
  // placeholder is folded too rather than skipped by a fixed step.
  for (std::string::size_type p = target.find("-unknown-");
       p != std::string::npos; p = target.find("-unknown-", p + 1)) {
    target.replace(p, std::strlen("-unknown-"), "--");
  }

  Image image;
  if (!map_image(object, &image)) return out;

  Walk walk;
  walk.image = &image;
  amd_comgr_disassembly_info_t info{};
  if (amd_comgr_create_disassembly_info(target.c_str(), &read_memory,
                                        &print_instruction, &print_annotation,
                                        &info) != AMD_COMGR_STATUS_SUCCESS) {
    return out;
  }

  out.reserve(image.kernels.size());
  for (const CodeRange& k : image.kernels) {
    std::vector<Instruction> body;
    walk.body = &body;
    std::uint64_t at = k.address;
    const std::uint64_t end = k.address + k.size;
    while (at < end) {
      walk.address = at;
      std::uint64_t size = 0;
      const std::size_t before = body.size();
      if (amd_comgr_disassemble_instruction(info, at, &walk, &size) !=
              AMD_COMGR_STATUS_SUCCESS ||
          size == 0) {
        break;
      }
      // A decode that produced no text still consumed bytes: that is an
      // instruction this reader cannot name, not an instruction that is absent.
      if (body.size() == before) body.push_back(Instruction{at, "?"});
      at += size;
    }
    if (body.empty()) continue;
    out.push_back(count_body(body, k.name));
  }

  amd_comgr_destroy_disassembly_info(info);
  return out;
}

}  // namespace lse::backend

#else  // !LSE_HAVE_COMGR

namespace lse::backend {

std::vector<KernelCensus> read_code_object_census(std::span<const std::byte>) {
  return {};
}

}  // namespace lse::backend

#endif  // LSE_HAVE_COMGR
