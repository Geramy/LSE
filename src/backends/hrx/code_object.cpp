#include "lse/backends/hrx/code_object.hpp"

#include "lse/backends/hrx/arch_database.hpp"

#include <array>
#include <cstddef>
#include <cstdlib>
#include <string>
#include <vector>

#if LSE_HAVE_COMGR
#include <amd_comgr/amd_comgr.h>
#endif

namespace lse::backend {

#if LSE_HAVE_COMGR

namespace {

// Destroys a metadata node on the way out. comgr hands out owning handles from
// every lookup, including the ones that fail a later parse.
struct NodeGuard {
  amd_comgr_metadata_node_t handle{};
  bool live = false;
  NodeGuard() = default;
  NodeGuard(const NodeGuard&) = delete;
  NodeGuard& operator=(const NodeGuard&) = delete;
  ~NodeGuard() {
    if (live) amd_comgr_destroy_metadata(handle);
  }
};

// Every scalar in both of comgr's metadata trees is a STRING node — including
// the register counts — so reading one is always a size query, a fetch and a
// parse. The size comgr reports includes the terminating NUL.
bool read_string(amd_comgr_metadata_node_t node, std::string* out) {
  amd_comgr_metadata_kind_t kind{};
  if (amd_comgr_get_metadata_kind(node, &kind) != AMD_COMGR_STATUS_SUCCESS ||
      kind != AMD_COMGR_METADATA_KIND_STRING) {
    return false;
  }
  std::size_t size = 0;
  if (amd_comgr_get_metadata_string(node, &size, nullptr) !=
          AMD_COMGR_STATUS_SUCCESS ||
      size == 0) {
    return false;
  }
  std::string text(size, '\0');
  if (amd_comgr_get_metadata_string(node, &size, text.data()) !=
      AMD_COMGR_STATUS_SUCCESS) {
    return false;
  }
  if (!text.empty() && text.back() == '\0') text.pop_back();
  *out = std::move(text);
  return true;
}

// A key that is absent returns an error status from the lookup rather than an
// empty node, and that is the normal path — .agpr_count on RDNA, the spill
// counts on a loomc object. It must read as unknown and never as zero.
DeviceFact<std::uint32_t> read_u32(amd_comgr_metadata_node_t map,
                                   const char* key) {
  NodeGuard v;
  if (amd_comgr_metadata_lookup(map, key, &v.handle) !=
      AMD_COMGR_STATUS_SUCCESS) {
    return {};
  }
  v.live = true;
  std::string text;
  if (!read_string(v.handle, &text)) return {};
  char* end = nullptr;
  const unsigned long long parsed = std::strtoull(text.c_str(), &end, 10);
  if (end == text.c_str()) return {};
  return DeviceFact<std::uint32_t>::queried(
      static_cast<std::uint32_t>(parsed));
}

std::string read_text(amd_comgr_metadata_node_t map, const char* key) {
  NodeGuard v;
  if (amd_comgr_metadata_lookup(map, key, &v.handle) !=
      AMD_COMGR_STATUS_SUCCESS) {
    return {};
  }
  v.live = true;
  std::string text;
  if (!read_string(v.handle, &text)) return {};
  return text;
}

DeviceFact<std::array<std::uint32_t, 3>> read_dim3(
    amd_comgr_metadata_node_t map, const char* key) {
  NodeGuard list;
  if (amd_comgr_metadata_lookup(map, key, &list.handle) !=
      AMD_COMGR_STATUS_SUCCESS) {
    return {};
  }
  list.live = true;
  std::size_t n = 0;
  if (amd_comgr_get_metadata_list_size(list.handle, &n) !=
          AMD_COMGR_STATUS_SUCCESS ||
      n != 3) {
    return {};
  }
  std::array<std::uint32_t, 3> dims{};
  for (std::size_t i = 0; i < 3; ++i) {
    NodeGuard item;
    if (amd_comgr_index_list_metadata(list.handle, i, &item.handle) !=
        AMD_COMGR_STATUS_SUCCESS) {
      return {};
    }
    item.live = true;
    std::string text;
    if (!read_string(item.handle, &text)) return {};
    dims[i] = static_cast<std::uint32_t>(std::strtoull(text.c_str(), nullptr, 10));
  }
  return DeviceFact<std::array<std::uint32_t, 3>>::queried(dims);
}

}  // namespace

std::vector<KernelResources> read_code_object_resources(
    std::span<const std::byte> object) {
  std::vector<KernelResources> out;
  if (object.empty()) return out;

  amd_comgr_data_t data{};
  if (amd_comgr_create_data(AMD_COMGR_DATA_KIND_EXECUTABLE, &data) !=
      AMD_COMGR_STATUS_SUCCESS) {
    return out;
  }
  // set_data_name is not required for a bare code object, and both of this
  // backend's compilers produce one. A clang offload bundle would need
  // unbundling first, which is why nothing here accepts one.
  if (amd_comgr_set_data(data, object.size(),
                         reinterpret_cast<const char*>(object.data())) !=
      AMD_COMGR_STATUS_SUCCESS) {
    amd_comgr_release_data(data);
    return out;
  }

  NodeGuard root;
  if (amd_comgr_get_data_metadata(data, &root.handle) !=
      AMD_COMGR_STATUS_SUCCESS) {
    amd_comgr_release_data(data);
    return out;
  }
  root.live = true;

  NodeGuard kernels;
  if (amd_comgr_metadata_lookup(root.handle, "amdhsa.kernels",
                                &kernels.handle) != AMD_COMGR_STATUS_SUCCESS) {
    amd_comgr_release_data(data);
    return out;
  }
  kernels.live = true;

  std::size_t count = 0;
  if (amd_comgr_get_metadata_list_size(kernels.handle, &count) !=
      AMD_COMGR_STATUS_SUCCESS) {
    amd_comgr_release_data(data);
    return out;
  }

  out.reserve(count);
  for (std::size_t i = 0; i < count; ++i) {
    NodeGuard k;
    if (amd_comgr_index_list_metadata(kernels.handle, i, &k.handle) !=
        AMD_COMGR_STATUS_SUCCESS) {
      continue;
    }
    k.live = true;

    KernelResources r;
    // .name, not .symbol: the note carries both, and .symbol is the loader
    // symbol with its ".kd" descriptor suffix. .name is the entry the rest of
    // the engine names a kernel by, so matching on it is what lets a caller
    // look resources up with the same string it launched with.
    r.entry = read_text(k.handle, ".name");
    if (r.entry.empty()) r.entry = read_text(k.handle, ".symbol");
    r.vector_registers = read_u32(k.handle, ".vgpr_count");
    r.scalar_registers = read_u32(k.handle, ".sgpr_count");
    r.accum_registers = read_u32(k.handle, ".agpr_count");
    r.workgroup_segment_bytes =
        read_u32(k.handle, ".group_segment_fixed_size");
    r.private_segment_bytes =
        read_u32(k.handle, ".private_segment_fixed_size");
    r.vector_spills = read_u32(k.handle, ".vgpr_spill_count");
    r.scalar_spills = read_u32(k.handle, ".sgpr_spill_count");
    r.kernarg_segment_bytes = read_u32(k.handle, ".kernarg_segment_size");
    r.max_flat_workgroup_size = read_u32(k.handle, ".max_flat_workgroup_size");
    r.wavefront_size = read_u32(k.handle, ".wavefront_size");
    r.required_workgroup_size = read_dim3(k.handle, ".reqd_workgroup_size");
    out.push_back(std::move(r));
  }

  amd_comgr_release_data(data);
  return out;
}

std::string read_code_object_target(std::span<const std::byte> object) {
  if (object.empty()) return {};
  amd_comgr_data_t data{};
  if (amd_comgr_create_data(AMD_COMGR_DATA_KIND_EXECUTABLE, &data) !=
      AMD_COMGR_STATUS_SUCCESS) {
    return {};
  }
  std::string target;
  if (amd_comgr_set_data(data, object.size(),
                         reinterpret_cast<const char*>(object.data())) ==
      AMD_COMGR_STATUS_SUCCESS) {
    NodeGuard root;
    if (amd_comgr_get_data_metadata(data, &root.handle) ==
        AMD_COMGR_STATUS_SUCCESS) {
      root.live = true;
      target = read_text(root.handle, "amdhsa.target");
    }
  }
  amd_comgr_release_data(data);
  return target;
}

ArchFacts query_isa_facts(std::string_view arch) {
  ArchFacts facts;
  if (arch.empty()) return facts;
  const std::string target = "amdgcn-amd-amdhsa--" + std::string(arch);

  NodeGuard isa;
  if (amd_comgr_get_isa_metadata(target.c_str(), &isa.handle) !=
      AMD_COMGR_STATUS_SUCCESS) {
    return facts;
  }
  isa.live = true;

  facts.vector_registers_per_simd = read_u32(isa.handle, "TotalNumVGPRs");
  facts.vector_register_alloc_granule = read_u32(isa.handle, "VGPRAllocGranule");
  facts.vector_registers_addressable_per_wave =
      read_u32(isa.handle, "AddressableNumVGPRs");
  facts.scalar_registers_per_simd = read_u32(isa.handle, "TotalNumSGPRs");
  facts.scalar_register_alloc_granule = read_u32(isa.handle, "SGPRAllocGranule");
  facts.scalar_registers_addressable_per_wave =
      read_u32(isa.handle, "AddressableNumSGPRs");
  facts.lds_bytes_addressable_per_workgroup =
      read_u32(isa.handle, "LocalMemorySize");
  facts.lds_banks = read_u32(isa.handle, "LDSBankCount");
  facts.max_flat_workgroup_size = read_u32(isa.handle, "MaxFlatWorkGroupSize");
  return facts;
}

#else  // !LSE_HAVE_COMGR

// The comgr-free path. A build without the comgr library (this macOS host
// adapter is one: the device compiler is loomc, and comgr is a Linux/ROCm
// dependency) still gets the measured resources, because the `amdhsa.kernels`
// note is a standard ELF note that both the comgr and loomc code objects carry.
// The note contains MessagePack. The portable reader scans token boundaries
// and decodes the unsigned resource fields emitted by loomc. Missing or
// malformed fields remain unknown, including absent allocator spill counts.

namespace {

// Reads a little-endian field of `bytes` from `off` of an ELF64 header/section
// table entry. The note section we walk is always a 64-bit object, so the
// access widths are fixed; a 32-bit object simply yields no `.note` the way we
// find it and degrades to "no resources" the same as before.
std::uint64_t le64(const std::byte* p) {
  std::uint64_t v = 0;
  for (int i = 7; i >= 0; --i) v = (v << 8) | static_cast<std::uint8_t>(p[i]);
  return v;
}
std::uint32_t le32(const std::byte* p) {
  std::uint32_t v = 0;
  for (int i = 3; i >= 0; --i) v = (v << 8) | static_cast<std::uint8_t>(p[i]);
  return v;
}
std::uint16_t le16(const std::byte* p) {
  return static_cast<std::uint16_t>(
      static_cast<std::uint8_t>(p[0]) | (static_cast<std::uint8_t>(p[1]) << 8));
}

// The byte sequence a key is stored as: its `0xa0 | len` tag then its bytes.
std::string key_bytes(const std::string& key) {
  if (key.size() > 31) return {};  // long keys do not occur for our fields
  std::string b;
  b.push_back(static_cast<char>(0xa0 | key.size()));
  b += key;
  return b;
}

// Advance past one token header/payload, visiting container children in order.
// Integer payload bytes can look like string tags; only token starts are keys.
bool next_token(std::string_view note, std::size_t pos, std::size_t* next) {
  if (pos >= note.size()) return false;
  const auto tag = static_cast<unsigned char>(note[pos]);
  std::size_t header = 1, payload = 0, length_bytes = 0;
  bool sized_payload = false;
  if (tag >= 0xa0 && tag <= 0xbf) {
    payload = tag & 0x1fu;
  } else if (tag < 0xa0 || tag >= 0xe0 || tag == 0xc0 ||
             tag == 0xc2 || tag == 0xc3) {
    // Fixints, fixed container headers, nil and booleans.
  } else if ((tag >= 0xcc && tag <= 0xcf) ||
             (tag >= 0xd0 && tag <= 0xd3)) {
    payload = std::size_t{1} << (tag & 3u);
  } else if (tag == 0xca || tag == 0xcb) {
    payload = tag == 0xca ? 4 : 8;
  } else if (tag >= 0xd4 && tag <= 0xd8) {
    payload = 1 + (std::size_t{1} << (tag - 0xd4));
  } else if ((tag >= 0xc4 && tag <= 0xc6) ||
             (tag >= 0xc7 && tag <= 0xc9) ||
             (tag >= 0xd9 && tag <= 0xdb)) {
    const auto first = tag <= 0xc6 ? 0xc4 : tag <= 0xc9 ? 0xc7 : 0xd9;
    length_bytes = std::size_t{1} << (tag - first);
    sized_payload = true;
    if (tag >= 0xc7 && tag <= 0xc9) ++header;  // extension type
  } else if (tag >= 0xdc && tag <= 0xdf) {
    length_bytes = (tag & 1u) == 0 ? 2 : 4;
  } else {
    return false;
  }
  if (length_bytes > note.size() - pos - 1) return false;
  if (sized_payload) {
    for (std::size_t i = 1; i <= length_bytes; ++i)
      payload = (payload << 8) | static_cast<unsigned char>(note[pos + i]);
  }
  header += length_bytes;
  if (header > note.size() - pos || payload > note.size() - pos - header)
    return false;
  *next = pos + header + payload;
  return true;
}

std::size_t find_key(std::string_view note, const std::string& key,
                     bool last = false) {
  const std::string needle = key_bytes(key);
  if (needle.empty()) return std::string_view::npos;
  std::size_t found = std::string_view::npos;
  for (std::size_t pos = 0; pos < note.size();) {
    if (note.substr(pos, needle.size()) == needle) {
      found = pos + needle.size();
      if (!last) return found;
    }
    if (!next_token(note, pos, &pos)) break;
  }
  return found;
}

// Argument names precede the kernel name in the current single-entry objects.
std::size_t find_last_key(std::string_view note, const std::string& key) {
  return find_key(note, key, true);
}

// Positive fixint and MessagePack uint8/16/32, as emitted by loomc.
bool read_int(std::string_view note, std::size_t pos, std::uint32_t* out,
              std::size_t* next = nullptr) {
  if (pos >= note.size()) return false;
  const auto tag = static_cast<unsigned char>(note[pos]);
  std::size_t payload = 0;
  if (tag == 0xcc) payload = 1;
  else if (tag == 0xcd) payload = 2;
  else if (tag == 0xce) payload = 4;
  else if (tag >= 0x80) return false;
  if (payload > note.size() - pos - 1) return false;
  std::uint32_t value = payload == 0 ? tag : 0;
  for (std::size_t i = 1; i <= payload; ++i) {
    value = (value << 8) | static_cast<unsigned char>(note[pos + i]);
  }
  *out = value;
  if (next != nullptr) *next = pos + payload + 1;
  return true;
}

// Reads the three unsigned elements of `.reqd_workgroup_size`.
bool read_reqd(std::string_view note, std::size_t pos, std::array<std::uint32_t, 3>* out) {
  if (pos >= note.size() || static_cast<unsigned char>(note[pos]) != 0x93)
    return false;
  std::size_t q = pos + 1;
  std::size_t got = 0;
  while (got < 3 && q < note.size()) {
    std::uint32_t v = 0;
    if (!read_int(note, q, &v, &q)) break;
    (*out)[got++] = v;
  }
  return got == 3;
}

// Walks the `.note` section of a 64-bit ELF and returns the description bytes
// of the note whose owner is `owner` (NUL-terminated). An ELF note is a
// 12-byte header (namesz, descsz, type, each 4 bytes LE) followed by the
// owner name padded to a 4-byte multiple, then the description padded the
// same way. Notes are laid out back to back in the section.
std::vector<std::byte> find_note(const std::byte* obj, std::size_t size,
                                 std::string_view owner) {
  if (size < 64 || obj[0] != static_cast<std::byte>(0x7f) ||
      obj[1] != static_cast<std::byte>('E') || obj[2] != static_cast<std::byte>('L') ||
      obj[3] != static_cast<std::byte>('F'))
    return {};
  if (obj[4] != static_cast<std::byte>(2))
    return {};  // ELFCLASS64 only; this tree's objects are 64-bit.
  const auto shoff = static_cast<std::size_t>(le64(obj + 0x28));
  const auto shentsize = le16(obj + 0x3A);
  const auto shnum = le16(obj + 0x3C);
  const auto shstrndx = le16(obj + 0x3E);
  if (shoff == 0 || shentsize < 64 || shnum == 0 || shnum > 4096) return {};
  if (shoff + static_cast<std::size_t>(shnum) * shentsize > size) return {};
  const std::byte* sh = obj + shoff;
  const auto section_names_off = static_cast<std::size_t>(le64(sh + shstrndx * shentsize + 0x18));
  const auto section_names_size = static_cast<std::size_t>(le64(sh + shstrndx * shentsize + 0x20));
  if (section_names_off + section_names_size > size) return {};
  for (std::size_t i = 0; i < shnum; ++i) {
    const std::byte* e = sh + i * shentsize;
    const auto sh_name_off = le32(e + 0x00);
    if (sh_name_off >= section_names_size) continue;
    const char* name = reinterpret_cast<const char*>(obj + section_names_off + sh_name_off);
    if (std::string(name) != ".note") continue;
    const auto note_off = static_cast<std::size_t>(le64(e + 0x18));
    const auto note_size = static_cast<std::size_t>(le64(e + 0x20));
    if (note_off + note_size > size) continue;
    std::size_t p = note_off;
    while (p + 12 <= note_off + note_size) {
      const auto namesz = le32(obj + p);
      const auto descsz = le32(obj + p + 4);
      // n_type at p+8; we match on owner text, not type, so it is read but unused.
      p += 12;
      const auto padded_name = (namesz + 3) & ~static_cast<std::uint32_t>(3);
      if (p + padded_name + descsz > note_off + note_size) break;
      if (namesz > 0 && namesz < 64) {
        std::string_view got(reinterpret_cast<const char*>(obj + p), namesz);
        // namesz counts the NUL terminator (the ELF note stores "AMDGPU\0"),
        // so drop it before comparing to the un-terminated owner.
        while (!got.empty() && got.back() == '\0') got.remove_suffix(1);
        if (got == owner) {
          return {obj + p + padded_name, obj + p + padded_name + descsz};
        }
      }
      p += padded_name + descsz;
    }
  }
  return {};
}

}  // namespace

std::vector<KernelResources> read_code_object_resources(
    std::span<const std::byte> object) {
  std::vector<KernelResources> out;
  if (object.empty()) return out;

  // The note is the standard `AMDGPU` owner / NT_AMDGPU_METADATA note whose
  // description is the LLVM metadata binary; `amdhsa.kernels` is a key inside
  // that document, not the ELF note owner. (The comgr path reads the same note
  // and walks into its `amdhsa.kernels` list; we decode the binary directly.)
  const std::vector<std::byte> note =
      find_note(object.data(), object.size(), "AMDGPU");
  if (note.empty()) return out;
  std::string_view doc(reinterpret_cast<const char*>(note.data()), note.size());

  // Locate the kernel's own `.name`. `.name` also appears once per argument
  // inside the `.args` list (the binding names and `count`), all of which come
  // before the kernel's own entry; the kernel's `.name` is the last `.name`
  // token in the entry. (The loader symbol `.symbol`, where present, is the
  // same string plus a ".kd" descriptor suffix.)
  const std::size_t entry_pos = find_last_key(doc, ".name");
  if (entry_pos == std::string_view::npos || entry_pos >= doc.size()) return out;
  // The `.name` value is a string token: `0xa0|len` tag then the bytes.
  const unsigned char tag = static_cast<unsigned char>(doc[entry_pos]);
  if (tag < 0xa0 || tag > 0xbf) return out;
  const std::size_t len = tag - 0xa0;
  if (len > doc.size() - entry_pos - 1) return out;
  const std::string_view entry(doc.data() + entry_pos + 1, len);

  KernelResources r;
  r.entry = std::string(entry);

  // Each numeric field: find the key, read the integer token that follows.
  // Key PRESENCE is the provenance: a field the note names (even as zero) is
  // a real answer; a field it does not name is UNKNOWN, not a guessed zero.
  auto num = [&](const char* key, DeviceFact<std::uint32_t>* slot) {
    const std::size_t p = find_key(doc, key);
    if (p == std::string_view::npos) return;
    std::uint32_t v = 0;
    if (read_int(doc, p, &v)) *slot = DeviceFact<std::uint32_t>::queried(v);
  };
  num(".vgpr_count", &r.vector_registers);
  num(".sgpr_count", &r.scalar_registers);
  num(".agpr_count", &r.accum_registers);
  num(".group_segment_fixed_size", &r.workgroup_segment_bytes);
  num(".private_segment_fixed_size", &r.private_segment_bytes);
  num(".vgpr_spill_count", &r.vector_spills);
  num(".sgpr_spill_count", &r.scalar_spills);
  num(".kernarg_segment_size", &r.kernarg_segment_bytes);
  num(".max_flat_workgroup_size", &r.max_flat_workgroup_size);
  num(".wavefront_size", &r.wavefront_size);
  const std::size_t reqd_pos = find_key(doc, ".reqd_workgroup_size");
  if (reqd_pos != std::string_view::npos) {
    std::array<std::uint32_t, 3> reqd{};
    if (read_reqd(doc, reqd_pos, &reqd) && reqd[0] != 0 &&
        reqd[1] != 0 && reqd[2] != 0)
      r.required_workgroup_size =
          DeviceFact<std::array<std::uint32_t, 3>>::queried(reqd);
  }
  out.push_back(std::move(r));
  return out;
}

std::string read_code_object_target(std::span<const std::byte> object) {
  if (object.empty()) return {};
  const std::vector<std::byte> note =
      find_note(object.data(), object.size(), "AMDGPU");
  if (note.empty()) return {};
  std::string_view doc(reinterpret_cast<const char*>(note.data()), note.size());
  // `amdhsa.target` is a top-level key (a `0xa0|len` string token).
  const std::size_t p = find_key(doc, "amdhsa.target");
  if (p == std::string_view::npos || p >= doc.size()) return {};
  const unsigned char tag = static_cast<unsigned char>(doc[p]);
  if (tag < 0xa0 || tag > 0xbf) return {};
  const std::size_t len = tag - 0xa0;
  if (len > doc.size() - p - 1) return {};
  return std::string(doc.data() + p + 1, len);
}

ArchFacts query_isa_facts(std::string_view) { return {}; }

#endif  // LSE_HAVE_COMGR

ArchFacts arch_facts_for(const DeviceInfo& info) {
  ArchFacts facts = query_isa_facts(info.arch);

  // The device runtime answered these before any compiler was consulted, and
  // a live query outranks a table row. It does NOT outrank the compiler's own
  // table for the same key: where the two differ the compiler's number is what
  // the code object was actually built against.
  if (!facts.lds_bytes_addressable_per_workgroup.known() &&
      info.lds_bytes_per_workgroup != 0) {
    facts.lds_bytes_addressable_per_workgroup =
        DeviceFact<std::uint32_t>::queried(info.lds_bytes_per_workgroup);
  }
  if (!facts.max_flat_workgroup_size.known() &&
      info.max_threads_per_workgroup != 0) {
    facts.max_flat_workgroup_size = DeviceFact<std::uint32_t>::queried(
        info.max_threads_per_workgroup);
  }

  // Last resort. The family rows carry no register facts at all, so those
  // stay unknown on a build with no comgr rather than acquiring a plausible
  // number — which is the whole point of the field being unknown-able.
  if (const FamilyIsa* isa = family_isa(arch_family(info.arch));
      isa != nullptr) {
    if (!facts.lds_bytes_addressable_per_workgroup.known()) {
      facts.lds_bytes_addressable_per_workgroup =
          DeviceFact<std::uint32_t>::declared(isa->lds_bytes_per_workgroup);
    }
    if (!facts.max_flat_workgroup_size.known()) {
      facts.max_flat_workgroup_size = DeviceFact<std::uint32_t>::declared(
          isa->max_threads_per_workgroup);
    }
  }
  // Residency capacity, from the same declaration apply_arch_defaults uses, so
  // a device built by hand and a device the runtime described answer alike.
  DeviceInfo residency = info;
  residency.arch_facts = ArchFacts{};
  apply_residency_facts(residency);
  facts.wave_slots_per_simd = residency.arch_facts.wave_slots_per_simd;
  facts.simds_per_lds_pool = residency.arch_facts.simds_per_lds_pool;
  facts.lds_bytes_per_pool = residency.arch_facts.lds_bytes_per_pool;
  facts.lds_alloc_granule_bytes =
      residency.arch_facts.lds_alloc_granule_bytes;
  return facts;
}

}  // namespace lse::backend
