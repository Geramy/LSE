#include "lse/backends/hrx/code_object.hpp"

#include "lse/backends/hrx/arch_database.hpp"

#include <cstdlib>
#include <string>

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

std::vector<KernelResources> read_code_object_resources(
    std::span<const std::byte>) {
  return {};
}

std::string read_code_object_target(std::span<const std::byte>) { return {}; }

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
