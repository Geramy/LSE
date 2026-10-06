// Tiny HSA helper: load a code object, allocate VRAM, copy, dispatch with timing.
#pragma once
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#define CK(x) do { hsa_status_t s_ = (x); if (s_ != HSA_STATUS_SUCCESS) { const char* m_ = nullptr; hsa_status_string(s_, &m_); \
  std::fprintf(stderr, "%s failed: %s (0x%x) %s:%d\n", #x, m_ ? m_ : "?", (unsigned)s_, __FILE__, __LINE__); std::exit(2);} } while (0)

struct Hk {
  hsa_agent_t gpu{}, cpu{};
  hsa_amd_memory_pool_t dev{}, host{};
  uint32_t cus = 0, mhz = 0, wave = 0;
  char name[64] = {};
  hsa_executable_t exe{};
  hsa_code_object_reader_t reader{};
  hsa_queue_t* q = nullptr;
  hsa_signal_t sig{};
  void* karg = nullptr;
  long spurious = 0;

  static hsa_status_t pick_dev(hsa_amd_memory_pool_t p, void* d) {
    hsa_amd_segment_t seg; hsa_amd_memory_pool_get_info(p, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &seg);
    auto* self = (Hk*)d; if (seg == HSA_AMD_SEGMENT_GLOBAL && !self->dev.handle) self->dev = p; return HSA_STATUS_SUCCESS;
  }
  static hsa_status_t pick_host(hsa_amd_memory_pool_t p, void* d) {
    hsa_amd_segment_t seg; hsa_amd_memory_pool_get_info(p, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &seg);
    uint32_t fl = 0; hsa_amd_memory_pool_get_info(p, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &fl);
    auto* self = (Hk*)d;
    if (seg == HSA_AMD_SEGMENT_GLOBAL && (fl & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT) && !self->host.handle) self->host = p;
    return HSA_STATUS_SUCCESS;
  }
  void init(const char* path) {
    CK(hsa_init());
    hsa_iterate_agents([](hsa_agent_t a, void* d) -> hsa_status_t {
      hsa_device_type_t t; hsa_agent_get_info(a, HSA_AGENT_INFO_DEVICE, &t); auto* self = (Hk*)d;
      if (t == HSA_DEVICE_TYPE_GPU && !self->gpu.handle) self->gpu = a;
      if (t == HSA_DEVICE_TYPE_CPU && !self->cpu.handle) self->cpu = a;
      return HSA_STATUS_SUCCESS; }, this);
    if (!gpu.handle) { std::fprintf(stderr, "no GPU agent\n"); std::exit(2); }
    CK(hsa_agent_get_info(gpu, HSA_AGENT_INFO_NAME, name));
    CK(hsa_agent_get_info(gpu, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_COMPUTE_UNIT_COUNT, &cus));
    CK(hsa_agent_get_info(gpu, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_MAX_CLOCK_FREQUENCY, &mhz));
    CK(hsa_agent_get_info(gpu, HSA_AGENT_INFO_WAVEFRONT_SIZE, &wave));
    hsa_amd_agent_iterate_memory_pools(gpu, pick_dev, this);
    if (cpu.handle) hsa_amd_agent_iterate_memory_pools(cpu, pick_host, this);
    if (!host.handle) hsa_amd_agent_iterate_memory_pools(gpu, pick_host, this);
    if (!dev.handle || !host.handle) { std::fprintf(stderr, "memory pools missing\n"); std::exit(2); }
    std::ifstream f(path, std::ios::binary); std::vector<char> obj((std::istreambuf_iterator<char>(f)), {});
    if (obj.empty()) { std::fprintf(stderr, "cannot read %s\n", path); std::exit(2); }
    CK(hsa_code_object_reader_create_from_memory(obj.data(), obj.size(), &reader));
    CK(hsa_executable_create_alt(HSA_PROFILE_BASE, HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT, nullptr, &exe));
    CK(hsa_executable_load_agent_code_object(exe, gpu, reader, nullptr, nullptr));
    CK(hsa_executable_freeze(exe, nullptr));
    CK(hsa_queue_create(gpu, 64, HSA_QUEUE_TYPE_SINGLE, nullptr, nullptr, UINT32_MAX, UINT32_MAX, &q));
    CK(hsa_signal_create(1, 0, nullptr, &sig));
    CK(hsa_amd_memory_pool_allocate(host, 4096, 0, &karg));
    hsa_amd_agents_allow_access(1, &gpu, nullptr, karg);
  }
  void* alloc(size_t bytes) { void* p = nullptr; CK(hsa_amd_memory_pool_allocate(dev, bytes, 0, &p)); return p; }
  void* alloc_host(size_t bytes) { void* p = nullptr; CK(hsa_amd_memory_pool_allocate(host, bytes, 0, &p)); hsa_amd_agents_allow_access(1, &gpu, nullptr, p); return p; }
  void upload(void* dst, const void* src, size_t n) {
    void* st = alloc_host(n); std::memcpy(st, src, n); CK(hsa_memory_copy(dst, st, n)); hsa_amd_memory_pool_free(st);
  }
  void download(void* dst, const void* src, size_t n) {
    void* st = alloc_host(n); CK(hsa_memory_copy(st, src, n)); std::memcpy(dst, st, n); hsa_amd_memory_pool_free(st);
  }
  struct Kernel { uint64_t obj; uint32_t kargsz, grp, priv; };
  Kernel kernel(const char* kname) {
    std::string sym = std::string(kname) + ".kd"; hsa_executable_symbol_t ks;
    CK(hsa_executable_get_symbol_by_name(exe, sym.c_str(), &gpu, &ks));
    Kernel k{};
    CK(hsa_executable_symbol_get_info(ks, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &k.obj));
    CK(hsa_executable_symbol_get_info(ks, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE, &k.kargsz));
    CK(hsa_executable_symbol_get_info(ks, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE, &k.grp));
    CK(hsa_executable_symbol_get_info(ks, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE, &k.priv));
    return k;
  }
  // args: raw kernarg bytes (explicit args); returns host wall ms.
  double run(const Kernel& k, const void* args, size_t argbytes, uint32_t gx, uint32_t gy, uint32_t wgx) {
    std::memset(karg, 0, k.kargsz > 4096 ? 4096 : k.kargsz); std::memcpy(karg, args, argbytes);
    hsa_signal_store_screlease(sig, 1);
    uint64_t idx = hsa_queue_add_write_index_relaxed(q, 1);
    auto* pk = (hsa_kernel_dispatch_packet_t*)q->base_address + (idx & (q->size - 1));
    std::memset((char*)pk + 4, 0, sizeof(*pk) - 4);
    uint16_t setup = 2 << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS;
    pk->workgroup_size_x = (uint16_t)wgx; pk->workgroup_size_y = 1; pk->workgroup_size_z = 1;
    pk->grid_size_x = gx * wgx; pk->grid_size_y = gy; pk->grid_size_z = 1;
    pk->private_segment_size = k.priv; pk->group_segment_size = k.grp;
    pk->kernel_object = k.obj; pk->kernarg_address = karg; pk->completion_signal = sig;
    uint16_t hdr = (HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE) | (1 << HSA_PACKET_HEADER_BARRIER) |
                   (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE) |
                   (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE);
    auto t0 = std::chrono::steady_clock::now();
    __atomic_store_n((uint32_t*)pk, (uint32_t)hdr | ((uint32_t)setup << 16), __ATOMIC_RELEASE);
    hsa_signal_store_screlease(q->doorbell_signal, (hsa_signal_value_t)idx);
    hsa_signal_value_t v = 1;
    auto deadline = t0 + std::chrono::seconds(30);
    while ((v = hsa_signal_wait_scacquire(sig, HSA_SIGNAL_CONDITION_LT, 1, 1000000000ull, HSA_WAIT_STATE_BLOCKED)) >= 1) {
      if (std::chrono::steady_clock::now() > deadline) {
        std::fprintf(stderr, "dispatch did not complete within 30 s (signal %ld, read idx %lu write idx %lu)\n", (long)v,
                     (unsigned long)hsa_queue_load_read_index_scacquire(q), (unsigned long)hsa_queue_load_write_index_scacquire(q));
        std::exit(3);
      }
      ++spurious;
    }
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  }
  void fini() {
    hsa_signal_destroy(sig); hsa_queue_destroy(q); hsa_amd_memory_pool_free(karg);
    hsa_executable_destroy(exe); hsa_code_object_reader_destroy(reader); hsa_shut_down();
  }
};
