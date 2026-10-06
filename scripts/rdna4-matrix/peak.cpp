// Minimal HSA launcher for matrix-core microbenchmarks.
// usage: peak <code.hsaco> <iters> <wg_per_cu> <wg_size> <reps> <kernel:ops_per_wave_iter>...
//   ops_per_wave_iter: arithmetic ops (2*M*N*K, dense-equivalent for SWMMAC) one wave
//   performs per loop iteration. Reports the best of <reps> host-timed dispatches.
// Device properties (CU count, max clock) are queried, never assumed.
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
  std::fprintf(stderr, "%s failed: %s (0x%x) line %d\n", #x, m_ ? m_ : "?", (unsigned)s_, __LINE__); std::exit(2);} } while (0)

static hsa_agent_t g_gpu{};
static hsa_amd_memory_pool_t g_dev_pool{}, g_kernarg_pool{};

int main(int argc, char** argv) {
  if (argc < 7) { std::fprintf(stderr, "usage: %s hsaco iters wg_per_cu wg_size reps kernel:ops_per_wave_iter...\n", argv[0]); return 1; }
  const char* path = argv[1]; int iters = atoi(argv[2]); int wg_per_cu = atoi(argv[3]); int wg_size = atoi(argv[4]);
  int reps = atoi(argv[5]);
  CK(hsa_init());
  hsa_iterate_agents([](hsa_agent_t a, void*) -> hsa_status_t {
    hsa_device_type_t t; hsa_agent_get_info(a, HSA_AGENT_INFO_DEVICE, &t);
    if (t == HSA_DEVICE_TYPE_GPU && !g_gpu.handle) g_gpu = a; return HSA_STATUS_SUCCESS; }, nullptr);
  if (!g_gpu.handle) { std::fprintf(stderr, "no GPU agent\n"); return 2; }
  char name[64] = {}; uint32_t cus = 0, mhz = 0, wave = 0;
  CK(hsa_agent_get_info(g_gpu, HSA_AGENT_INFO_NAME, name));
  CK(hsa_agent_get_info(g_gpu, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_COMPUTE_UNIT_COUNT, &cus));
  CK(hsa_agent_get_info(g_gpu, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_MAX_CLOCK_FREQUENCY, &mhz));
  CK(hsa_agent_get_info(g_gpu, HSA_AGENT_INFO_WAVEFRONT_SIZE, &wave));
  hsa_amd_agent_iterate_memory_pools(g_gpu, [](hsa_amd_memory_pool_t p, void*) -> hsa_status_t {
    hsa_amd_segment_t seg; hsa_amd_memory_pool_get_info(p, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &seg);
    if (seg == HSA_AMD_SEGMENT_GLOBAL && !g_dev_pool.handle) g_dev_pool = p; return HSA_STATUS_SUCCESS; }, nullptr);
  hsa_agent_t cpu{};
  hsa_iterate_agents([](hsa_agent_t a, void* d) -> hsa_status_t {
    hsa_device_type_t t; hsa_agent_get_info(a, HSA_AGENT_INFO_DEVICE, &t);
    if (t == HSA_DEVICE_TYPE_CPU) *(hsa_agent_t*)d = a; return HSA_STATUS_SUCCESS; }, &cpu);
  if (cpu.handle) hsa_amd_agent_iterate_memory_pools(cpu, [](hsa_amd_memory_pool_t p, void*) -> hsa_status_t {
    hsa_amd_segment_t seg; hsa_amd_memory_pool_get_info(p, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &seg);
    uint32_t fl = 0; hsa_amd_memory_pool_get_info(p, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &fl);
    if (seg == HSA_AMD_SEGMENT_GLOBAL && (fl & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT) && !g_kernarg_pool.handle) g_kernarg_pool = p;
    return HSA_STATUS_SUCCESS; }, nullptr);
  if (!g_kernarg_pool.handle) {
    hsa_amd_agent_iterate_memory_pools(g_gpu, [](hsa_amd_memory_pool_t p, void*) -> hsa_status_t {
      hsa_amd_segment_t seg; hsa_amd_memory_pool_get_info(p, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &seg);
      uint32_t fl = 0; hsa_amd_memory_pool_get_info(p, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &fl);
      if (seg == HSA_AMD_SEGMENT_GLOBAL && (fl & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT) && !g_kernarg_pool.handle) g_kernarg_pool = p;
      return HSA_STATUS_SUCCESS; }, nullptr);
  }
  if (!g_kernarg_pool.handle) { std::fprintf(stderr, "no kernarg pool\n"); return 2; }

  std::ifstream f(path, std::ios::binary); std::vector<char> obj((std::istreambuf_iterator<char>(f)), {});
  hsa_code_object_reader_t reader; CK(hsa_code_object_reader_create_from_memory(obj.data(), obj.size(), &reader));
  hsa_executable_t exe; CK(hsa_executable_create_alt(HSA_PROFILE_BASE, HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT, nullptr, &exe));
  CK(hsa_executable_load_agent_code_object(exe, g_gpu, reader, nullptr, nullptr));
  CK(hsa_executable_freeze(exe, nullptr));
  hsa_queue_t* q; CK(hsa_queue_create(g_gpu, 64, HSA_QUEUE_TYPE_SINGLE, nullptr, nullptr, UINT32_MAX, UINT32_MAX, &q));
  hsa_signal_t sig; CK(hsa_signal_create(1, 0, nullptr, &sig));
  for (int argi = 6; argi < argc; ++argi) {
  std::string spec = argv[argi]; auto colon = spec.find(':');
  std::string kn = spec.substr(0, colon); const char* kname = kn.c_str();
  double ops_per_wave_iter = atof(spec.c_str() + colon + 1);
  std::string sym = std::string(kname) + ".kd";
  hsa_executable_symbol_t ks; CK(hsa_executable_get_symbol_by_name(exe, sym.c_str(), &g_gpu, &ks));
  uint64_t kobj; uint32_t kargsz, grpsz, privsz;
  CK(hsa_executable_symbol_get_info(ks, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &kobj));
  CK(hsa_executable_symbol_get_info(ks, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE, &kargsz));
  CK(hsa_executable_symbol_get_info(ks, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE, &grpsz));
  CK(hsa_executable_symbol_get_info(ks, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE, &privsz));

  const uint32_t groups = cus * (uint32_t)wg_per_cu;
  void* out = nullptr; CK(hsa_amd_memory_pool_allocate(g_dev_pool, (size_t)groups * wg_size * 64, 0, &out));
  void* karg = nullptr; CK(hsa_amd_memory_pool_allocate(g_kernarg_pool, kargsz < 64 ? 64 : kargsz, 0, &karg));
  hsa_amd_agents_allow_access(1, &g_gpu, nullptr, karg);
  std::memset(karg, 0, kargsz);
  int flag = 0;
  std::memcpy((char*)karg + 0, &out, 8); std::memcpy((char*)karg + 8, &iters, 4); std::memcpy((char*)karg + 12, &flag, 4);

  auto dispatch = [&](double& host_ms, double& gpu_ms) {
    hsa_signal_store_screlease(sig, 1);
    uint64_t idx = hsa_queue_add_write_index_relaxed(q, 1);
    auto* pk = (hsa_kernel_dispatch_packet_t*)q->base_address + (idx & (q->size - 1));
    std::memset((char*)pk + 4, 0, sizeof(*pk) - 4);
    pk->setup = 1 << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS;
    pk->workgroup_size_x = (uint16_t)wg_size; pk->workgroup_size_y = 1; pk->workgroup_size_z = 1;
    pk->grid_size_x = groups * (uint32_t)wg_size; pk->grid_size_y = 1; pk->grid_size_z = 1;
    pk->private_segment_size = privsz; pk->group_segment_size = grpsz;
    pk->kernel_object = kobj; pk->kernarg_address = karg; pk->completion_signal = sig;
    uint16_t hdr = (HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE) |
                   (1 << HSA_PACKET_HEADER_BARRIER) |
                   (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE) |
                   (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE);
    auto t0 = std::chrono::steady_clock::now();
    __atomic_store_n((uint32_t*)pk, (uint32_t)hdr | ((uint32_t)pk->setup << 16), __ATOMIC_RELEASE);
    hsa_signal_store_screlease(q->doorbell_signal, (hsa_signal_value_t)idx);
    if (hsa_signal_wait_scacquire(sig, HSA_SIGNAL_CONDITION_LT, 1, 30ull * 1000000000ull, HSA_WAIT_STATE_BLOCKED) != 0) {
      std::fprintf(stderr, "dispatch did not complete within 30 s\n"); std::exit(3);
    }
    auto t1 = std::chrono::steady_clock::now();
    host_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    gpu_ms = -1;
  };
  double h, g, best_h = 1e30, best_g = 1e30;
  dispatch(h, g);  // warmup
  for (int r = 0; r < reps; ++r) { dispatch(h, g); if (h < best_h) best_h = h; if (g > 0 && g < best_g) best_g = g; }
  const double waves = double(groups) * wg_size / wave;
  const double ops = waves * iters * ops_per_wave_iter;
  const double ms = best_g < 1e29 ? best_g : best_h;
  const double tops = ops / (ms * 1e-3) / 1e12;
  const double per_cu_clk = ops / (ms * 1e-3) / (double(cus) * double(mhz) * 1e6);
  std::printf("%-14s agent=%s cus=%u max_mhz=%u wave=%u groups=%u wg=%d vgpr_priv=%u  host_ms=%.3f gpu_ms=%.3f  TOPS=%.1f  ops/clk/CU@max=%.0f\n",
              kname, name, cus, mhz, wave, groups, wg_size, privsz, best_h, best_g < 1e29 ? best_g : -1.0, tops, per_cu_clk);
  hsa_amd_memory_pool_free(out); hsa_amd_memory_pool_free(karg);
  std::fflush(stdout);
  }
  hsa_signal_destroy(sig); hsa_queue_destroy(q);
  hsa_executable_destroy(exe); hsa_code_object_reader_destroy(reader);
  hsa_shut_down();
  return 0;
}
