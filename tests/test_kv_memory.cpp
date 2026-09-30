#include "harness.hpp"
#include "lse/backends/cpu/cpu_backend.hpp"
#include "lse/kv/memory.hpp"
#include "lse/kv/allocator.hpp"

using namespace lse;

LSE_TEST(kv_arenas_fill_every_slot_before_growth_and_reuse_released_slots) {
  backend::BackendAdapter<backend::CpuBackend> cpu;
  LSE_EXPECT_OK(cpu.init(0));
  auto manager = kv::MemoryManager::create();
  std::vector<backend::DeviceBuffer> fragments;
  constexpr auto count = kv::kArenaBytes / kv::kFragmentBytes;
  for (std::size_t i = 0; i < count; ++i) {
    auto allocation = manager->acquire(cpu, backend::kDefaultStream);
    LSE_EXPECT_OK(allocation.status());
    if (!allocation.ok()) return;
    fragments.push_back(allocation.release());
    LSE_EXPECT_EQ(fragments.back().offset, i * kv::kFragmentBytes);
    LSE_EXPECT_EQ(manager->stats().arenas, 1u);
  }
  LSE_EXPECT_EQ(manager->stats().assigned_bytes, kv::kArenaBytes);
  LSE_EXPECT_EQ(manager->stats().reserved_bytes, kv::kArenaBytes);
  const auto released_offset = fragments[17].offset;
  fragments[17] = {};
  auto reused = manager->acquire(cpu, backend::kDefaultStream);
  LSE_EXPECT_OK(reused.status());
  if (!reused.ok()) return;
  fragments[17] = reused.release();
  LSE_EXPECT_EQ(fragments[17].offset, released_offset);
  LSE_EXPECT_EQ(manager->stats().arenas, 1u);
  auto next = manager->acquire(cpu, backend::kDefaultStream);
  LSE_EXPECT_OK(next.status());
  if (!next.ok()) return;
  auto second_arena = next.release();
  LSE_EXPECT_EQ(manager->stats().arenas, 2u);
  LSE_EXPECT_EQ(manager->stats().assigned_bytes,
                kv::kArenaBytes + kv::kFragmentBytes);
  auto retained = fragments.front();
  fragments.clear();
  LSE_EXPECT_EQ(manager->stats().arenas, 2u);
  LSE_EXPECT_EQ(manager->stats().assigned_bytes, 2u * kv::kFragmentBytes);
  retained = {};
  LSE_EXPECT_EQ(manager->stats().arenas, 1u);
  second_arena = {};
  LSE_EXPECT_EQ(manager->stats().reserved_bytes, 0u);
}

LSE_TEST(kv_fragment_lease_preserves_storage_after_manager_owner_is_released) {
  backend::BackendAdapter<backend::CpuBackend> cpu;
  LSE_EXPECT_OK(cpu.init(0));
  auto manager = kv::MemoryManager::create();
  auto allocated = manager->acquire(cpu, backend::kDefaultStream);
  LSE_EXPECT_OK(allocated.status());
  if (!allocated.ok()) return;
  auto fragment = allocated.release();
  manager.reset();
  const std::uint32_t expected = 0x12345678;
  std::uint32_t actual = 0;
  LSE_EXPECT_OK(cpu.copy_h2d(&expected, fragment, sizeof(expected), 0));
  LSE_EXPECT_OK(cpu.copy_d2h(fragment, &actual, sizeof(actual), 0));
  LSE_EXPECT_EQ(actual, expected);
  fragment = {};
}

LSE_TEST(kv_fragment_growth_preserves_addresses_and_cross_fragment_data) {
  backend::BackendAdapter<backend::CpuBackend> cpu;
  LSE_EXPECT_OK(cpu.init(0));
  auto manager = kv::MemoryManager::create();
  auto storage = std::make_shared<kv::FragmentStorage>(manager, cpu, backend::kDefaultStream);
  LSE_EXPECT_OK(storage->reserve(2 * kv::kFragmentBytes));
  LSE_EXPECT_EQ(manager->stats().reserved_bytes, 0u);
  LSE_EXPECT_OK(storage->grow(kv::kFragmentBytes));
  auto first_binding = storage->binding();
  LSE_EXPECT_OK(first_binding.status());
  if (!first_binding.ok()) return;
  auto first = first_binding.release();
  std::uint64_t old_address = 0;
  LSE_EXPECT_OK(cpu.copy_d2h(first, &old_address, sizeof(old_address), 0));
  const std::uint32_t marker = 0x12345678;
  LSE_EXPECT_OK(storage->write(&marker, sizeof(marker), kv::kFragmentBytes - sizeof(marker)));
  LSE_EXPECT_OK(storage->grow(kv::kFragmentBytes * 2));
  auto next_binding = storage->binding();
  LSE_EXPECT_OK(next_binding.status());
  if (!next_binding.ok()) return;
  auto next = next_binding.release();
  LSE_EXPECT(first.ptr == next.ptr);
  LSE_EXPECT_EQ(first.handle, next.handle);
  std::uint64_t addresses[2]{};
  LSE_EXPECT_OK(cpu.copy_d2h(next, addresses, sizeof(addresses), 0));
  LSE_EXPECT_EQ(addresses[0], old_address);
  std::uint32_t boundary[2]{};
  LSE_EXPECT_OK(storage->read(boundary, sizeof(boundary), kv::kFragmentBytes - sizeof(marker)));
  LSE_EXPECT_EQ(boundary[0], marker);
  LSE_EXPECT_EQ(boundary[1], 0u);
  boundary[1] = 0xabcdef01;
  LSE_EXPECT_OK(storage->write(boundary, sizeof(boundary), kv::kFragmentBytes - sizeof(marker)));
  std::uint32_t actual[2]{};
  LSE_EXPECT_OK(storage->read(actual, sizeof(actual), kv::kFragmentBytes - sizeof(marker)));
  LSE_EXPECT_EQ(actual[0], boundary[0]);
  LSE_EXPECT_EQ(actual[1], boundary[1]);
  LSE_EXPECT_EQ(manager->stats().assigned_bytes, 2u * kv::kFragmentBytes);
  storage.reset();
  next = {};
  LSE_EXPECT_EQ(manager->stats().arenas, 1u);
  first = {};
  LSE_EXPECT_EQ(manager->stats().arenas, 0u);
}

LSE_TEST(kv_block_growth_consumes_existing_free_blocks_first) {
  kv::BlockAllocator pool(4);
  auto first = pool.acquire();
  LSE_EXPECT_OK(first.status());
  if (!first.ok()) return;
  LSE_EXPECT_EQ(*first, 0u);
  LSE_EXPECT_OK(pool.grow(8));
  for (std::uint32_t expected = 1; expected < 8; ++expected) {
    auto block = pool.acquire();
    LSE_EXPECT_OK(block.status());
    if (!block.ok()) return;
    LSE_EXPECT_EQ(*block, expected);
  }
}

LSE_TEST_MAIN()
