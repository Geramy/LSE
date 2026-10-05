// Model-independent layer machinery: weight binding and the generic block.
#include "lse/model/layer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cstring>
#include <mutex>
#include <new>
#include <vector>

// host_bytes / sync_to_device: a checkpoint tensor is host data and Array has
// no host-write path, so the buffer is filled behind the graph.
#include "lse/core/file_read.hpp"
#include "lse/graph/interpreter.hpp"
#include "lse/graph/ops.hpp"
#include "lse/ops/norm.hpp"
#include "lse/dispatch/q8_matrix.hpp"
#include "lse/quant/q8_matrix_pack.hpp"

namespace lse::model {

// Weight-load accounting, reported by WeightBinder::finish() under
// LSE_TIME_LOAD=1. Times are wall time on the loading thread; an upload's
// time is what it took to hand the bytes to the backend, which for a backend
// that queues uploads is its wait for staging room -- the copy engine's pace.
double g_load_total_ms = 0.0;
double g_load_host_ms = 0.0;
double g_load_dev_ms = 0.0;
double g_load_read_ms = 0.0;
double g_load_gather_ms = 0.0;
double g_load_alloc_ms = 0.0;
double g_load_zeros_ms = 0.0;
double g_load_buf_ms = 0.0;
double g_load_direct_ms = 0.0;
double g_load_pack_ms = 0.0;
double g_load_finish_ms = 0.0;
std::size_t g_load_tensors = 0;
std::size_t g_load_direct_bytes = 0;   // file -> device, no host pass
std::size_t g_load_staged_bytes = 0;   // read whole, then gathered/widened
std::size_t g_load_pushed_bytes = 0;   // host results pushed (gathers, packs)
namespace {
// Weights get windows of a few big allocations, not one allocation each.
//
// Every device allocation is a driver round trip, and the cost is per CALL,
// not per byte: on this box it measured 13.7 ms apiece, so 1847 tensors spent
// 25 s in allocation alone against 2.2 s to read the checkpoint and 0.5 s to
// upload it. That dominated model load, and on two GPUs -- twice the
// allocations, both waited on -- it pushed load past any sane timeout and read
// as a hang. Slabbing makes the count scale with total bytes instead of with
// how finely the checkpoint is split.
//
// Lifetime is unchanged: a weight buffer is written once and read until the
// process exits, and the per-tensor allocations were never released either.
// The slabs outlive every window by construction, so a window can never
// outlive its memory.
struct WeightSlab {
  const backend::IBackend* be = nullptr;
  std::uint32_t stream = 0;
  backend::DeviceBuffer base;
  std::size_t used = 0;
};

std::size_t weight_slab_bytes(const backend::IBackend&) {
  // Bound unused tail space without returning to per-tensor allocations.
  return kWeightSlabBytes;
}

std::mutex& slab_mutex() {
  static std::mutex mu;
  return mu;
}
std::vector<WeightSlab>& weight_slabs() {
  static std::vector<WeightSlab> slabs;
  return slabs;
}

Result<backend::DeviceBuffer> slab_window(std::size_t bytes,
                                        backend::IBackend& be,
                                        backend::Stream at) {
  constexpr std::size_t kAlign = kWeightAlignment;
  const auto kSlab = weight_slab_bytes(be);
  std::vector<WeightSlab>& slabs = weight_slabs();
  const std::size_t need = (bytes + kAlign - 1) & ~(kAlign - 1);

  const std::lock_guard lock(slab_mutex());
  WeightSlab* use = nullptr;
  for (WeightSlab& s : slabs) {
    if (s.be == &be && s.stream == at.index &&
        s.used + need <= s.base.size_bytes) {
      use = &s;
      break;
    }
  }
  if (use == nullptr) {
    // A tensor bigger than the slab gets its own exact-sized one rather than
    // rounding a single embedding table up to the next slab boundary.
    const std::size_t want = need > kSlab ? need : kSlab;
    const backend::ScopedAllocationSite site(backend::AllocationSite::kWeights);
    auto got = be.allocate(want, backend::MemoryClass::kDevice, at);
    if (!got.ok()) return got.status();
    WeightSlab made;
    made.be = &be;
    made.stream = at.index;
    made.base = got.release();
    slabs.push_back(std::move(made));
    use = &slabs.back();
    if (std::getenv("LSE_TIME_LOAD") != nullptr) {
      std::size_t total = 0;
      for (const WeightSlab& slab : slabs) total += slab.base.size_bytes;
      std::fprintf(stderr, "lse weights: %zu live slabs, %.3f GiB reserved; "
                           "new slab %.1f MiB\n",
                   slabs.size(), double(total) / double(std::size_t{1} << 30),
                   double(want) / double(std::size_t{1} << 20));
      std::fflush(stderr);
    }
  }
  // A view: same allocation, same residency and member, its own window. This
  // is the shape DeviceBuffer already documents for a reshape's alias.
  backend::DeviceBuffer view = use->base;
  view.offset = use->base.offset + use->used;
  view.size_bytes = bytes;
  use->used += need;
  return view;
}
}  // namespace


bool packs_q8_matrix(const quant::GroupAffine& spec, std::size_t rank,
                     DType scales, std::int64_t rows,
                     std::int64_t features) noexcept {
  return spec.bits == 8 && spec.group_size == 64 && rank == 2 &&
         scales == DType::kBF16 && rows > 0 && rows <= UINT32_MAX &&
         features > 0 && features <= UINT32_MAX &&
         dispatch::q8_packed_weight_shape(static_cast<std::uint32_t>(rows),
                                          static_cast<std::uint32_t>(features));
}

std::size_t packed_q8_matrix_bytes(std::int64_t rows,
                                   std::int64_t features) noexcept {
  if (rows <= 0 || features <= 0) return 0;
  const auto align = [](std::size_t bytes) {
    return (bytes + kWeightAlignment - 1) & ~(kWeightAlignment - 1);
  };
  const auto padded = static_cast<std::size_t>((rows + 15) / 16 * 16);
  const auto words = padded * static_cast<std::size_t>(features / 4);
  const auto groups = padded * static_cast<std::size_t>(features / 64);
  return align(dtype_storage_bytes(DType::kU32, words)) +
         2 * align(dtype_storage_bytes(DType::kBF16, groups));
}

std::size_t release_weight_slabs() {
  std::vector<WeightSlab> released;
  {
    const std::lock_guard lock(slab_mutex());
    released.swap(weight_slabs());
  }
  std::size_t bytes = 0;
  for (const WeightSlab& slab : released) bytes += slab.base.size_bytes;
  // A slab also lives on in every window still held, so dropping this vector
  // frees exactly the slabs no live weight uses.
  return bytes;
}

Status WeightBinder::finish(std::string_view what) {
  const auto t0 = std::chrono::steady_clock::now();
  Status first = OkStatus();
  if (graph::Scheduler* sched = graph::default_scheduler()) {
    backend::IDeviceSet& set = sched->devices();
    const auto finish_one = [&first](backend::IBackend& be) {
      const Status s = be.finish_uploads();
      if (!s.ok() && first.ok()) first = s;
    };
    if (set.size() == 0) {
      finish_one(sched->backend());
    } else {
      for (std::size_t i = 0; i < set.size(); ++i) finish_one(set.device(i));
    }
  }
  g_load_finish_ms += std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - t0)
                          .count();
  if (std::getenv("LSE_TIME_LOAD") != nullptr) {
    const double mib = 1048576.0;
    const std::size_t moved =
        g_load_direct_bytes + g_load_pushed_bytes;
    const double ms = g_load_total_ms + g_load_finish_ms;
    std::fprintf(stderr,
                 "[load] %.*s: %zu tensors, %.0f ms (%.2f GB/s to the device): "
                 "file->device %.0f MiB, read whole %.0f MiB, pushed from host "
                 "%.0f MiB | zeros %.0f + devbuf %.0f + direct %.0f + "
                 "stage-alloc %.0f + stage-read %.0f + gather %.0f + pack %.0f "
                 "+ push %.0f + finish %.0f ms\n",
                 static_cast<int>(what.size()), what.data(), g_load_tensors, ms,
                 ms > 0 ? double(moved) / (ms * 1e6) : 0.0,
                 double(g_load_direct_bytes) / mib,
                 double(g_load_staged_bytes) / mib,
                 double(g_load_pushed_bytes) / mib, g_load_zeros_ms,
                 g_load_buf_ms, g_load_direct_ms, g_load_alloc_ms,
                 g_load_read_ms, g_load_gather_ms, g_load_pack_ms,
                 g_load_dev_ms, g_load_finish_ms);
    std::fflush(stderr);
  }
  g_load_total_ms = g_load_host_ms = g_load_dev_ms = g_load_read_ms = 0.0;
  g_load_gather_ms = g_load_alloc_ms = g_load_zeros_ms = g_load_buf_ms = 0.0;
  g_load_direct_ms = g_load_pack_ms = g_load_finish_ms = 0.0;
  g_load_tensors = g_load_direct_bytes = g_load_staged_bytes = 0;
  g_load_pushed_bytes = 0;
  return first;
}

Result<Array> WeightBinder::require(std::string_view name) {
  auto got = optional(name);
  if (got.ok()) return got;
  // Only a missing tensor becomes a "no such tensor" report; anything else
  // (no backend, allocation failure) would otherwise be misattributed.
  if (got.status().code() != StatusCode::kNotFound) return got.status();
  return LSE_ERROR(kNotFound, "checkpoint has no tensor '", std::string(name),
                   "'");
}

namespace {

// The dtype a checkpoint tensor is held in on the device. Float formats keep
// the format they were stored in — widening them in memory is a pure bandwidth
// tax on every token, and converting in register costs nothing. So does the
// packed plane of a group-affine weight: quant_linear unpacks it in register,
// and widening it here would not just cost bandwidth, it would destroy the
// codes. Block-quantized storage still widens; no kernel reads it directly.
DType device_storage(DType checkpoint) noexcept {
  switch (checkpoint) {
    case DType::kF32:
    case DType::kF16:
    case DType::kBF16:
    case DType::kU32:
      return checkpoint;
    default:
      return DType::kF32;
  }
}

// Uploads one checkpoint tensor and reports it as `shape`. `order`, when
// non-null, selects and reorders rows — a row being one span of the tensor's
// last axis.
//
// A weight is data, not a computation. Going through eval() would dispatch a
// fill kernel across every element and then overwrite the result one element
// at a time — for the tied head that is a quarter of a billion pointless
// writes on each side, and it dominated model load. Allocate the buffer and
// read the tensor straight into it instead.
// `window`, when non-empty, keeps only that span of every row's last axis.
// Staged whole and copied span by span for the same reason the row path is:
// one sequential pass over the mapping beats a scattered read per row.
Result<Array> upload(const TensorView& v, Shape shape,
                     const std::vector<std::int64_t>* order,
                     std::int64_t win_first = 0, std::int64_t win_count = 0) {
  // LSE_TIME_LOAD=1: split weight loading into the read/gather on the host and
  // the upload to the device, so a slow load names its own half.
  struct Phase {
    std::chrono::steady_clock::time_point t0;
    double* sink;
    explicit Phase(double* into)
        : t0(std::chrono::steady_clock::now()), sink(into) {}
    ~Phase() {
      *sink += std::chrono::duration<double, std::milli>(
                   std::chrono::steady_clock::now() - t0).count();
    }
  };
  Phase whole{&g_load_total_ms};
  ++g_load_tensors;
  graph::Scheduler* sched = graph::default_scheduler();
  if (sched == nullptr) {
    return LSE_ERROR(kInternal, "no usable backend to load '", v.name,
                     "' into");
  }
  // Where this tensor lives. Weight loading names a member per layer so a
  // model spans the pool; everything else leaves it unset and gets the
  // primary, which is what a single-device run has always done.
  backend::IDeviceSet& set = sched->devices();
  const std::size_t member = graph::preferred_member();
  backend::IBackend& be =
      member < set.size() ? set.device(member) : sched->backend();
  // On a set that places by stream, the stream IS the member -- this is what
  // puts a shard's weights in its own card's VRAM.
  const backend::Stream at =
      member < set.size()
          ? set.stream_for(member).value_or(backend::kDefaultStream)
          : backend::kDefaultStream;

  const DType dt = device_storage(v.dtype);
  double* zsink = &g_load_zeros_ms;
  Array a = [&]{ Phase z{zsink}; return Array::zeros(shape, dt); }();
  graph::Node& n = *a.node();
  { Phase b{&g_load_buf_ms};
    auto win = slab_window(dtype_storage_bytes(n.dtype, n.element_count()),
                           be, at);
    if (!win.ok()) return win.status();
    n.buffer = win.release(); }
  const bool native = dt == v.dtype;

  if (order == nullptr && win_count <= 0) {
    Phase d{&g_load_direct_ms};
    const std::size_t want = dtype_storage_bytes(dt, n.element_count());
    if (native && v.data.size() >= want) {
      // Straight from the checkpoint file to the card, with no host copy of
      // the tensor: the backend reads the file into its own staging and
      // queues the transfer, so the next tensor is being read while this one
      // crosses the link. Read through the file rather than the mapping: the
      // mapping faults one page at a time (1.4 GB/s cold, measured) where a
      // positioned read runs at the drive's rate. The device holds the truth
      // from here; anything that wants these bytes on the host pulls them
      // back the way it would for any other device-resident value.
      if (v.fd >= 0) {
        LSE_RETURN_IF_ERROR(
            be.upload_file(v.fd, v.file_offset, n.buffer, want, 0));
      } else {
        LSE_RETURN_IF_ERROR(be.upload(v.data.data(), n.buffer, want, 0));
      }
      g_load_direct_bytes += want;
      n.materialized = true;
      n.device_dirty = true;
      n.host_dirty = false;
      return a;
    }
  }

  // Everything below produces the tensor on the host first -- widened, or
  // gathered by rows or by a window of every row -- and then sends it. A
  // device buffer with a host address (the CPU backend) takes the result in
  // place; otherwise it is built in scratch memory that is unmapped as soon
  // as the backend has taken the bytes, rather than in the node's host
  // mirror, which would be freed into the allocator's cache of large blocks
  // and stay resident.
  const std::size_t out_bytes = dtype_storage_bytes(dt, n.element_count());
  HostScratch built;
  std::byte* out = nullptr;
  if (n.buffer.ptr != nullptr) {
    out = static_cast<std::byte*>(graph::interpreter::host_bytes(n));
  } else {
    LSE_ASSIGN_OR(built, HostScratch::allocate(out_bytes, "'" + v.name + "'"));
    out = built.data();
  }
  // The whole stored tensor, for the gathers: one positioned read of the
  // file (the mapping would fault in page by page; see the direct path), or
  // the widened copy when the dtype changes.
  HostScratch staged;
  const auto stage_whole = [&](std::size_t whole) -> Result<const std::byte*> {
    if (native && v.data.size() >= whole && v.fd < 0) return v.data.data();
    Phase al{&g_load_alloc_ms};
    LSE_ASSIGN_OR(staged, HostScratch::allocate(whole, "'" + v.name + "'"));
    Phase rd{&g_load_read_ms};
    if (native && v.data.size() >= whole) {
      LSE_RETURN_IF_ERROR(v.read_file(staged.data(), whole));
      g_load_staged_bytes += whole;
    } else if (native) {
      LSE_RETURN_IF_ERROR(v.read_native(staged.data(), whole));
    } else {
      LSE_RETURN_IF_ERROR(v.read_f32(reinterpret_cast<float*>(staged.data()),
                                     v.element_count()));
    }
    return static_cast<const std::byte*>(staged.data());
  };

  if (order == nullptr && win_count <= 0) {
    Phase d{&g_load_direct_ms};
    if (native) {
      LSE_RETURN_IF_ERROR(v.read_native(out, out_bytes));
    } else {
      LSE_RETURN_IF_ERROR(
          v.read_f32(reinterpret_cast<float*>(out), n.element_count()));
    }
  } else if (order == nullptr) {
    const std::size_t rank = v.shape.rank();
    const auto width =
        static_cast<std::size_t>(rank >= 2 ? v.shape.dim(rank - 1) : 1);
    const std::size_t elem = dtype_storage_bytes(dt, 1);
    const std::size_t rows = width > 0 ? v.element_count() / width : 0;
    LSE_ASSIGN_OR(const std::byte* from,
                  stage_whole(dtype_storage_bytes(dt, v.element_count())));
    const auto first = static_cast<std::size_t>(win_first);
    const auto count = static_cast<std::size_t>(win_count);
    Phase gv{&g_load_gather_ms};
    for (std::size_t r = 0; r < rows; ++r) {
      std::memcpy(out + r * count * elem,
                  from + (r * width + first) * elem, count * elem);
    }
  } else {
    // Staged whole rather than read row by row: one sequential read beats
    // `order->size()` scattered ones.
    const std::size_t rank = v.shape.rank();
    const auto width =
        static_cast<std::size_t>(rank >= 2 ? v.shape.dim(rank - 1) : 1);
    const std::size_t elem = dtype_storage_bytes(dt, 1);
    LSE_ASSIGN_OR(const std::byte* from,
                  stage_whole(dtype_storage_bytes(dt, v.element_count())));
    const std::size_t row_bytes = width * elem;
    Phase gr{&g_load_gather_ms};
    for (std::size_t i = 0; i < order->size(); ++i) {
      std::memcpy(
          out + i * row_bytes,
          from + static_cast<std::size_t>((*order)[i]) * row_bytes,
          row_bytes);
    }
  }

  n.materialized = true;
  if (n.buffer.ptr != nullptr) {
    // Written in place: the host copy is the device's memory.
    n.host_dirty = true;
    n.device_dirty = false;
    return a;
  }
  {
    // The backend takes the bytes into its staging before returning, so the
    // scratch can go while the transfer is still queued. The device holds
    // the weight from here; anything that wants the bytes on the host pulls
    // them back.
    Phase up{&g_load_dev_ms};
    LSE_RETURN_IF_ERROR(be.upload(out, n.buffer, out_bytes, 0));
    g_load_pushed_bytes += out_bytes;
  }
  n.host_dirty = false;
  n.device_dirty = true;
  return a;
}

std::int64_t last_dim(const Shape& s) {
  return s.rank() == 0 ? 0 : s.dim(s.rank() - 1);
}

Status check_rows(std::string_view name, const Shape& shape,
                  const std::vector<std::int64_t>& order) {
  const auto width = static_cast<std::size_t>(
      shape.rank() >= 2 ? shape.dim(shape.rank() - 1) : 1);
  const std::size_t rows = width > 0 ? shape.elem_count() / width : 0;
  if (width == 0) {
    return LSE_ERROR(kInvalidArgument, "'", std::string(name),
                     "' has no rows to take");
  }
  for (std::int64_t r : order) {
    if (r < 0 || static_cast<std::size_t>(r) >= rows) {
      return LSE_ERROR(kOutOfRange, "row ", std::to_string(r), " of '",
                       std::string(name), "' is outside its ",
                       std::to_string(rows), " rows");
    }
  }
  return OkStatus();
}

}  // namespace

Result<std::array<const TensorView*, 2>> WeightBinder::quant_planes(
    std::string_view name) const {
  constexpr std::string_view kWeight = ".weight";
  std::array<const TensorView*, 2> planes{nullptr, nullptr};
  if (name.size() <= kWeight.size() ||
      name.substr(name.size() - kWeight.size()) != kWeight) {
    return planes;
  }
  const std::string base(name.substr(0, name.size() - kWeight.size()));
  planes[0] = weights_->find(base + ".scales");
  planes[1] = weights_->find(base + ".biases");
  if ((planes[0] == nullptr) != (planes[1] == nullptr)) {
    const bool have_scales = planes[0] != nullptr;
    return LSE_ERROR(kInvalidArgument, "'", std::string(name), "' has a '",
                     have_scales ? ".scales" : ".biases", "' plane but no '",
                     have_scales ? ".biases" : ".scales",
                     "'; a group-affine weight needs both");
  }
  return planes;
}

void WeightBinder::record_original_upload(std::string_view name, const Array& value) {
  original_uploads_.push_back({std::string(name), value.node()->buffer.residency,
                               value.node()->buffer.member});
}

std::vector<std::size_t> WeightBinder::remaining_original_bytes(
    const backend::DeviceBuffer& placement) const {
  std::vector<std::size_t> remaining;
  for (const auto& [name, tensor] : weights_->tensors()) {
    const auto complete = std::any_of(original_uploads_.begin(), original_uploads_.end(),
        [&](const UploadedOriginal& uploaded) {
          return uploaded.name == name && uploaded.residency == placement.residency &&
                 uploaded.member == placement.member;
        });
    if (!complete)
      remaining.push_back(dtype_storage_bytes(device_storage(tensor.dtype), tensor.element_count()));
  }
  return remaining;
}

Result<Array> WeightBinder::optional(std::string_view name) {
  const TensorView *v = weights_->find(name);
  if (v == nullptr)
    return LSE_ERROR(kNotFound, std::string(name));

  LSE_ASSIGN_OR(const auto planes, quant_planes(name));
  if (planes[0] == nullptr) {
    LSE_ASSIGN_OR(Array a, upload(*v, v->shape, nullptr));
    record_original_upload(name, a);
    claimed_.emplace_back(name);
    return a;
  }
  return bind_quantized(name, *v, *planes[0], *planes[1], nullptr, Shape{});
}

Result<Array> WeightBinder::require_rows(std::string_view name,
                                         const std::vector<std::int64_t> &order,
                                         Shape shape) {
  const TensorView* v = weights_->find(name);
  if (v == nullptr) {
    return LSE_ERROR(kNotFound, "checkpoint has no tensor '", std::string(name),
                     "'");
  }
  LSE_ASSIGN_OR(const auto planes, quant_planes(name));
  if (planes[0] != nullptr) {
    return bind_quantized(name, *v, *planes[0], *planes[1], &order, shape);
  }

  const std::size_t rank = v->shape.rank();
  const auto width =
      static_cast<std::size_t>(rank >= 2 ? v->shape.dim(rank - 1) : 1);
  const std::size_t rows = width > 0 ? v->element_count() / width : 0;
  if (width == 0 || shape.elem_count() != order.size() * width) {
    return LSE_ERROR(kInvalidArgument, "'", std::string(name), "' has ",
                     std::to_string(rows), " rows of ", std::to_string(width),
                     "; taking ", std::to_string(order.size()),
                     " of them cannot fill the requested ",
                     std::to_string(shape.elem_count()), " elements");
  }
  LSE_RETURN_IF_ERROR(check_rows(name, v->shape, order));
  LSE_ASSIGN_OR(Array a, upload(*v, shape, &order));
  claimed_.emplace_back(name);
  return a;
}

Result<Array> WeightBinder::require_as(std::string_view name, Shape shape) {
  const TensorView* v = weights_->find(name);
  if (v == nullptr) {
    return LSE_ERROR(kNotFound, "checkpoint has no tensor '", std::string(name),
                     "'");
  }
  if (v->element_count() != shape.elem_count()) {
    return LSE_ERROR(kInvalidArgument, "'", std::string(name), "' is ",
                     v->shape.to_string(), ", which is not ",
                     shape.to_string(), " read differently");
  }
  LSE_ASSIGN_OR(Array a, upload(*v, shape, nullptr));
  record_original_upload(name, a);
  claimed_.emplace_back(name);
  return a;
}

Result<Array> WeightBinder::bind_quantized(
    std::string_view name, const TensorView& packed, const TensorView& scales,
    const TensorView& biases, const std::vector<std::int64_t>* order,
    Shape logical, TensorWindow window) {
  if (quantization_ == nullptr) {
    return LSE_ERROR(kInvalidArgument, "'", std::string(name),
                     "' is stored with .scales/.biases planes but the config "
                     "declared no quantization block, so its group size is "
                     "unknown");
  }
  if (packed.dtype != DType::kU32) {
    return LSE_ERROR(kInvalidArgument, "'", std::string(name),
                     "' has .scales/.biases planes but is stored as ",
                     to_string(packed.dtype),
                     "; a group-affine plane is packed into u32 lanes");
  }
  if (scales.dtype != biases.dtype) {
    return LSE_ERROR(kInvalidArgument, "'", std::string(name),
                     "' stores its scales as ", to_string(scales.dtype),
                     " and its biases as ", to_string(biases.dtype));
  }
  // Rank 2 is one [out, in] matrix; rank 3 is MLX's SwitchGLU stack, read by
  // quant_linear_indexed one expert at a time. Nothing is unstacked here — the
  // expert axis stays the leading axis of the plane all the way to the kernel.
  const std::size_t rank = packed.shape.rank();
  if (rank != 2 && rank != 3) {
    return LSE_ERROR(kUnimplemented, "'", std::string(name), "' is ",
                     packed.shape.to_string(),
                     "; a group-affine weight is read as an [out, in] matrix "
                     "or an [expert, out, in] stack of them");
  }
  bool agree = scales.shape == biases.shape && scales.shape.rank() == rank;
  for (std::size_t i = 0; agree && i + 1 < rank; ++i) {
    if (scales.shape.dim(i) != packed.shape.dim(i)) agree = false;
  }
  if (!agree) {
    return LSE_ERROR(kInvalidArgument, "'", std::string(name), "' is ",
                     packed.shape.to_string(), " with scales ",
                     scales.shape.to_string(), " and biases ",
                     biases.shape.to_string(),
                     "; the three planes must agree on every axis but the "
                     "last, which counts lanes on one and groups on the other");
  }
  if (order != nullptr && rank != 2) {
    return LSE_ERROR(kUnimplemented, "'", std::string(name),
                     "' is a stack; taking rows out of one would have to name "
                     "an expert as well as a row");
  }

  LSE_ASSIGN_OR(const quant::GroupAffine spec,
                quantization_->resolve_checked(name, last_dim(packed.shape),
                                               last_dim(scales.shape)));
  const std::int64_t in_features =
      last_dim(packed.shape) * 32 / spec.bits;

  Shape packed_shape = packed.shape;
  Shape group_shape = scales.shape;
  if (order != nullptr) {
    LSE_RETURN_IF_ERROR(check_rows(name, packed.shape, *order));
    const auto rows = static_cast<std::int64_t>(order->size());
    if (logical.elem_count() != static_cast<std::size_t>(rows * in_features)) {
      return LSE_ERROR(kInvalidArgument, "'", std::string(name), "' has ",
                       std::to_string(packed.shape.dim(0)), " rows of ",
                       std::to_string(in_features), " weights; taking ",
                       std::to_string(order->size()),
                       " of them cannot fill the requested ",
                       std::to_string(logical.elem_count()), " elements");
    }
    packed_shape = Shape{rows, last_dim(packed.shape)};
    group_shape = Shape{rows, last_dim(scales.shape)};
  }

  // An input-feature window becomes two different windows: one over packed
  // lanes and one over groups. Both are exact only on a boundary, which is
  // checked in require_columns before we get here.
  std::int64_t lane_first = 0, lane_count = 0;
  std::int64_t grp_first = 0, grp_count = 0;
  std::int64_t sliced_in = in_features;
  if (!window.empty()) {
    lane_first = window.first * spec.bits / 32;
    lane_count = window.count * spec.bits / 32;
    grp_first = window.first / spec.group_size;
    grp_count = window.count / spec.group_size;
    sliced_in = window.count;
    packed_shape = Shape{packed_shape.dim(0), lane_count};
    group_shape = Shape{group_shape.dim(0), grp_count};
  }

  LSE_ASSIGN_OR(Array a,
                upload(packed, packed_shape, order, lane_first, lane_count));
  LSE_ASSIGN_OR(Array s,
                upload(scales, group_shape, order, grp_first, grp_count));
  LSE_ASSIGN_OR(Array b,
                upload(biases, group_shape, order, grp_first, grp_count));

  if (order == nullptr && window.empty()) {
    record_original_upload(name, a);
    record_original_upload(scales.name, s);
    record_original_upload(biases.name, b);
  }
  auto planes = std::make_shared<graph::QuantPlanes>();
  planes->scales = s.node();
  planes->biases = b.node();
  planes->bits = spec.bits;
  planes->group_size = spec.group_size;
  planes->in_features = sliced_in;
  if (order == nullptr && window.empty() &&
      packs_q8_matrix(spec, rank, scales.dtype, packed_shape.dim(0), sliced_in)) {
    auto *scheduler = graph::default_scheduler();
    if (scheduler == nullptr)
      return LSE_ERROR(kInternal, "no backend for packed Q8 weights");
    auto &devices = scheduler->devices();
    const auto member = graph::preferred_member();
    auto &be =
        member < devices.size() ? devices.device(member) : scheduler->backend();
    const auto stream =
        member < devices.size()
            ? devices.stream_for(member).value_or(backend::kDefaultStream)
            : backend::kDefaultStream;
    try {
      if (dispatch::q8_packed_weight_device(be.device_info())) {
        constexpr std::size_t alignment = 4096;
        const auto align = [](std::size_t bytes) {
          return (bytes + alignment - 1) & ~(alignment - 1);
        };
        const auto columns = static_cast<std::uint32_t>(packed_shape.dim(0));
        const auto features = static_cast<std::uint32_t>(sliced_in);
        const auto padded = (columns + 15u) / 16u * 16u;
        const std::array<Shape, 3> shapes{Shape{padded, features / 4u},
                                          Shape{padded, features / 64u},
                                          Shape{padded, features / 64u}};
        const std::array<DType, 3> dtypes{DType::kU32, DType::kBF16,
                                          DType::kBF16};
        std::array<std::size_t, 3> offsets{}, sizes{};
        std::size_t allocation_bytes = 0;
        for (std::size_t i = 0; i < shapes.size(); ++i) {
          offsets[i] = allocation_bytes;
          sizes[i] = dtype_storage_bytes(dtypes[i], shapes[i].elem_count());
          allocation_bytes += align(sizes[i]);
        }
        // packed_q8_matrix_bytes states this sum for callers without a device.
        if (allocation_bytes != packed_q8_matrix_bytes(columns, features))
          return LSE_ERROR(kInternal, "packed Q8 layout disagrees with its size rule");
        const auto free = be.sample_free_memory();
        const auto remaining = remaining_original_bytes(a.node()->buffer);
        if (quant::q8_packed_memory_admitted(
                free.ok() ? std::optional<std::size_t>(*free) : std::nullopt,
                remaining, weight_slab_bytes(be), allocation_bytes)) {
          const auto upload_matrix =
              [&]() -> Result<std::shared_ptr<graph::PackedQuantMatrix>> {
            std::array<HostScratch, 3> staged;
            const std::array<const TensorView *, 3> views{&packed, &scales,
                                                          &biases};
            std::array<std::span<const std::byte>, 3> bytes;
            for (std::size_t i = 0; i < views.size(); ++i) {
              const auto want = dtype_storage_bytes(views[i]->dtype,
                                                    views[i]->element_count());
              if (views[i]->data.size() >= want && views[i]->fd < 0) {
                bytes[i] = views[i]->data.first(want);
              } else {
                // From the file, not the mapping (see upload()).
                LSE_ASSIGN_OR(staged[i],
                              HostScratch::allocate(want, "'" + views[i]->name + "'"));
                LSE_RETURN_IF_ERROR(views[i]->fd >= 0
                    ? views[i]->read_file(staged[i].data(), want)
                    : views[i]->read_native(staged[i].data(), want));
                g_load_staged_bytes += want;
                bytes[i] = std::span<const std::byte>(staged[i].data(), want);
              }
            }
            const auto packing = std::chrono::steady_clock::now();
            LSE_ASSIGN_OR(auto matrix,
                          quant::pack_q8_matrix(columns, features, bytes[0],
                                                bytes[1], bytes[2]));
            g_load_pack_ms += std::chrono::duration<double, std::milli>(
                                  std::chrono::steady_clock::now() - packing)
                                  .count();
            const std::array<const void *, 3> data{matrix.words.data(),
                                                   matrix.scales.data(),
                                                   matrix.biases.data()};
            LSE_ASSIGN_OR(auto base,
                          be.allocate(allocation_bytes,
                                      backend::MemoryClass::kDevice, stream));
            if (!base.storage) {
              be.deallocate(base);
              return LSE_ERROR(kUnimplemented,
                               "packed Q8 allocation needs managed storage");
            }
            const std::array<graph::NodePtr, 3> sources{a.node(), s.node(),
                                                        b.node()};
            auto storage = std::make_shared<graph::PackedQuantMatrix>();
            for (std::size_t i = 0; i < data.size(); ++i) {
              auto view = base;
              view.offset += offsets[i];
              view.size_bytes = sizes[i];
              LSE_RETURN_IF_ERROR(be.upload(data[i], view, sizes[i], 0));
              g_load_pushed_bytes += sizes[i];
              storage->packed[i] =
                  Array::from_buffer(std::move(view), shapes[i], dtypes[i])
                      .node();
              storage->sources[i] = sources[i];
              storage->source_buffers[i] = sources[i]->buffer;
            }
            return storage;
          };
          auto storage = upload_matrix();
          if (storage.ok())
            planes->matrix_storage = storage.release();
        }
      }
    } catch (const std::bad_alloc &) {
      // Packing is optional; the original planes remain usable.
    }
  }
  a.node()->quant = std::move(planes);

  claimed_.emplace_back(name);
  claimed_.emplace_back(scales.name);
  claimed_.emplace_back(biases.name);
  return a;
}

Result<Array> WeightBinder::require_columns(std::string_view name,
                                            std::int64_t first,
                                            std::int64_t count, Shape shape) {
  const TensorView *v = weights_->find(name);
  if (v == nullptr) {
    return LSE_ERROR(kNotFound, "checkpoint has no tensor '", std::string(name),
                     "'");
  }
  if (first < 0 || count <= 0) {
    return LSE_ERROR(kInvalidArgument, "'", std::string(name),
                     "' asked for an empty column window");
  }
  LSE_ASSIGN_OR(const auto planes, quant_planes(name));
  if (planes[0] != nullptr) {
    if (quantization_ == nullptr) {
      return LSE_ERROR(kInvalidArgument, "'", std::string(name),
                       "' is group-affine but no quantization was declared");
    }
    LSE_ASSIGN_OR(const quant::GroupAffine spec,
                  quantization_->resolve_checked(name, last_dim(v->shape),
                                                 last_dim(planes[0]->shape)));
    const std::int64_t per_lane = 32 / spec.bits;
    if (first % spec.group_size != 0 || count % spec.group_size != 0 ||
        first % per_lane != 0 || count % per_lane != 0) {
      return LSE_ERROR(kInvalidArgument, "'", std::string(name),
                       "' is quantized in groups of ",
                       std::to_string(spec.group_size), " with ",
                       std::to_string(per_lane),
                       " weights to a lane; input features [",
                       std::to_string(first), ", ",
                       std::to_string(first + count),
                       ") would split a group or a lane, and slicing one would "
                       "mean re-quantizing rather than reading");
    }
    return bind_quantized(name, *v, *planes[0], *planes[1], nullptr, shape,
                          TensorWindow{first, count});
  }

  const std::size_t rank = v->shape.rank();
  const auto width =
      static_cast<std::size_t>(rank >= 2 ? v->shape.dim(rank - 1) : 1);
  if (static_cast<std::size_t>(first + count) > width) {
    return LSE_ERROR(kOutOfRange, "'", std::string(name), "' rows are ",
                     std::to_string(width), " wide; [", std::to_string(first),
                     ", ", std::to_string(first + count), ") runs past that");
  }
  // The invariant the row path checks and this one has to as well: upload's
  // windowed copy writes `rows` spans of `count` derived from the SOURCE, so
  // a destination shape that disagrees is a heap overrun, not a wrong answer.
  const std::size_t rows = width > 0 ? v->element_count() / width : 0;
  if (shape.elem_count() != rows * static_cast<std::size_t>(count)) {
    return LSE_ERROR(kInvalidArgument, "'", std::string(name), "' has ",
                     std::to_string(rows), " rows; a window of ",
                     std::to_string(count),
                     " columns cannot fill the requested ",
                     std::to_string(shape.elem_count()), " elements");
  }
  LSE_ASSIGN_OR(Array a, upload(*v, shape, nullptr, first, count));
  claimed_.emplace_back(name);
  return a;
}

std::vector<std::string> WeightBinder::unclaimed() const {
  std::vector<std::string> out;
  for (const auto& [name, _] : weights_->tensors()) {
    if (std::find(claimed_.begin(), claimed_.end(), name) == claimed_.end()) {
      out.push_back(name);
    }
  }
  return out;
}

Status HybridBlock::load(WeightBinder& binder, std::string_view prefix,
                         const LayerContext& ctx) {
  const std::string p(prefix);
  LSE_ASSIGN_OR(norm1_weight_, binder.require(p + spec_.norm1_name));
  LSE_ASSIGN_OR(norm2_weight_, binder.require(p + spec_.norm2_name));
  // From the context, the same place the mixers take it. Reading the scheme
  // and the device set here instead made the block's idea of how many ways it
  // was split disagree with its own mixer's, which is only invisible while
  // both happen to be derived from the same run -- a block asked to split two
  // ways with one copy of its norms indexes past what it has.
  const auto shards =
      static_cast<std::size_t>(ctx.shards > 0 ? ctx.shards : 1);
  norm1_shards_.resize(shards);
  norm2_shards_.resize(shards);
  for (std::size_t m = 0; m < shards; ++m) {
    if (m == 0) {
      norm1_shards_[0] = norm1_weight_;
      norm2_shards_[0] = norm2_weight_;
      continue;
    }
    // A norm is elementwise over the hidden width and is not cut; each member
    // keeps its own copy so the work either side of a reduce stays local.
    const graph::ScopedMember on(m);
    LSE_ASSIGN_OR(norm1_shards_[m], binder.require(p + spec_.norm1_name));
    LSE_ASSIGN_OR(norm2_shards_[m], binder.require(p + spec_.norm2_name));
  }
  LSE_RETURN_IF_ERROR(mixer_->load(binder, prefix, ctx));
  LSE_RETURN_IF_ERROR(ffn_->load(binder, prefix, ctx));
  if (mod_) LSE_RETURN_IF_ERROR(mod_->load(binder, prefix));
  return OkStatus();
}


namespace {
// LSE_TRACE_BLOCK=1: drain and checksum a stage of the block, whole and split
// paths alike, so a whole-vs-split divergence names its stage. Diagnostic
// only: the drain forces evaluation and changes phase structure.
void trace_stage(const char* path, const char* stage, const Array& a) {
  static const bool on = std::getenv("LSE_TRACE_BLOCK") != nullptr;
  if (!on || !a.valid()) return;
  Array& mut = const_cast<Array&>(a);
  if (!mut.eval().ok()) return;
  const std::size_t n = mut.shape().elem_count();
  // Per-token checksums, so a divergence names its position instead of
  // cancelling into a whole-tensor sum. Assumes [1, tokens, hidden].
  const std::size_t rank = mut.shape().rank();
  const std::size_t hid = rank >= 1
      ? static_cast<std::size_t>(mut.shape().dim(
            static_cast<std::int64_t>(rank) - 1))
      : n;
  const std::size_t toks = hid != 0 ? n / hid : 1;
  std::fprintf(stderr, "[block] %s %-8s", path, stage);
  for (std::size_t t = 0; t < toks; ++t) {
    double chk = 0.0;
    for (std::size_t i = t * hid; i < (t + 1) * hid; ++i) {
      const double v = graph::interpreter::load_element(*mut.node(), i);
      chk += v * static_cast<double>((i % 97) + 1);
    }
    std::fprintf(stderr, " t%zu=%.9e", t, chk);
  }
  std::fprintf(stderr, "\n");
}
}  // namespace

Result<Array> HybridBlock::forward(const Array& x, MixerState* state,
                                   Array* aux_loss, const LayerContext& ctx) {
  if (ctx.config == nullptr) {
    return LSE_ERROR(kInvalidArgument, "HybridBlock::forward needs a config");
  }
  const float eps = ctx.config->rms_eps;
  auto norm = [&](const Array& v, const Array& w) {
    return zero_centered_norm_ ? ops::rms_norm_zero_centered(v, w, eps)
                               : ops::rms_norm(v, w, eps);
  };

  Array normed1 = norm(x, norm1_weight_);
  trace_stage("whole", "norm1", normed1);
  LSE_ASSIGN_OR(Array mixed, mixer_->forward(normed1, state, ctx));
  trace_stage("whole", "mix", mixed);
  Array h = x + mixed;

  Array h2 = norm(h, norm2_weight_);
  trace_stage("whole", "norm2", h2);
  LSE_ASSIGN_OR(Array ff, ffn_->forward(h2, aux_loss, ctx));
  trace_stage("whole", "ffn", ff);
  if (mod_) {
    LSE_ASSIGN_OR(ff, mod_->gate_all(h2, ff));
  }
  LSE_ASSIGN_OR(Array dense, ffn_->ungated(h2));
  if (dense.valid()) ff = ff + dense;
  return h + ff;
}

Result<std::vector<Array>> HybridBlock::forward_shards(
    const std::vector<Array>& xs, MixerState* state, Array* aux_loss,
    const LayerContext& ctx) {
  if (ctx.config == nullptr) {
    return LSE_ERROR(kInvalidArgument, "HybridBlock::forward needs a config");
  }
  const std::size_t n = xs.size();
  const float eps = ctx.config->rms_eps;
  auto norm = [&](const Array& v, const Array& w) {
    return zero_centered_norm_ ? ops::rms_norm_zero_centered(v, w, eps)
                               : ops::rms_norm(v, w, eps);
  };
  // Every member adds every partial, so each ends up with the same total and
  // the work after it is local. What crosses is one partial per member per
  // reduce, in opposite directions, instead of the activation on the way in
  // AND the sum on the way out.
  auto reduce = [&](const std::vector<Array>& parts) -> std::vector<Array> {
    // An op that did not shard hands back ONE value, and that value is already
    // the whole answer: every member takes it as it is. Summing it per member
    // would add it to itself, and indexing parts[m] past what it returned is
    // how a block that shards its feed-forward but not its mixer -- or the
    // other way round -- reads off the end of the vector. Nothing today
    // returns zero partials, but the contract does not forbid it, and the
    // deref below must not be the thing that finds out.
    std::vector<Array> out(n);
    if (parts.empty()) return out;
    if (parts.size() < n) {
      for (std::size_t m = 0; m < n; ++m) out[m] = parts.front();
      return out;
    }
    for (std::size_t m = 0; m < n; ++m) {
      const graph::ScopedMember on(m);
      // The sum runs in f32 whatever the partials are stored as. Each partial
      // already paid one rounding when its half-width matmul left its f32
      // accumulator; adding them at storage precision pays another per member
      // and the block lands ~2e-4 off the unsplit answer — measured by
      // a_split_block_rebuilds_what_the_whole_one_computes, and compounded
      // over 64 blocks' residual streams it flips greedy tokens. Summing in
      // f32 keeps the join's error at the one rounding the cast back costs.
      const DType kept = parts[m].dtype();
      Array acc = kept == DType::kF32 ? parts[m]
                                      : graph::cast(parts[m], DType::kF32);
      for (std::size_t j = 0; j < parts.size(); ++j) {
        if (j != m) {
          const Array& p = parts[j];
          acc = graph::add(acc, p.dtype() == DType::kF32
                                    ? p
                                    : graph::cast(p, DType::kF32));
        }
      }
      out[m] = kept == DType::kF32 ? acc : graph::cast(acc, kept);
    }
    return out;
  };

  std::vector<Array> normed(n);
  for (std::size_t m = 0; m < n; ++m) {
    const graph::ScopedMember on(m);
    normed[m] = norm(xs[m], norm1_shards_[m]);
  }
  trace_stage("split", "norm1", normed[0]);
  LSE_ASSIGN_OR(std::vector<Array> mixed,
                mixer_->forward_shards(normed, state, ctx));
  const std::vector<Array> mix_sum = reduce(mixed);
  trace_stage("split", "mix", mix_sum[0]);

  std::vector<Array> h(n), h2(n);
  for (std::size_t m = 0; m < n; ++m) {
    const graph::ScopedMember on(m);
    h[m] = xs[m] + mix_sum[m];
    h2[m] = norm(h[m], norm2_shards_[m]);
  }

  trace_stage("split", "norm2", h2[0]);
  LSE_ASSIGN_OR(std::vector<Array> ff, ffn_->forward_shards(h2, aux_loss, ctx));
  const std::vector<Array> ff_sum = reduce(ff);
  trace_stage("split", "ffn", ff_sum[0]);

  // The whole path's tail, per member: the MoD gate scales the routed output,
  // and whatever the FFN keeps outside the gate (a shared expert) is added
  // after it. Each member holds the full h2 and the full ff_sum, so both are
  // local work — dropping them here silently ran a split model without a
  // whole branch of its feed-forward.
  std::vector<Array> out(n);
  for (std::size_t m = 0; m < n; ++m) {
    const graph::ScopedMember on(m);
    Array ff_m = ff_sum[m];
    if (mod_) {
      LSE_ASSIGN_OR(ff_m, mod_->gate_all(h2[m], ff_m));
    }
    LSE_ASSIGN_OR(Array dense, ffn_->ungated(h2[m]));
    if (dense.valid()) ff_m = ff_m + dense;
    out[m] = h[m] + ff_m;
  }
  return out;
}

}  // namespace lse::model
