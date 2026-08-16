// What a pool member can actually do — measured, not declared.
//
// A capability says a device *has* a matrix core; a profile says what that
// device *achieved* on it, for each operand form, on this machine, this run.
// Placement, sharding and scheme selection are decisions about time, and time
// is the one thing a capability bit cannot supply.
//
// Every number carries its Provenance, and the four values are not
// interchangeable: kMeasured is this hardware, kDeclared is a table or a
// capability that nobody timed, kUnsupported is "this device does not have it
// at all", and kUnknown is "nothing here knows". A probe that cannot reach a
// number leaves it kUnknown — it never fills in a plausible one, because a
// fabricated number is indistinguishable from a measured one at the point it
// is used and the cost model would act on it.
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "lse/core/dtype.hpp"
#include "lse/core/enum_names.hpp"
#include "lse/core/status.hpp"
#include "lse/math.hpp"

namespace lse::probe {

#define LSE_PROVENANCE_LIST(X)                    \
  X(kUnknown, "unknown")                          \
  X(kDeclared, "declared")                        \
  X(kMeasured, "measured")                        \
  X(kUnsupported, "unsupported")

LSE_DECLARE_ENUM(Provenance, std::uint8_t, LSE_PROVENANCE_LIST)

// A quantity and where it came from. Arithmetic on two of these takes the
// weaker provenance, so a derived number can never claim to be better founded
// than its worst input.
struct Measured {
  double value = 0.0;
  Provenance provenance = Provenance::kUnknown;

  [[nodiscard]] bool known() const noexcept {
    return provenance == Provenance::kMeasured ||
           provenance == Provenance::kDeclared;
  }
  [[nodiscard]] bool positive() const noexcept { return known() && value > 0.0; }

  static Measured measured(double v) noexcept {
    return {v, Provenance::kMeasured};
  }
  static Measured declared(double v) noexcept {
    return {v, Provenance::kDeclared};
  }
  static Measured unknown() noexcept { return {}; }
  static Measured unsupported() noexcept {
    return {0.0, Provenance::kUnsupported};
  }
};

// The weaker of two provenances, in the order unknown < unsupported < declared
// < measured. Combining a measured term with an unknown one yields unknown.
[[nodiscard]] Provenance weaker(Provenance a, Provenance b) noexcept;

// A device, addressed the way the whole engine addresses one: backend-qualified,
// never a bare ordinal. "hrx:0", "cpu:0".
struct DeviceId {
  std::string backend;
  int ordinal = 0;

  [[nodiscard]] std::string str() const;
  friend auto operator<=>(const DeviceId&, const DeviceId&) = default;
  friend bool operator==(const DeviceId&, const DeviceId&) = default;
};

[[nodiscard]] Result<DeviceId> parse_device_id(std::string_view text);

// How this device answers for one row of lse::math's matrix-core table.
//
// The four states are different failures and the cost model reads them
// differently. kAbsent is the heterogeneous-pool case the pool exists to
// handle: the device is still a member, its work for that operand simply runs
// somewhere else in the table and costs what that path costs.
#define LSE_ROW_SUPPORT_LIST(X)                                             \
  X(kMeasured, "measured")   /* device has it and the probe timed it */     \
  X(kDeclared, "declared")   /* device has it; only the table's ratio */    \
  X(kUnverified, "unverified") /* row exists, lane layout never measured */ \
  X(kAbsent, "absent")       /* this matrix core has no such operand form */

LSE_DECLARE_ENUM(RowSupport, std::uint8_t, LSE_ROW_SUPPORT_LIST)

struct MatrixRowRate {
  std::string key;                      // the dialect row this names
  math::MatrixElem acc{};
  math::MatrixElem operand{};
  int m = 0, n = 0, k_step = 0;
  int relative = 0;                     // the table's own per-CU ratio
  RowSupport support = RowSupport::kAbsent;
  Measured flops;                       // achieved 2*M*N*K/s, when timed
};

// How one operand format actually executes here.
//
// `native` false is not a disqualification: an fp8 checkpoint on a device with
// no fp8 matrix core still runs, by dequantizing inside the kernel at the
// register boundary and multiplying in `executed_as`. Nothing converts the
// tensor in memory or on the wire — the storage format is the checkpoint's
// either way — so the only thing that differs between two pool members is the
// rate below, which is exactly what the cost model needs to split work.
struct ComputePath {
  math::MatrixElem operand{};
  math::MatrixElem executed_as{};
  std::string row_key;                  // empty when no matrix row serves it
  bool native = false;
  Measured flops;
};

struct DeviceProfile {
  DeviceId id;
  std::string arch;
  std::string name;
  std::size_t total_memory = 0;
  std::uint16_t compute_units = 0;
  bool unified_memory = false;

  // The roofline. Achieved streaming read rate out of device memory, which is
  // what decode is bound by; not the theoretical pin rate.
  Measured dram_bytes_per_s;
  // Wall time one dispatch costs when the kernel does nothing, on the
  // submission path the engine actually uses.
  Measured launch_overhead_ns;
  Measured h2d_bytes_per_s;
  Measured d2h_bytes_per_s;

  // Every row of the matrix-core table that belongs to this device's ISA
  // generation, including the ones it does not have — an omitted row and an
  // absent row read the same to a consumer, and only one of them is true.
  std::vector<MatrixRowRate> rows;
  // One entry per operand family the pool may hold, whether or not this device
  // has a native row for it.
  std::vector<ComputePath> paths;

  [[nodiscard]] const ComputePath* path_for(math::MatrixElem operand) const noexcept;
  [[nodiscard]] const MatrixRowRate* row(std::string_view key) const noexcept;
  // Time to move `bytes` out of this device's own memory. The floor under any
  // kernel that streams them.
  [[nodiscard]] Measured stream_ns(std::size_t bytes) const noexcept;
};

// What kind of path a pair of members has. Measured where it can be, declared
// from the transport's capabilities where it cannot.
#define LSE_PATH_KIND_LIST(X)                                                \
  X(kUnknown, "unknown")                                                     \
  X(kSameDevice, "same-device")     /* no transfer at all */                 \
  X(kHostStaged, "host-staged")     /* device -> host bounce -> device */    \
  X(kPeerDirect, "peer-direct")     /* PCIe / XGMI peer copy, no bounce */   \
  X(kRdmaDirect, "rdma-direct")     /* the NIC DMAs out of device memory */  \
  X(kRdmaStaged, "rdma-staged")     /* RDMA, but through a host bounce */

LSE_DECLARE_ENUM(PathKind, std::uint8_t, LSE_PATH_KIND_LIST)

struct TransferPoint {
  std::size_t bytes = 0;
  double ns = 0.0;
};

// One ordered pair. Asymmetry is real — a link can be faster one way — so
// (a,b) and (b,a) are separate records and both are measured.
//
// Latency and bandwidth are separate terms on purpose. A single GB/s figure is
// wrong for exactly the small transfers that decide whether an op is worth
// moving, which is the decision this whole file exists to inform.
struct LinkProfile {
  DeviceId src, dst;
  PathKind path = PathKind::kUnknown;

  Measured latency_ns;                 // the intercept: cost of a zero-byte move
  Measured bandwidth_bytes_per_s;      // the slope
  std::vector<TransferPoint> points;   // what the fit came from
  // Worst relative disagreement between the fit and a measured point. Large
  // means the link is not linear in this size range and the fit is a summary,
  // not a model.
  double fit_error = 0.0;

  [[nodiscard]] Measured cost_ns(std::size_t bytes) const noexcept;
};

// Fits latency + bytes/bandwidth to the points, by least squares. Fewer than
// two distinct sizes cannot separate the two terms, and the result then says
// so rather than attributing everything to one of them.
void fit_link(LinkProfile& link);

struct PoolProfile {
  // Identity of the pool this was measured on: every member's device identity
  // and the topology between them. A change here invalidates every record.
  std::string fingerprint;
  std::vector<DeviceProfile> devices;
  // Ordered pairs, row-major over `devices`. Size is devices.size()^2.
  std::vector<LinkProfile> links;

  [[nodiscard]] const DeviceProfile* device(const DeviceId& id) const noexcept;
  [[nodiscard]] std::size_t index_of(const DeviceId& id) const noexcept;
  [[nodiscard]] const LinkProfile* link(const DeviceId& src,
                                        const DeviceId& dst) const noexcept;
  [[nodiscard]] bool complete() const noexcept;
  [[nodiscard]] std::string describe() const;
};

// Sizes the links are measured at. Spread over four decades so the fit
// separates the intercept from the slope instead of trading one against the
// other, and capped so the whole pass stays well under a second.
[[nodiscard]] std::span<const std::size_t> default_transfer_sizes() noexcept;

// The operand families the pool profile carries a path for. This is the table's
// own operand set, not a second list.
[[nodiscard]] std::span<const math::MatrixElem> profiled_operands() noexcept;

// The operand form a stored dtype feeds. kF32 has no matrix operand form on any
// target in the table and takes the f16 one, which is the one narrowing the
// engine allows; every other format feeds its own.
[[nodiscard]] math::MatrixElem operand_of_storage(DType storage) noexcept;

[[nodiscard]] std::string_view to_string(math::MatrixElem e) noexcept;

}  // namespace lse::probe
