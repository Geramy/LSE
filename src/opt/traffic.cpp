#include "lse/opt/traffic.hpp"

#include <sstream>

namespace lse::opt {

namespace {

std::size_t index_of(OperandClass what) {
  return static_cast<std::size_t>(what);
}

}  // namespace

void TrafficModel::add_read(OperandClass what, std::uint64_t bytes) {
  read[index_of(what)] += bytes;
  stated = true;
}

void TrafficModel::add_written(OperandClass what, std::uint64_t bytes) {
  written[index_of(what)] += bytes;
  stated = true;
}

std::uint64_t TrafficModel::bytes_read(OperandClass what) const {
  return read[index_of(what)];
}

std::uint64_t TrafficModel::bytes_read() const {
  std::uint64_t n = 0;
  for (std::uint64_t b : read) n += b;
  return n;
}

std::uint64_t TrafficModel::bytes_written() const {
  std::uint64_t n = 0;
  for (std::uint64_t b : written) n += b;
  return n;
}

std::uint64_t TrafficModel::total_bytes() const {
  return bytes_read() + bytes_written();
}

double TrafficModel::bytes_per_work() const {
  if (work == 0) return 0.0;
  return static_cast<double>(total_bytes()) / static_cast<double>(work);
}

double TrafficModel::read_share(OperandClass what) const {
  const std::uint64_t all = bytes_read();
  if (all == 0) return 0.0;
  return static_cast<double>(bytes_read(what)) / static_cast<double>(all);
}

TrafficModel& TrafficModel::operator+=(const TrafficModel& other) {
  for (std::size_t i = 0; i < kOperandClasses; ++i) {
    read[i] += other.read[i];
    written[i] += other.written[i];
  }
  work += other.work;
  if (workgroups == 0) workgroups = other.workgroups;
  if (workgroup_threads == 0) workgroup_threads = other.workgroup_threads;
  stated = stated && other.stated;
  return *this;
}

std::string TrafficModel::describe() const {
  std::ostringstream os;
  if (!stated) return "traffic: unstated\n";
  os << "traffic per workgroup\n";
  for (std::size_t i = 0; i < kOperandClasses; ++i) {
    const auto what = static_cast<OperandClass>(i);
    if (read[i] == 0 && written[i] == 0) continue;
    os << "  " << to_string(what) << ": " << read[i] << " B read";
    if (written[i] != 0) os << ", " << written[i] << " B written";
    if (read[i] != 0) {
      os << " (" << static_cast<int>(read_share(what) * 100.0 + 0.5)
         << "% of reads)";
    }
    os << '\n';
  }
  os << "  total: " << total_bytes() << " B";
  if (work != 0) os << ", " << bytes_per_work() << " B per mac";
  if (workgroups != 0) os << ", over " << workgroups << " workgroups";
  os << '\n';
  return os.str();
}

TrafficModel contraction_traffic(std::uint64_t rows, std::uint64_t cols,
                                 std::uint64_t depth,
                                 std::uint32_t weight_bits,
                                 std::uint32_t activation_bytes,
                                 std::uint64_t scale_bytes,
                                 std::uint64_t output_bytes) {
  TrafficModel m;
  if (rows == 0 || cols == 0 || depth == 0) return m;
  m.add_read(OperandClass::kWeight, cols * depth * weight_bits / 8);
  m.add_read(OperandClass::kActivation, rows * depth * activation_bytes);
  if (scale_bytes != 0) m.add_read(OperandClass::kScale, scale_bytes);
  if (output_bytes != 0) m.add_written(OperandClass::kOutput, output_bytes);
  m.work = rows * cols * depth;
  return m;
}

EmittedTraffic emitted_traffic(const backend::KernelCensus& c,
                               std::uint32_t workgroup_threads) {
  EmittedTraffic out;
  if (!c.any() || workgroup_threads == 0) return out;
  out.known = true;
  const bool loops = c.global_loads.loops() || c.global_stores.loops() ||
                     c.shared_loads.loops() || c.shared_stores.loops();
  out.exact = !loops;
  const auto scale = [&](const backend::AccessCensus& a) {
    return a.bytes() * workgroup_threads;
  };
  const auto floor_of = [&](const backend::AccessCensus& a) {
    return a.straight_line_bytes() * workgroup_threads;
  };
  out.global_read = scale(c.global_loads);
  out.global_written = scale(c.global_stores);
  out.shared_read = scale(c.shared_loads);
  out.shared_written = scale(c.shared_stores);
  out.global_floor = floor_of(c.global_loads) + floor_of(c.global_stores);
  if (c.multiply_accumulates.known()) {
    out.work = static_cast<std::uint64_t>(c.multiply_accumulates.value) *
               workgroup_threads;
  }
  return out;
}

double TrafficCheck::ratio() const {
  if (intended_bytes == 0) return 0.0;
  return static_cast<double>(emitted_bytes) /
         static_cast<double>(intended_bytes);
}

double TrafficCheck::intensity_ratio() const {
  if (intended_bytes_per_work == 0.0 || emitted_bytes_per_work == 0.0) {
    return 0.0;
  }
  return emitted_bytes_per_work / intended_bytes_per_work;
}

std::string TrafficCheck::describe() const {
  std::ostringstream os;
  os << "traffic " << to_string(verdict) << ": intended " << intended_bytes
     << " B/workgroup, emitted " << emitted_bytes << " B/workgroup";
  if (!emitted_exact) {
    os << " per traversal, floor " << emitted_floor << " B/workgroup";
  }
  if (intended_bytes != 0) os << ", ratio " << ratio();
  if (intensity_ratio() != 0.0) {
    os << "; " << intended_bytes_per_work << " vs " << emitted_bytes_per_work
       << " B/mac, ratio " << intensity_ratio();
  }
  os << '\n';
  return os.str();
}

TrafficCheck check_traffic(const TrafficModel& intended,
                           const backend::KernelCensus& emitted,
                           std::uint32_t workgroup_threads) {
  TrafficCheck out;
  const EmittedTraffic actual = emitted_traffic(emitted, workgroup_threads);
  if (!intended.stated || !actual.known) return out;
  out.intended_bytes = intended.total_bytes();
  out.emitted_bytes = actual.global_bytes();
  out.emitted_floor = actual.global_floor;
  out.emitted_exact = actual.exact;
  out.intended_bytes_per_work = intended.bytes_per_work();
  out.emitted_bytes_per_work = actual.bytes_per_work();
  if (out.intended_bytes == 0) return out;

  // Intensity first, because it is the one comparison a loop cannot distort:
  // both sides are bytes over the work done for them, counted over the same
  // body. Only the upward direction is decidable — the emitted figure bounds
  // the true intensity from below.
  const double intensity = out.intensity_ratio();
  if (intensity >= kTrafficDisagreementFactor) {
    out.verdict = TrafficVerdict::kEmitsMore;
    return out;
  }

  const auto want = static_cast<double>(out.intended_bytes);
  // The floor cannot be repeated away, so it settles kEmitsMore on a looping
  // body as firmly as on a straight one.
  if (static_cast<double>(out.emitted_floor) >=
      kTrafficDisagreementFactor * want) {
    out.verdict = TrafficVerdict::kEmitsMore;
    return out;
  }
  const double r = out.ratio();
  if (r >= kTrafficDisagreementFactor) {
    // One traversal of the body already moves more than the whole workgroup was
    // supposed to, and the loops can only add to that.
    out.verdict = TrafficVerdict::kEmitsMore;
    return out;
  }
  // Under the intent decides nothing on a looping body: a K loop counted once
  // looks like a kernel that reads almost nothing, which is what every
  // correctly priced contraction here looks like.
  if (!out.emitted_exact) {
    out.verdict = TrafficVerdict::kUndecided;
    return out;
  }
  if (r != 0.0 && 1.0 / r >= kTrafficDisagreementFactor) {
    out.verdict = TrafficVerdict::kEmitsLess;
  } else {
    out.verdict = TrafficVerdict::kAgrees;
  }
  return out;
}

}  // namespace lse::opt
