#include "lse/place/placement.hpp"

#include <string>

namespace lse::place {

namespace {

std::size_t index_of(const Devices& devices,
                     const probe::DeviceId& id) noexcept {
  const std::span<const Member> members = devices.members();
  for (std::size_t i = 0; i < members.size(); ++i) {
    if (members[i].id == id) return i;
  }
  return members.size();
}

}  // namespace

Planner::Planner(const Devices& devices,
                 const probe::PoolProfile& pool) noexcept
    : devices_(&devices), model_(pool) {}

Result<Assignment> Planner::place(const probe::Work& work, std::size_t home,
                                 Transfer moved, std::uint32_t depth) const {
  const std::span<const Member> members = devices_->members();
  if (home >= members.size()) {
    return LSE_ERROR(kOutOfRange, "this set holds ",
                     std::to_string(members.size()),
                     " devices; there is no member ", std::to_string(home));
  }
  const probe::DeviceId& home_id = members[home].id;

  Assignment out;
  out.member = home;
  out.device = members[home].index;
  out.capacity = model_.capacity_for(work, home_id);
  if (members.size() == 1) {
    // Not a shortcut: with nowhere else to run, asking the cost model would
    // spend a measurement on a decision that is already made.
    out.reason = "the set holds one device";
    return out;
  }

  out.reason = "nothing priced beat staying where the operands are";
  double best = 0.0;
  for (std::size_t i = 0; i < members.size(); ++i) {
    if (i == home) continue;
    const probe::CostModel::OffloadDecision d = model_.should_offload(
        work, home_id, members[i].id, moved.out, moved.back, depth);
    if (!d.relocate) continue;
    if (!d.moved.known() || d.moved.items_per_s <= best) continue;
    best = d.moved.items_per_s;
    out.member = i;
    out.device = members[i].index;
    out.capacity = d.candidate_capacity;
    out.relocates = true;
    out.reason = d.reason;
  }
  return out;
}

Result<Planner::Division> Planner::divide(const probe::Work& work,
                                          std::size_t home,
                                          std::uint32_t depth) const {
  const std::span<const Member> members = devices_->members();
  if (home >= members.size()) {
    return LSE_ERROR(kOutOfRange, "this set holds ",
                     std::to_string(members.size()),
                     " devices; there is no member ", std::to_string(home));
  }
  std::vector<probe::DeviceId> ids;
  ids.reserve(members.size());
  for (const Member& m : members) ids.push_back(m.id);

  const probe::CostModel::Split split =
      model_.split(work, ids, members[home].id, depth);

  Division out;
  out.fit = split.fit;
  out.provenance = split.provenance;
  out.reason = split.reason;
  for (const probe::CostModel::Share& s : split.shares) {
    const std::size_t i = index_of(*devices_, s.device);
    if (i >= members.size()) {
      // The cost model was given this set's ids, so a share naming something
      // else means the profile is not this pool's — report it rather than drop
      // work on the floor.
      return LSE_ERROR(kInternal, "the split named ", s.device.str(),
                       ", which this set does not hold");
    }
    out.portions.push_back(Portion{i, s.fraction, s.per_item, s.capacity});
  }
  return out;
}

}  // namespace lse::place
