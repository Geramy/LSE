#include "lse/ir/pass/pass.hpp"

#include <mutex>

#include "lse/ir/pass/cse.hpp"
#include "lse/ir/pass/dce.hpp"
#include "lse/ir/pass/factor_hoist.hpp"
#include "lse/ir/pass/lds_fold.hpp"
#include "lse/ir/verify.hpp"

namespace lse::ir {

PassPipeline& PassPipeline::add(std::unique_ptr<Pass> pass) {
  passes_.push_back(std::move(pass));
  return *this;
}

Status PassPipeline::run(Body& body, std::vector<PassStat>* stats) const {
  // Before any pass runs, so a body that arrived malformed is reported against
  // whoever wrote it. Verifying only after each pass named the first pass in
  // the pipeline as the culprit for a defect it inherited, which sent a day
  // into reading that pass instead of the kernel that emitted the body.
  if (Status s = verify(body); !s.ok()) {
    return LSE_ERROR(kInternal,
                     "the emitted kernel body is malformed before any pass "
                     "ran: ", s.message());
  }
  for (const std::unique_ptr<Pass>& p : passes_) {
    const std::size_t fired = p->run(body);
    if (stats != nullptr) stats->push_back(PassStat{p->name(), fired});
    if (Status s = verify(body); !s.ok()) {
      return LSE_ERROR(kInternal, "pass '", std::string(p->name()),
                       "' left the body malformed: ", s.message());
    }
  }
  return OkStatus();
}

const PassPipeline& default_pipeline() {
  // Order matters and each step earns the next: CSE gives the two stages one
  // value for the same number, which is what lets the LDS fold see one fill;
  // the fold empties a loop and orphans an allocation, which is what gives DCE
  // something to delete.
  static const PassPipeline kPipeline = [] {
    PassPipeline p;
    p.add(make_cse());
    // After CSE so the invariant factor is ONE value at every accumulate --
    // two spellings of the same scalar would read as two factors and the
    // uniformity test would refuse the hoist. Before DCE so the multiply it
    // strips out of the loop can be collected if nothing else wanted it.
    p.add(make_factor_hoist());
    p.add(make_lds_fold());
    p.add(make_dce());
    return p;
  }();
  return kPipeline;
}

namespace {

std::mutex& totals_mutex() {
  static std::mutex mu;
  return mu;
}

std::vector<PassStat>& totals() {
  static std::vector<PassStat> t;
  return t;
}

}  // namespace

void record_pass_totals(const std::vector<PassStat>& stats) {
  const std::lock_guard<std::mutex> lock(totals_mutex());
  std::vector<PassStat>& t = totals();
  for (const PassStat& s : stats) {
    bool found = false;
    for (PassStat& have : t) {
      if (have.name == s.name) {
        have.fired += s.fired;
        found = true;
      }
    }
    if (!found) t.push_back(s);
  }
}

std::vector<PassStat> pass_totals() {
  const std::lock_guard<std::mutex> lock(totals_mutex());
  return totals();
}

}  // namespace lse::ir
