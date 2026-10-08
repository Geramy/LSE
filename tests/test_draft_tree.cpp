// Draft trees: construction, layout, the acceptance walk, and that tree
// speculation leaves the target's output distribution exactly as it was.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <map>
#include <vector>

#include "harness.hpp"
#include "lse/runtime/draft_tree.hpp"
#include "lse/runtime/sampler.hpp"

using namespace lse::runtime;

namespace {

std::uint64_t mix(std::uint64_t x) {
  x += 0x9e3779b97f4a7c15ull;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
  return x ^ (x >> 31);
}
double unit(std::uint64_t x) { return static_cast<double>(mix(x) >> 11) * 0x1.0p-53; }

// A lattice with distinct ids per position drawn from `vocab`, scores from `seed`.
DraftLattice make_lattice(std::uint32_t positions, std::uint32_t top, std::uint32_t vocab,
                          std::uint64_t seed, double spread = 4.0) {
  DraftLattice l;
  l.positions = positions;
  l.top = top;
  for (std::uint32_t p = 0; p < positions; ++p) {
    std::vector<std::uint32_t> pool(vocab);
    for (std::uint32_t i = 0; i < vocab; ++i) pool[i] = i;
    for (std::uint32_t i = 0; i < vocab; ++i)
      std::swap(pool[i], pool[i + static_cast<std::uint32_t>(unit(seed * 131 + p * 17 + i) * (vocab - i))]);
    for (std::uint32_t k = 0; k < top; ++k) l.ids.push_back(pool[k]);
    for (std::uint32_t pred = 0; pred < top; ++pred)
      for (std::uint32_t k = 0; k < top; ++k)
        l.scores.push_back(static_cast<float>(
            spread * unit(seed * 7919 + (p * top + (p == 0 ? 0 : pred)) * top + k)));
  }
  return l;
}

// Every node of the full lattice tree with its path value.
struct FullNode {
  int parent;
  double value;
};
void enumerate(const DraftLattice& l, double t, std::vector<FullNode>& out) {
  std::vector<double> q(l.top);
  std::function<void(int, std::uint32_t, std::uint32_t, double)> rec =
      [&](int parent, std::uint32_t position, std::uint32_t pred, double value) {
        if (position >= l.positions) return;
        l.conditional(position, pred, t, q);
        const std::vector<double> here = q;
        for (std::uint32_t k = 0; k < l.top; ++k) {
          const int me = static_cast<int>(out.size());
          out.push_back({parent, value * here[k]});
          rec(me, position + 1, k, value * here[k]);
        }
      };
  rec(-1, 0, 0, 1.0);
}

}  // namespace

LSE_TEST(expansion_pops_in_value_order_with_parents_first) {
  const DraftLattice l = make_lattice(7, 16, 64, 3);
  auto e = expand_tree(l, 0.6, 64);
  LSE_EXPECT_OK(e.status());
  if (!e.ok()) return;
  LSE_EXPECT_EQ(e->nodes.size(), std::size_t{64});
  for (std::size_t i = 0; i < e->nodes.size(); ++i) {
    const auto& n = e->nodes[i];
    LSE_EXPECT(n.parent < static_cast<std::int32_t>(i));
    if (i > 0) LSE_EXPECT(n.value <= e->nodes[i - 1].value);
    if (n.parent >= 0) {
      LSE_EXPECT(n.position == e->nodes[static_cast<std::size_t>(n.parent)].position + 1);
      LSE_EXPECT_NEAR(n.value, e->nodes[static_cast<std::size_t>(n.parent)].value * n.probability, 1e-12);
    } else {
      LSE_EXPECT(n.position == 0);
    }
    LSE_EXPECT_NEAR(e->value_prefix[i + 1] - e->value_prefix[i], n.value, 1e-12);
  }
}

// DDTree's optimality (Proposition 1): the first B pops have the largest
// total path value of any B-node subtree. Brute force over every subtree.
LSE_TEST(first_pops_are_the_best_tree_of_their_size) {
  for (std::uint64_t seed = 1; seed <= 6; ++seed) {
    const DraftLattice l = make_lattice(3, 3, 9, seed, 3.0);
    std::vector<FullNode> all;
    enumerate(l, 1.0, all);
    LSE_EXPECT_EQ(all.size(), std::size_t{39});
    auto e = expand_tree(l, 1.0, 39);
    LSE_EXPECT_OK(e.status());
    if (!e.ok()) return;
    for (std::uint32_t b = 1; b <= 5; ++b) {
      double best = 0;
      std::vector<int> pick;
      std::function<void(std::size_t, double)> rec = [&](std::size_t from, double sum) {
        if (pick.size() == b) {
          best = std::max(best, sum);
          return;
        }
        for (std::size_t i = from; i < all.size(); ++i) {
          const int p = all[i].parent;
          // Parents precede children in `all`, so closure is checked on entry.
          if (p >= 0 && std::find(pick.begin(), pick.end(), p) == pick.end()) continue;
          pick.push_back(static_cast<int>(i));
          rec(i + 1, sum + all[i].value);
          pick.pop_back();
        }
      };
      rec(0, 0.0);
      LSE_EXPECT_NEAR(e->value_prefix[b], best, 1e-12);
    }
  }
}

LSE_TEST(layout_is_preorder_with_the_top_path_first) {
  const DraftLattice l = make_lattice(7, 16, 300, 11, 6.0);
  auto e = expand_tree(l, 0.6, 31);
  LSE_EXPECT_OK(e.status());
  if (!e.ok()) return;
  for (std::uint32_t nodes : {0u, 1u, 7u, 15u, 31u}) {
    auto tree = layout_tree(l, 42, *e, nodes);
    LSE_EXPECT_OK(tree.status());
    if (!tree.ok()) continue;
    LSE_EXPECT_EQ(tree->rows(), std::size_t{nodes} + 1);
    LSE_EXPECT_OK(tree->validate());
    LSE_EXPECT_EQ(tree->tokens[0], 42u);
    // The top path follows the best conditional at each position.
    std::uint32_t pred = 0;
    std::vector<double> q(l.top);
    for (std::uint32_t r = 1; r < tree->main_rows; ++r) {
      l.conditional(r - 1, pred, 0.6, q);
      const auto best = static_cast<std::uint32_t>(std::max_element(q.begin(), q.end()) - q.begin());
      LSE_EXPECT_EQ(tree->tokens[r], l.ids[(r - 1) * l.top + best]);
      pred = best;
    }
    for (std::uint32_t r = 0; r < tree->rows(); ++r) {
      LSE_EXPECT(tree->sees(r, 0));
      LSE_EXPECT(tree->sees(r, r));
      for (std::uint32_t up = 0; up <= tree->depth[r]; ++up) {
        const std::int32_t a = tree->ancestor(r, up);
        LSE_EXPECT(a >= 0 && tree->depth[static_cast<std::size_t>(a)] == tree->depth[r] - up);
      }
      LSE_EXPECT_EQ(tree->ancestor(r, tree->depth[r] + 1), -1);
    }
  }
}

LSE_TEST(chain_tree_is_a_single_path) {
  const std::uint32_t tokens[] = {5, 6, 7};
  const DraftTree t = chain_tree(4, tokens);
  LSE_EXPECT_OK(t.validate());
  LSE_EXPECT_EQ(t.main_rows, 4u);
  LSE_EXPECT_EQ(t.child_with(1, 6), 2);
  LSE_EXPECT_EQ(t.child_with(1, 7), -1);
}

LSE_TEST(walk_follows_matching_children_and_stops_at_a_miss) {
  const DraftLattice l = make_lattice(4, 3, 20, 5, 1.0);
  auto e = expand_tree(l, 1.0, 12);
  LSE_EXPECT_OK(e.status());
  if (!e.ok()) return;
  auto tree = layout_tree(l, 1, *e, 12);
  LSE_EXPECT_OK(tree.status());
  if (!tree.ok()) return;
  // Target agrees with a second child at the root, then with the first child,
  // then answers something no child holds.
  const DraftTree& t = *tree;
  if (t.children[0].size() < 2) return;
  const std::uint32_t second = t.children[0][1];
  std::vector<std::uint32_t> plan;
  plan.push_back(t.tokens[second]);
  std::uint32_t row = second;
  if (!t.children[row].empty()) {
    plan.push_back(t.tokens[t.children[row][0]]);
    row = t.children[row][0];
  }
  plan.push_back(999);
  std::size_t asked = 0;
  auto w = walk_tree(t, [&](std::uint32_t) -> lse::Result<std::pair<std::uint32_t, bool>> {
    return std::pair(plan[asked++], true);
  });
  LSE_EXPECT_OK(w.status());
  if (!w.ok()) return;
  LSE_EXPECT_EQ(w->answers.size(), plan.size());
  LSE_EXPECT_EQ(w->path.size(), plan.size());
  LSE_EXPECT_EQ(w->path.back(), row);
  LSE_EXPECT(!w->stopped);
  // A stop from the caller ends the walk with the answer emitted.
  asked = 0;
  auto stop = walk_tree(t, [&](std::uint32_t) -> lse::Result<std::pair<std::uint32_t, bool>> {
    return std::pair(plan[asked++], false);
  });
  LSE_EXPECT_OK(stop.status());
  if (stop.ok()) {
    LSE_EXPECT(stop->stopped);
    LSE_EXPECT_EQ(stop->path.size(), std::size_t{1});
  }
}

// Tree speculation emits exactly the target's distribution. A synthetic target
// over five tokens whose next-token distribution depends on the whole context,
// a draft lattice that depends on the context too (and is often wrong), and
// trees of several budgets: three generated tokens' joint frequencies over
// 300k runs against the exact product of the target's conditionals.
LSE_TEST(tree_speculation_samples_the_target_distribution) {
  constexpr std::uint32_t V = 5;
  const auto target = [](const std::vector<std::uint32_t>& ctx) {
    std::uint64_t h = 1469598103934665603ull;
    for (const auto x : ctx) h = mix(h ^ (x + 1));
    DiscreteDistribution d;
    double total = 0;
    for (std::uint32_t v = 0; v < V; ++v) {
      d.ids.push_back(v);
      const double w = std::exp(3.0 * unit(h + v));
      d.probabilities.push_back(w);
      total += w;
    }
    for (auto& p : d.probabilities) p /= total;
    return d;
  };
  for (std::uint32_t budget : {1u, 3u, 8u, 20u}) {
    SpeculativeSampler sampler(1234 + budget);
    std::map<std::vector<std::uint32_t>, double> counts;
    constexpr int kRuns = 300000;
    for (int run = 0; run < kRuns; ++run) {
      std::vector<std::uint32_t> out;
      auto first = sampler.sample_target(target(out));
      if (!first.ok()) { LSE_EXPECT_OK(first.status()); return; }
      out.push_back(*first);
      while (out.size() < 3) {
        std::uint64_t h = 77;
        for (const auto x : out) h = mix(h ^ (x + 3));
        const DraftLattice l = make_lattice(3, 3, V, h % 1000003, 3.0);
        auto e = expand_tree(l, 1.0, budget);
        if (!e.ok()) { LSE_EXPECT_OK(e.status()); return; }
        auto tree = layout_tree(l, out.back(), *e, static_cast<std::uint32_t>(e->nodes.size()));
        if (!tree.ok()) { LSE_EXPECT_OK(tree.status()); return; }
        const std::vector<std::uint32_t> before = out;
        auto w = walk_tree(*tree, [&](std::uint32_t row) -> lse::Result<std::pair<std::uint32_t, bool>> {
          // The context of a row: what was emitted, then the path to it.
          std::vector<std::uint32_t> ctx = before;
          std::vector<std::uint32_t> path;
          for (auto r = static_cast<std::int32_t>(row); r > 0; r = tree->parent[static_cast<std::size_t>(r)])
            path.push_back(tree->tokens[static_cast<std::size_t>(r)]);
          ctx.insert(ctx.end(), path.rbegin(), path.rend());
          LSE_ASSIGN_OR(const std::uint32_t x, sampler.sample_target(target(ctx)));
          return std::pair(x, true);
        });
        if (!w.ok()) { LSE_EXPECT_OK(w.status()); return; }
        for (const auto x : w->answers) out.push_back(x);
      }
      out.resize(3);
      counts[out] += 1;
    }
    double chi2 = 0;
    for (std::uint32_t a = 0; a < V; ++a)
      for (std::uint32_t b = 0; b < V; ++b)
        for (std::uint32_t c = 0; c < V; ++c) {
          const double p = target({}).probabilities[a] * target({a}).probabilities[b] *
                           target({a, b}).probabilities[c];
          const double expected = p * kRuns;
          const double seen = counts[{a, b, c}];
          chi2 += (seen - expected) * (seen - expected) / expected;
        }
    // 124 degrees of freedom: the 1e-6 upper tail starts near 219.
    std::printf("    budget %u: chi2 %.1f over 124 dof\n", budget, chi2);
    LSE_EXPECT(chi2 < 219.0);
  }
}

LSE_TEST_MAIN()
