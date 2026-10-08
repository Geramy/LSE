// Draft trees from a DFlash2 candidate lattice, verified in one target pass.
//
// A DFlash2 draft scores, for each of its seven proposal positions, its top
// candidates given each candidate of the position before (the first position
// given the anchor): a first-order lattice of conditional distributions. A
// chain verifies one path through it. A tree verifies the B most probable
// paths at once (DDTree, Ringel and Romano, arXiv 2604.12989): every node's
// path probability is the product of the conditionals along it, the best B
// prefixes are popped from a heap in order of that probability, and since a
// child is never more probable than its parent the first B pops always form a
// tree. That tree maximizes the expected number of accepted nodes over all
// B-node trees, and the expected value of every smaller budget is a prefix sum
// of the same pop order.
//
// Verification walks the tree from the root (the anchor, already emitted): at
// a node the target answers (its argmax, or a sample of its distribution at
// that node's context), and the walk descends into the child holding that
// token or stops and emits the answer. Every emitted token is therefore the
// target's own answer at its context, so the output follows the target's
// distribution exactly. For a candidate set fixed before the target is
// consulted this is also the most any lossless rule can accept: the walk
// continues with probability equal to the target's mass on the children.
//
// Rows are laid out in depth-first preorder, best child first: rows 0..D are
// the top path (what a chain verifies), every node comes after its parent,
// and the most recent row at depth d - 1 before a row at depth d is its
// parent, which is what lets a recurrent layer keep one state per depth.
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <vector>

#include "lse/core/status.hpp"

namespace lse::runtime {

// One draft's candidates. scores[(p * top + predecessor) * top + k] scores
// candidate k of position p given candidate `predecessor` of position p - 1;
// at p = 0 every predecessor row is the anchor's. ids[p * top + k] is the
// token of candidate k at position p; the ids of one position are distinct.
struct DraftLattice {
  std::uint32_t positions = 0;
  std::uint32_t top = 0;
  std::vector<float> scores;
  std::vector<std::uint32_t> ids;

  [[nodiscard]] Status validate(std::uint32_t vocab_size) const;
  // The draft's conditional distribution at position p given predecessor
  // candidate `predecessor`, at `temperature` (> 0).
  void conditional(std::uint32_t position, std::uint32_t predecessor, double temperature,
                   std::span<double> out) const;
};

// The best-first order of a lattice's nodes, by path value.
struct TreeExpansion {
  struct Node {
    std::int32_t parent = -1;      // index into nodes; -1: a child of the root
    std::uint8_t position = 0;     // proposal position (depth - 1)
    std::uint8_t candidate = 0;    // candidate index at that position
    double probability = 0;        // the draft's conditional given the parent
    double local = 0;              // its calibrated acceptance
    double value = 0;              // product of `local` along the path
  };
  // Pop order: non-increasing value, every parent before its children.
  std::vector<Node> nodes;
  // value_prefix[b] = expected accepted nodes of the first b pops.
  std::vector<double> value_prefix;
};

// Pops up to `max_nodes` nodes. `calibrate` maps a conditional probability to
// the chance the target picks that candidate; it must return values in
// [0, 1] (identity when null). `temperature` <= 0 scores at temperature 1.
[[nodiscard]] Result<TreeExpansion> expand_tree(
    const DraftLattice& lattice, double temperature, std::uint32_t max_nodes,
    const std::function<double(double)>& calibrate = {});

// A tree laid out for a verify pass: row 0 is the anchor, rows in depth-first
// preorder with the best child first.
struct DraftTree {
  std::vector<std::uint32_t> tokens;
  std::vector<std::int32_t> parent;      // parent[0] = -1
  std::vector<std::uint32_t> depth;      // depth[0] = 0
  std::vector<std::int32_t> node;        // index into the expansion; -1 for the root
  std::vector<std::int32_t> candidate;   // candidate index at its position; -1 for the root
  std::vector<std::vector<std::uint32_t>> children;  // best first
  // Rows 0..main_rows-1 are the top path: each row's first child is the next.
  std::uint32_t main_rows = 1;

  [[nodiscard]] std::size_t rows() const noexcept { return tokens.size(); }
  [[nodiscard]] std::uint32_t max_depth() const noexcept;
  // Row r's ancestor `up` levels above (0: r itself), or -1 past the root.
  [[nodiscard]] std::int32_t ancestor(std::uint32_t row, std::uint32_t up) const noexcept;
  // Whether `a` is `row` or one of its ancestors.
  [[nodiscard]] bool sees(std::uint32_t row, std::uint32_t a) const noexcept;
  // The child of `row` holding `token`, or -1.
  [[nodiscard]] std::int32_t child_with(std::uint32_t row, std::uint32_t token) const noexcept;
  // Checks the layout invariants: preorder, parents first, depth, the top path.
  [[nodiscard]] Status validate() const;
};

// The first `nodes` pops of `expansion` under `anchor`.
[[nodiscard]] Result<DraftTree> layout_tree(const DraftLattice& lattice, std::uint32_t anchor,
                                            const TreeExpansion& expansion,
                                            std::uint32_t nodes);

// A tree in the shape of a chain: `tokens` after `anchor`, each the only child
// of the one before.
[[nodiscard]] DraftTree chain_tree(std::uint32_t anchor, std::span<const std::uint32_t> tokens);

// The rows a walk accepted, from row 0, and the target's answer at each of
// them: answers.back() is the token after the last accepted row.
struct TreeWalk {
  std::vector<std::uint32_t> path;
  std::vector<std::uint32_t> answers;
  // The walk ended because answer() asked it to (a stop token, a limit),
  // not at a token outside the tree.
  bool stopped = false;
};

// `answer(row)` returns the target's token at a row (its argmax, or a sample
// of its distribution there given the context and the path so far) and
// whether the walk may go on after emitting it.
using TreeAnswer = std::function<Result<std::pair<std::uint32_t, bool>>(std::uint32_t row)>;
[[nodiscard]] Result<TreeWalk> walk_tree(const DraftTree& tree, const TreeAnswer& answer);

// The per-row descriptors a tree verify pass reads (model::TreeLayout), as a
// flat list: see model::TreeLayout for the meaning of each.
struct TreeRows {
  std::vector<std::int32_t> depth;
  std::vector<std::int32_t> parent;
  std::uint32_t main_rows = 1;
};
[[nodiscard]] TreeRows tree_rows(const DraftTree& tree);

}  // namespace lse::runtime
