#include "lse/runtime/draft_tree.hpp"

#include <algorithm>
#include <cmath>
#include <queue>
#include <string>

namespace lse::runtime {

Status DraftLattice::validate(std::uint32_t vocab_size) const {
  if (positions == 0 || top == 0 || positions > 255 || top > 255 ||
      scores.size() != static_cast<std::size_t>(positions) * top * top ||
      ids.size() != static_cast<std::size_t>(positions) * top)
    return LSE_ERROR(kInvalidArgument, "invalid DFlash2 candidate lattice");
  for (const float s : scores)
    if (!std::isfinite(s)) return LSE_ERROR(kInvalidArgument, "nonfinite DFlash2 selector score");
  for (std::uint32_t p = 0; p < positions; ++p)
    for (std::uint32_t k = 0; k < top; ++k) {
      const std::uint32_t id = ids[static_cast<std::size_t>(p) * top + k];
      if (vocab_size != 0 && id >= vocab_size)
        return LSE_ERROR(kInvalidArgument, "DFlash2 candidate ", std::to_string(id),
                         " is outside the vocabulary");
      for (std::uint32_t j = 0; j < k; ++j)
        if (ids[static_cast<std::size_t>(p) * top + j] == id)
          return LSE_ERROR(kInvalidArgument, "repeated DFlash2 candidate at position ",
                           std::to_string(p));
    }
  return OkStatus();
}

void DraftLattice::conditional(std::uint32_t position, std::uint32_t predecessor,
                               double temperature, std::span<double> out) const {
  const float* row = scores.data() + (static_cast<std::size_t>(position) * top + predecessor) * top;
  double maximum = -INFINITY;
  for (std::uint32_t k = 0; k < top; ++k) maximum = std::max(maximum, static_cast<double>(row[k]));
  double total = 0;
  for (std::uint32_t k = 0; k < top; ++k) {
    out[k] = std::exp((static_cast<double>(row[k]) - maximum) / temperature);
    total += out[k];
  }
  for (std::uint32_t k = 0; k < top; ++k) out[k] /= total;
}

Result<TreeExpansion> expand_tree(const DraftLattice& lattice, double temperature,
                                  std::uint32_t max_nodes,
                                  const std::function<double(double)>& calibrate) {
  LSE_RETURN_IF_ERROR(lattice.validate(0));
  const double t = temperature > 0 ? temperature : 1.0;
  const std::uint32_t top = lattice.top;
  // Conditionals at every (position, predecessor); position 0 has one row.
  std::vector<double> cond(static_cast<std::size_t>(lattice.positions) * top * top);
  for (std::uint32_t p = 0; p < lattice.positions; ++p)
    for (std::uint32_t pred = 0; pred < (p == 0 ? 1u : top); ++pred)
      lattice.conditional(p, pred, t,
          std::span(cond).subspan((static_cast<std::size_t>(p) * top + pred) * top, top));
  const auto local_of = [&](double q) {
    const double c = calibrate ? calibrate(q) : q;
    return std::isfinite(c) ? std::clamp(c, 0.0, 1.0) : 0.0;
  };

  struct Entry {
    double value;
    std::uint64_t order;  // ties pop in push order, which is deterministic
    TreeExpansion::Node node;
  };
  const auto worse = [](const Entry& a, const Entry& b) {
    return a.value != b.value ? a.value < b.value : a.order > b.order;
  };
  std::priority_queue<Entry, std::vector<Entry>, decltype(worse)> heap(worse);
  std::uint64_t order = 0;
  const auto push_children = [&](std::int32_t parent, std::uint32_t position,
                                 std::uint32_t predecessor, double parent_value) {
    const double* q = cond.data() + (static_cast<std::size_t>(position) * top + predecessor) * top;
    // Best conditional first, so equal values pop best candidate first.
    std::vector<std::uint32_t> ranked(top);
    for (std::uint32_t k = 0; k < top; ++k) ranked[k] = k;
    std::stable_sort(ranked.begin(), ranked.end(),
                     [&](std::uint32_t a, std::uint32_t b) { return q[a] > q[b]; });
    for (const std::uint32_t k : ranked) {
      TreeExpansion::Node n;
      n.parent = parent;
      n.position = static_cast<std::uint8_t>(position);
      n.candidate = static_cast<std::uint8_t>(k);
      n.probability = q[k];
      n.local = local_of(q[k]);
      n.value = parent_value * n.local;
      if (!(n.value > 0.0)) continue;
      heap.push({n.value, order++, n});
    }
  };

  TreeExpansion out;
  out.value_prefix.push_back(0.0);
  push_children(-1, 0, 0, 1.0);
  while (out.nodes.size() < max_nodes && !heap.empty()) {
    const Entry e = heap.top();
    heap.pop();
    const auto index = static_cast<std::int32_t>(out.nodes.size());
    out.nodes.push_back(e.node);
    out.value_prefix.push_back(out.value_prefix.back() + e.node.value);
    if (e.node.position + 1u < lattice.positions)
      push_children(index, e.node.position + 1u, e.node.candidate, e.node.value);
  }
  return out;
}

std::uint32_t DraftTree::max_depth() const noexcept {
  std::uint32_t d = 0;
  for (const std::uint32_t x : depth) d = std::max(d, x);
  return d;
}

std::int32_t DraftTree::ancestor(std::uint32_t row, std::uint32_t up) const noexcept {
  auto r = static_cast<std::int32_t>(row);
  for (std::uint32_t i = 0; i < up && r >= 0; ++i) r = parent[static_cast<std::size_t>(r)];
  return r;
}

bool DraftTree::sees(std::uint32_t row, std::uint32_t a) const noexcept {
  for (auto r = static_cast<std::int32_t>(row); r >= 0; r = parent[static_cast<std::size_t>(r)])
    if (static_cast<std::uint32_t>(r) == a) return true;
  return false;
}

std::int32_t DraftTree::child_with(std::uint32_t row, std::uint32_t token) const noexcept {
  for (const std::uint32_t c : children[row])
    if (tokens[c] == token) return static_cast<std::int32_t>(c);
  return -1;
}

Status DraftTree::validate() const {
  const std::size_t n = tokens.size();
  if (n == 0 || parent.size() != n || depth.size() != n || children.size() != n ||
      parent[0] != -1 || depth[0] != 0 || main_rows == 0 || main_rows > n)
    return LSE_ERROR(kInternal, "malformed draft tree");
  // The latest row seen at each depth: in preorder it is the parent of the
  // next row one level deeper.
  std::vector<std::int32_t> last_at(n + 1, -1);
  last_at[0] = 0;
  for (std::size_t r = 1; r < n; ++r) {
    const std::int32_t p = parent[r];
    if (p < 0 || static_cast<std::size_t>(p) >= r || depth[r] != depth[static_cast<std::size_t>(p)] + 1 ||
        last_at[depth[r] - 1] != p)
      return LSE_ERROR(kInternal, "draft tree row ", std::to_string(r), " is not in preorder");
    last_at[depth[r]] = static_cast<std::int32_t>(r);
    for (std::size_t d = depth[r] + 1; d <= n; ++d) last_at[d] = -1;
  }
  for (std::size_t r = 0; r < n; ++r) {
    for (const std::uint32_t c : children[r])
      if (c >= n || parent[c] != static_cast<std::int32_t>(r))
        return LSE_ERROR(kInternal, "draft tree child list disagrees with its parents");
    for (std::size_t i = 0; i < children[r].size(); ++i)
      for (std::size_t j = 0; j < i; ++j)
        if (tokens[children[r][i]] == tokens[children[r][j]])
          return LSE_ERROR(kInternal, "draft tree siblings repeat a token");
  }
  for (std::uint32_t r = 0; r + 1 < main_rows; ++r)
    if (children[r].empty() || children[r][0] != r + 1)
      return LSE_ERROR(kInternal, "draft tree top path is not rows 0..", std::to_string(main_rows - 1));
  if (!children[main_rows - 1].empty())
    return LSE_ERROR(kInternal, "draft tree top path stops before a leaf");
  return OkStatus();
}

Result<DraftTree> layout_tree(const DraftLattice& lattice, std::uint32_t anchor,
                              const TreeExpansion& expansion, std::uint32_t nodes) {
  if (nodes > expansion.nodes.size())
    return LSE_ERROR(kInvalidArgument, "a tree of ", std::to_string(nodes), " nodes from ",
                     std::to_string(expansion.nodes.size()), " expanded");
  // Children in pop order, which is best first.
  std::vector<std::vector<std::uint32_t>> kids(nodes + 1);  // [0]: the root's
  for (std::uint32_t i = 0; i < nodes; ++i) {
    const std::int32_t p = expansion.nodes[i].parent;
    if (p >= static_cast<std::int32_t>(i)) return LSE_ERROR(kInternal, "expansion pops a child first");
    kids[static_cast<std::size_t>(p + 1)].push_back(i);
  }
  DraftTree tree;
  tree.tokens.reserve(nodes + 1);
  // Preorder with an explicit stack: (expansion index + 1, parent row).
  std::vector<std::pair<std::uint32_t, std::int32_t>> stack{{0u, -1}};
  while (!stack.empty()) {
    const auto [slot, parent_row] = stack.back();
    stack.pop_back();
    const auto row = static_cast<std::uint32_t>(tree.tokens.size());
    if (slot == 0) {
      tree.tokens.push_back(anchor);
      tree.depth.push_back(0);
      tree.node.push_back(-1);
      tree.candidate.push_back(-1);
    } else {
      const TreeExpansion::Node& n = expansion.nodes[slot - 1];
      tree.tokens.push_back(lattice.ids[static_cast<std::size_t>(n.position) * lattice.top + n.candidate]);
      tree.depth.push_back(n.position + 1u);
      tree.node.push_back(static_cast<std::int32_t>(slot - 1));
      tree.candidate.push_back(n.candidate);
    }
    tree.parent.push_back(parent_row);
    tree.children.emplace_back();
    if (parent_row >= 0) tree.children[static_cast<std::size_t>(parent_row)].push_back(row);
    const auto& mine = kids[slot];
    for (auto it = mine.rbegin(); it != mine.rend(); ++it)
      stack.emplace_back(*it + 1, static_cast<std::int32_t>(row));
  }
  tree.main_rows = 1;
  while (!tree.children[tree.main_rows - 1].empty()) ++tree.main_rows;
  LSE_RETURN_IF_ERROR(tree.validate());
  return tree;
}

DraftTree chain_tree(std::uint32_t anchor, std::span<const std::uint32_t> tokens) {
  DraftTree tree;
  const std::size_t n = tokens.size() + 1;
  tree.tokens.reserve(n);
  tree.tokens.push_back(anchor);
  tree.tokens.insert(tree.tokens.end(), tokens.begin(), tokens.end());
  tree.children.resize(n);
  for (std::size_t r = 0; r < n; ++r) {
    tree.parent.push_back(static_cast<std::int32_t>(r) - 1);
    tree.depth.push_back(static_cast<std::uint32_t>(r));
    tree.node.push_back(r == 0 ? -1 : static_cast<std::int32_t>(r - 1));
    tree.candidate.push_back(-1);
    if (r + 1 < n) tree.children[r].push_back(static_cast<std::uint32_t>(r + 1));
  }
  tree.main_rows = static_cast<std::uint32_t>(n);
  return tree;
}

Result<TreeWalk> walk_tree(const DraftTree& tree, const TreeAnswer& answer) {
  TreeWalk walk;
  std::uint32_t row = 0;
  for (;;) {
    walk.path.push_back(row);
    LSE_ASSIGN_OR(const auto said, answer(row));
    walk.answers.push_back(said.first);
    if (!said.second) {
      walk.stopped = true;
      break;
    }
    const std::int32_t next = tree.child_with(row, said.first);
    if (next < 0) break;
    row = static_cast<std::uint32_t>(next);
  }
  return walk;
}

TreeRows tree_rows(const DraftTree& tree) {
  TreeRows out;
  out.main_rows = tree.main_rows;
  for (std::size_t r = 0; r < tree.rows(); ++r) {
    out.depth.push_back(static_cast<std::int32_t>(tree.depth[r]));
    out.parent.push_back(tree.parent[r]);
  }
  return out;
}

}  // namespace lse::runtime
