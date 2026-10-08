#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "lse/kv/cache_dtype.hpp"
#include "lse/model/layer.hpp"
#include "lse/graph/program.hpp"

namespace lse::model {

// Retain verifier inputs, then overwrite its carry outputs with a valid prefix.
class PrefixStateCommit {
 public:
  Status retain(std::span<const MixerState> states, std::int64_t sequence_rows,
                std::vector<graph::NodePtr>& roots);
  Status commit(std::size_t rows, graph::Scheduler& scheduler);
  // The kernels commit(rows) launches, made resident (Scheduler::prepare)
  // without the verifier having run.
  Status prepare(std::size_t rows, graph::Scheduler& scheduler) const;
  [[nodiscard]] bool ready() const noexcept { return sequence_rows_ > 1; }

 private:
  struct Entry {
    graph::NodePtr output;
    std::vector<graph::NodePtr> sources;
    bool recurrent = false;
  };
  struct Replay {
    graph::Program program;
    std::vector<graph::Array> inputs;
    std::vector<graph::Array> outputs;
  };
  void build_replay(Replay& replay, std::size_t rows, graph::Scheduler& scheduler) const;
  std::int64_t sequence_rows_ = 0;
  std::vector<Entry> entries_;
  std::vector<Replay> replays_;
};

// A draft tree's verify pass (TreeLayout) keeps what its accepted path is
// replayed from: every GDN layer's k, v, alpha, beta and starting state, its
// conv input rows and starting conv tail, and every attention layer's rotated
// keys and values. commit() writes the path's recurrent state and conv tail
// into the pass's carried outputs and its keys and values to the positions
// that follow the pass's start, one small program for the whole stack.
class TreeStateCommit {
 public:
  // Path rows a commit can name: a tree's depth bound plus its root.
  static constexpr std::size_t kMaxPath = 9;
  // `features`, when valid, is the pass's captured target features [1, T, F]:
  // the commit also gathers the path's rows of them (path_features()).
  Status retain(std::span<const MixerState> states, std::int64_t rows,
                std::vector<graph::NodePtr>& roots, const graph::Array& features = {});
  // `path` holds pass rows, root first; `first` is the pass's first position.
  Status commit(std::span<const std::uint32_t> path, std::int32_t first,
                graph::Scheduler& scheduler);
  // The last commit's path rows of the features, [1, path, F] (invalid when
  // the pass captured none). Valid until the next commit.
  [[nodiscard]] graph::Array path_features() const;
  Status prepare(graph::Scheduler& scheduler);
  [[nodiscard]] bool ready() const noexcept { return rows_ > 1; }

 private:
  struct Entry {
    enum class Kind : std::uint8_t { kRecurrent, kConvTail, kKeysValues } kind;
    graph::NodePtr output;                 // carried output, or the pool
    std::vector<graph::NodePtr> sources;   // as the replay reads them
    kv::CacheDType storage = kv::CacheDType::kF32;
  };
  Status build_replay(graph::Scheduler& scheduler);
  std::int64_t rows_ = 0;
  std::vector<Entry> entries_;
  graph::Program program_;
  std::vector<graph::Array> inputs_;
  std::vector<graph::Array> outputs_;
  graph::Array path_;
  graph::NodePtr features_;
  graph::Array feature_leaf_, gathered_;
  std::size_t gathered_rows_ = 0;
  bool built_ = false;
};

}  // namespace lse::model
