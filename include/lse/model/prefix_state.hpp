#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "lse/model/layer.hpp"
#include "lse/graph/program.hpp"

namespace lse::model {

// Retain verifier inputs, then overwrite its carry outputs with a valid prefix.
class PrefixStateCommit {
 public:
  Status retain(std::span<const MixerState> states, std::int64_t sequence_rows,
                std::vector<graph::NodePtr>& roots);
  Status commit(std::size_t rows, graph::Scheduler& scheduler);
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
  std::int64_t sequence_rows_ = 0;
  std::vector<Entry> entries_;
  std::vector<Replay> replays_;
};

}  // namespace lse::model
