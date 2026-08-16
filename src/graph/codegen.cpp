#include "lse/graph/codegen.hpp"

#include "lse/graph/graph.hpp"


namespace lse::graph {

std::uint64_t IKernelEmitter::cache_key(const FusionGroup& group,
                                        const backend::DeviceInfo&) const {
  return group.signature();
}

std::uint16_t ConstantsLayout::add(std::string name, std::uint8_t size) {
  const std::uint16_t offset = static_cast<std::uint16_t>(total_bytes);
  fields.push_back(Field{std::move(name), offset, size});
  total_bytes += size;
  return offset;
}

}  // namespace lse::graph
