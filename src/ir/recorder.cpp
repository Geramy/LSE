#include "lse/ir/recorder.hpp"

#include <cstdio>

#include "lse/core/debug.hpp"
#include "lse/ir/pass/pass.hpp"

namespace lse::ir {

thread_local KernelBody* KernelBody::current_ = nullptr;
thread_local const RecordOptions* KernelBody::options_ = nullptr;
thread_local KernelBody::Capture* KernelBody::capture_ = nullptr;

KernelBody::KernelBody(const TypeTable& types,
                       const DialectSourceTable& intrinsics,
                       std::uint32_t lds_budget)
    : types_(&types), intrinsics_(&intrinsics), ir_(types, intrinsics),
      lds_(lds_budget) {
  if (options_ != nullptr) ir_.set_name_prefix(std::string(options_->name_prefix));
  lds_.attach(this);
  bind();
}

void KernelBody::bind() noexcept {
  prev_ = current_;
  current_ = this;
}

void KernelBody::unbind() noexcept {
  // The outermost body of a recording is the one the primitive built; a helper
  // that opened a second one is not what the emitter asked for.
  if (capture_ != nullptr && prev_ == nullptr && !capture_->taken_.has_value()) {
    capture_->taken_.emplace(std::move(ir_));
  }
  current_ = prev_;
}

KernelBody::Capture::Capture() noexcept : prev_(capture_) { capture_ = this; }

KernelBody::Capture::~Capture() { capture_ = prev_; }

KernelBody* KernelBody::try_current() noexcept { return current_; }

KernelBody::Recording::Recording(RecordOptions o) noexcept
    : prev_(options_), opts_(o) {
  options_ = &opts_;
}

KernelBody::Recording::~Recording() { options_ = prev_; }

std::string KernelBody::input_name(std::size_t index) const {
  if (options_ != nullptr && index < options_->input_names.size()) {
    return options_->input_names[index];
  }
  return "in" + std::to_string(index);
}

std::string KernelBody::output_name() const {
  if (options_ != nullptr && !options_->output_name.empty()) {
    return std::string(options_->output_name);
  }
  return "out";
}

Val<u32> KernelBody::thread_id() {
  return {types_, &ir_, ir_.symbol("i", scalar_type(Scalar::kU32))};
}

namespace detail {

Val<u32> lift_u32(Body* b, const TypeTable* tt, std::uint32_t v) {
  if (b == nullptr) return {};
  return {tt, b,
          b->constant(literal_u32(v), scalar_type(Scalar::kU32),
                      static_cast<std::int64_t>(v), true)};
}

Val<i32> lift_i32(Body* b, const TypeTable* tt, std::int32_t v) {
  if (b == nullptr) return {};
  return {tt, b,
          b->constant(literal_i32(v), scalar_type(Scalar::kI32), v, true)};
}

Val<f32> lift_f32(Body* b, const TypeTable* tt, float v) {
  if (b == nullptr) return {};
  return {tt, b, b->constant(float_literal(v), scalar_type(Scalar::kF32), 0,
                             false)};
}

ValueId int_literal(Body* body, int i) {
  return body->constant(std::to_string(i), scalar_type(Scalar::kI32), i, true);
}

}  // namespace detail

std::string KernelBody::str() {
  std::vector<PassStat> stats;
  // A pass that leaves the body malformed is a bug in the pass, and the point
  // of the verifier is that it says so here rather than in the generated
  // source. Lowering the pre-pass body would hide it; there is nothing to fall
  // back to, so the failure is loud.
  if (const Status s = default_pipeline().run(ir_, &stats); !s.ok()) {
    std::fprintf(stderr, "lse: kernel IR pass pipeline failed: %s\n",
                 s.message().c_str());
    return {};
  }
  record_pass_totals(stats);
  return lower(ir_);
}

void KernelBody::statement(std::string text) {
  Operation o;
  o.kind = OpKind::kRawStmt;
  o.text = std::move(text);
  ir_.add(std::move(o));
}

void KernelBody::barrier() {
  const std::string_view spell = intrinsics_->find("barrier");
  if (spell.empty()) return;
  Operation o;
  o.kind = OpKind::kBarrier;
  o.text = std::string(spell);
  ir_.add(std::move(o));
}

void KernelBody::ret() {
  Operation o;
  o.kind = OpKind::kReturn;
  ir_.add(std::move(o));
}

void KernelBody::ret_if(const Val<boolean>& cond) {
  Operation o;
  o.kind = OpKind::kReturnIf;
  o.operands = {cond.id()};
  ir_.add(std::move(o));
}

void KernelBody::begin_if(const Val<boolean>& cond, bool indent_body) {
  Operation o;
  o.kind = OpKind::kIf;
  o.operands = {cond.id()};
  if (indent_body) o.flags |= kFlagIndentBody;
  const OpId id = ir_.add(std::move(o));
  ir_.push(ir_.open_region(id));
}

Val<u32> KernelBody::begin_for(std::string_view var, const Val<u32>& lo,
                               const Val<u32>& hi, const Val<u32>& step,
                               bool unroll, bool indent_body) {
  Operation o;
  o.kind = OpKind::kFor;
  o.type = scalar_type(Scalar::kU32);
  o.operands = {lo.id(), hi.id(), step.id()};
  o.key = std::string(var);
  if (unroll) o.flags |= kFlagUnroll;
  if (indent_body) o.flags |= kFlagIndentBody;
  const OpId id = ir_.add(std::move(o), std::string(var));
  const ValueId iv = ir_.op(id).result;
  ir_.push(ir_.open_region(id));
  return {types_, &ir_, iv};
}

void KernelBody::end_block() { ir_.pop(); }

void KernelBody::store(const Val<u32>& index, const Val<f32>& value) {
  if (!store_) return;  // emit_kernel declines when the emitter set no hook
  // Braced so the epilogue's own locals are scoped to this store: a kernel
  // that stores several results does so in one scope.
  Operation scope;
  scope.kind = OpKind::kScope;
  const OpId sid = ir_.add(std::move(scope));
  ir_.push(ir_.open_region(sid));

  const std::string text = store_(index.text(), value.text());
  bool first = true;
  std::size_t pos = 0;
  while (pos <= text.size()) {
    const std::size_t nl = text.find('\n', pos);
    const std::string line =
        text.substr(pos, nl == std::string::npos ? text.size() - pos : nl - pos);
    if (!line.empty()) {
      Operation o;
      o.kind = OpKind::kRawStmt;
      o.text = line;
      // The hook expanded the index and the value into opaque text. Naming
      // them as operands is what stops a later pass deleting the definitions
      // the text still mentions.
      if (first) {
        o.operands = {index.id(), value.id()};
        first = false;
      }
      ir_.add(std::move(o));
    }
    if (nl == std::string::npos) break;
    pos = nl + 1;
  }
  ir_.pop();
}

}  // namespace lse::ir
