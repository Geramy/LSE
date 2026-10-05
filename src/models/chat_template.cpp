#include "lse/models/chat_template.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace lse::models {
namespace jinja {
namespace {

using ojson = nlohmann::ordered_json;

// Thrown inside the interpreter and turned into a Status at its boundary.
struct Error : std::runtime_error {
  using std::runtime_error::runtime_error;
};
// raise_exception() in a template.
struct Raised : Error {
  using Error::Error;
};

[[noreturn]] void fail(int line, const std::string& what) {
  throw Error("chat template line " + std::to_string(line) + ": " + what);
}

// --- UTF-8: Python strings count and index code points -----------------------

std::vector<std::uint32_t> decode_utf8(const std::string& s) {
  std::vector<std::uint32_t> out;
  out.reserve(s.size());
  for (std::size_t i = 0; i < s.size();) {
    const auto c = static_cast<unsigned char>(s[i]);
    std::uint32_t cp = c;
    std::size_t extra = 0;
    if (c >= 0xF0) { cp = c & 0x07u; extra = 3; }
    else if (c >= 0xE0) { cp = c & 0x0Fu; extra = 2; }
    else if (c >= 0xC0) { cp = c & 0x1Fu; extra = 1; }
    ++i;
    for (std::size_t k = 0; k < extra && i < s.size(); ++k, ++i)
      cp = (cp << 6) | (static_cast<unsigned char>(s[i]) & 0x3Fu);
    out.push_back(cp);
  }
  return out;
}

std::string encode_utf8(const std::vector<std::uint32_t>& cps) {
  std::string out;
  for (std::uint32_t cp : cps) {
    if (cp < 0x80) {
      out += static_cast<char>(cp);
    } else if (cp < 0x800) {
      out += static_cast<char>(0xC0 | (cp >> 6));
      out += static_cast<char>(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
      out += static_cast<char>(0xE0 | (cp >> 12));
      out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
      out += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
      out += static_cast<char>(0xF0 | (cp >> 18));
      out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
      out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
      out += static_cast<char>(0x80 | (cp & 0x3F));
    }
  }
  return out;
}

constexpr std::string_view kWhitespace = " \t\n\r\f\v";

// --- the program ---------------------------------------------------------------

enum class Op {
  kLiteral, kName, kList, kTuple, kDict, kAttr, kItem, kSlice, kCall,
  kFilter, kTest, kNot, kNeg, kPos, kBinary, kAnd, kOr, kCond
};

struct Expr;
using ExprPtr = std::unique_ptr<Expr>;
struct Expr {
  Op op = Op::kLiteral;
  int line = 0;
  ojson literal;
  // Name, attribute, filter, test or binary operator.
  std::string name;
  // Operands, items, call or filter arguments; a dict alternates key, value.
  // A missing slice bound is null.
  std::vector<ExprPtr> args;
  std::vector<std::pair<std::string, ExprPtr>> kwargs;
  bool negated = false;
};

enum class NodeKind { kText, kOutput, kIf, kFor, kSet, kSetBlock, kMacro, kBreak, kContinue, kBlock };

struct Node;
using NodePtr = std::unique_ptr<Node>;
using Body = std::vector<NodePtr>;
struct Node {
  NodeKind kind = NodeKind::kText;
  int line = 0;
  std::string text;                                // kText; macro and set-block name
  ExprPtr expr;                                    // output, set value, for iterable
  std::vector<std::pair<ExprPtr, Body>> branches;  // if / elif
  Body else_body;                                  // if, for
  std::vector<std::string> targets;                // for / set names
  std::string attribute;                           // set ns.attribute
  ExprPtr filter;                                  // for ... if filter
  Body body;
  std::vector<std::string> params;                 // macro
  std::vector<ExprPtr> defaults;                   // macro, null when none
};

}  // namespace

struct Program {
  Body body;
};

namespace {

// --- lexing: text and tags, with Jinja's whitespace control --------------------

enum class SegKind { kText, kExpr, kStmt, kComment };
struct Segment {
  SegKind kind = SegKind::kText;
  std::string text;
  int line = 1;
  bool trim_left = false;   // {{- {%- {#-
  bool trim_right = false;  // -}} -%} -#}
  bool keep_left = false;   // {%+ disables lstrip_blocks
};

std::size_t find_close(std::string_view src, std::size_t from, std::string_view close, bool strings) {
  char quote = 0;
  for (std::size_t i = from; i + close.size() <= src.size(); ++i) {
    const char c = src[i];
    if (quote != 0) {
      if (c == '\\') ++i;
      else if (c == quote) quote = 0;
      continue;
    }
    if (strings && (c == '\'' || c == '"')) {
      quote = c;
      continue;
    }
    if (src.substr(i, close.size()) == close) return i;
  }
  return std::string_view::npos;
}

std::vector<Segment> lex(std::string_view src) {
  std::vector<Segment> raw;
  int line = 1;
  std::size_t i = 0;
  const auto count_lines = [&](std::size_t from, std::size_t to) {
    for (std::size_t k = from; k < to; ++k)
      if (src[k] == '\n') ++line;
  };
  while (i < src.size()) {
    std::size_t open = i;
    for (;;) {
      open = src.find('{', open);
      if (open == std::string_view::npos || open + 1 >= src.size()) {
        open = std::string_view::npos;
        break;
      }
      const char n = src[open + 1];
      if (n == '{' || n == '%' || n == '#') break;
      ++open;
    }
    if (open == std::string_view::npos) {
      raw.push_back({SegKind::kText, std::string(src.substr(i)), line});
      break;
    }
    if (open > i) {
      raw.push_back({SegKind::kText, std::string(src.substr(i, open - i)), line});
      count_lines(i, open);
    }
    Segment seg;
    seg.line = line;
    const char n = src[open + 1];
    seg.kind = n == '{' ? SegKind::kExpr : n == '%' ? SegKind::kStmt : SegKind::kComment;
    std::size_t body = open + 2;
    if (body < src.size() && src[body] == '-') { seg.trim_left = true; ++body; }
    else if (body < src.size() && src[body] == '+') { seg.keep_left = true; ++body; }
    const std::string_view closer = seg.kind == SegKind::kExpr ? "}}" : seg.kind == SegKind::kStmt ? "%}" : "#}";
    const std::size_t close = find_close(src, body, closer, seg.kind != SegKind::kComment);
    if (close == std::string_view::npos) fail(line, "unterminated tag");
    std::size_t end = close;
    if (end > body && src[end - 1] == '-') { seg.trim_right = true; --end; }
    else if (end > body && src[end - 1] == '+') { --end; }
    seg.text = std::string(src.substr(body, end - body));
    count_lines(open, close + 2);
    raw.push_back(std::move(seg));
    i = close + 2;
  }

  // transformers renders with trim_blocks and lstrip_blocks. Decisions are made
  // on the original text, then applied.
  std::vector<std::size_t> strip_front(raw.size(), 0);
  std::vector<std::size_t> strip_back(raw.size(), 0);
  for (std::size_t k = 0; k < raw.size(); ++k) {
    const Segment& s = raw[k];
    if (s.kind == SegKind::kText) continue;
    const bool block = s.kind != SegKind::kExpr;
    if (k > 0 && raw[k - 1].kind == SegKind::kText) {
      const std::string& t = raw[k - 1].text;
      if (s.trim_left) {
        const auto keep = t.find_last_not_of(kWhitespace);
        strip_back[k - 1] = keep == std::string::npos ? t.size() : t.size() - keep - 1;
      } else if (block && !s.keep_left) {
        std::size_t m = t.size();
        while (m > 0 && (t[m - 1] == ' ' || t[m - 1] == '\t')) --m;
        const bool line_start = m == 0 ? k - 1 == 0 : t[m - 1] == '\n';
        if (line_start) strip_back[k - 1] = t.size() - m;
      }
    }
    if (k + 1 < raw.size() && raw[k + 1].kind == SegKind::kText) {
      const std::string& t = raw[k + 1].text;
      if (s.trim_right) {
        const auto keep = t.find_first_not_of(kWhitespace);
        strip_front[k + 1] = keep == std::string::npos ? t.size() : keep;
      } else if (block) {
        if (t.starts_with("\r\n")) strip_front[k + 1] = 2;
        else if (t.starts_with("\n")) strip_front[k + 1] = 1;
      }
    }
  }
  std::vector<Segment> out;
  for (std::size_t k = 0; k < raw.size(); ++k) {
    Segment s = std::move(raw[k]);
    if (s.kind == SegKind::kComment) continue;
    if (s.kind == SegKind::kText) {
      const std::size_t front = std::min(strip_front[k], s.text.size());
      const std::size_t back = std::min(strip_back[k], s.text.size() - front);
      s.text = s.text.substr(front, s.text.size() - front - back);
      if (s.text.empty()) continue;
    }
    out.push_back(std::move(s));
  }
  return out;
}

// --- expression tokens ---------------------------------------------------------

struct Tok {
  enum Type { kName, kString, kInt, kFloat, kOp, kEnd } type = kEnd;
  std::string s;
};

std::vector<Tok> tokenize(const std::string& src, int line) {
  std::vector<Tok> out;
  std::size_t i = 0;
  while (i < src.size()) {
    const char c = src[i];
    if (kWhitespace.find(c) != std::string_view::npos) { ++i; continue; }
    if (std::isalpha(static_cast<unsigned char>(c)) || c == '_') {
      std::size_t j = i + 1;
      while (j < src.size() && (std::isalnum(static_cast<unsigned char>(src[j])) || src[j] == '_')) ++j;
      out.push_back({Tok::kName, src.substr(i, j - i)});
      i = j;
      continue;
    }
    if (std::isdigit(static_cast<unsigned char>(c))) {
      std::size_t j = i;
      bool real = false;
      while (j < src.size() && (std::isdigit(static_cast<unsigned char>(src[j])) || src[j] == '_')) ++j;
      if (j + 1 < src.size() && src[j] == '.' && std::isdigit(static_cast<unsigned char>(src[j + 1]))) {
        real = true;
        ++j;
        while (j < src.size() && std::isdigit(static_cast<unsigned char>(src[j]))) ++j;
      }
      if (j < src.size() && (src[j] == 'e' || src[j] == 'E')) {
        std::size_t k = j + 1;
        if (k < src.size() && (src[k] == '+' || src[k] == '-')) ++k;
        if (k < src.size() && std::isdigit(static_cast<unsigned char>(src[k]))) {
          real = true;
          j = k;
          while (j < src.size() && std::isdigit(static_cast<unsigned char>(src[j]))) ++j;
        }
      }
      std::string digits = src.substr(i, j - i);
      digits.erase(std::remove(digits.begin(), digits.end(), '_'), digits.end());
      out.push_back({real ? Tok::kFloat : Tok::kInt, digits});
      i = j;
      continue;
    }
    if (c == '\'' || c == '"') {
      std::string value;
      std::size_t j = i + 1;
      for (;; ++j) {
        if (j >= src.size()) fail(line, "unterminated string literal");
        const char d = src[j];
        if (d == c) break;
        if (d != '\\' || j + 1 >= src.size()) { value += d; continue; }
        const char e = src[++j];
        switch (e) {
          case 'n': value += '\n'; break;
          case 't': value += '\t'; break;
          case 'r': value += '\r'; break;
          case '0': value += '\0'; break;
          case '\\': value += '\\'; break;
          case '\'': value += '\''; break;
          case '"': value += '"'; break;
          case '\n': break;
          case 'u': case 'x': {
            const std::size_t width = e == 'u' ? 4 : 2;
            if (j + width >= src.size()) fail(line, "truncated escape");
            const auto cp = static_cast<std::uint32_t>(std::stoul(src.substr(j + 1, width), nullptr, 16));
            value += encode_utf8({cp});
            j += width;
            break;
          }
          default: value += '\\'; value += e; break;
        }
      }
      out.push_back({Tok::kString, value});
      i = j + 1;
      continue;
    }
    static constexpr std::string_view kTwo[] = {"//", "**", "==", "!=", "<=", ">="};
    bool matched = false;
    for (std::string_view two : kTwo) {
      if (src.compare(i, 2, two) == 0) {
        out.push_back({Tok::kOp, std::string(two)});
        i += 2;
        matched = true;
        break;
      }
    }
    if (matched) continue;
    if (std::string_view("+-*/%~<>=()[]{},.:|").find(c) != std::string_view::npos) {
      out.push_back({Tok::kOp, std::string(1, c)});
      ++i;
      continue;
    }
    fail(line, std::string("unexpected character '") + c + "'");
  }
  out.push_back({Tok::kEnd, ""});
  return out;
}

class ExprParser {
 public:
  ExprParser(std::vector<Tok> toks, int line) : toks_(std::move(toks)), line_(line) {}

  const Tok& peek(std::size_t ahead = 0) const {
    return toks_[std::min(pos_ + ahead, toks_.size() - 1)];
  }
  bool at_end() const { return peek().type == Tok::kEnd; }
  bool is_op(std::string_view s, std::size_t ahead = 0) const {
    return peek(ahead).type == Tok::kOp && peek(ahead).s == s;
  }
  bool is_name(std::string_view s, std::size_t ahead = 0) const {
    return peek(ahead).type == Tok::kName && peek(ahead).s == s;
  }
  Tok next() { Tok t = peek(); if (pos_ < toks_.size() - 1) ++pos_; return t; }
  void expect_op(std::string_view s) {
    if (!is_op(s)) fail(line_, "expected '" + std::string(s) + "' but found '" + peek().s + "'");
    next();
  }
  std::string expect_name() {
    if (peek().type != Tok::kName) fail(line_, "expected a name but found '" + peek().s + "'");
    return next().s;
  }
  void expect_end() {
    if (!at_end()) fail(line_, "unexpected '" + peek().s + "'");
  }

  ExprPtr make(Op op) {
    auto e = std::make_unique<Expr>();
    e->op = op;
    e->line = line_;
    return e;
  }

  ExprPtr expression(bool with_condition = true) {
    ExprPtr e = parse_or();
    while (with_condition && is_name("if")) {
      next();
      ExprPtr cond = parse_or();
      ExprPtr otherwise;
      if (is_name("else")) {
        next();
        otherwise = expression();
      }
      ExprPtr c = make(Op::kCond);
      c->args.push_back(std::move(e));
      c->args.push_back(std::move(cond));
      c->args.push_back(std::move(otherwise));
      e = std::move(c);
    }
    return e;
  }

  // `a, b` without parentheses, as for and set statements write their values.
  ExprPtr tuple_or_expression() {
    ExprPtr first = expression();
    if (!is_op(",")) return first;
    ExprPtr t = make(Op::kTuple);
    t->args.push_back(std::move(first));
    while (is_op(",")) {
      next();
      if (at_end()) break;
      t->args.push_back(expression());
    }
    return t;
  }

 private:
  ExprPtr binary(Op op, std::string name, ExprPtr a, ExprPtr b) {
    ExprPtr e = make(op);
    e->name = std::move(name);
    e->args.push_back(std::move(a));
    e->args.push_back(std::move(b));
    return e;
  }

  ExprPtr parse_or() {
    ExprPtr e = parse_and();
    while (is_name("or")) { next(); e = binary(Op::kOr, "or", std::move(e), parse_and()); }
    return e;
  }
  ExprPtr parse_and() {
    ExprPtr e = parse_not();
    while (is_name("and")) { next(); e = binary(Op::kAnd, "and", std::move(e), parse_not()); }
    return e;
  }
  ExprPtr parse_not() {
    if (is_name("not")) {
      next();
      ExprPtr e = make(Op::kNot);
      e->args.push_back(parse_not());
      return e;
    }
    return parse_compare();
  }
  ExprPtr parse_compare() {
    ExprPtr e = parse_math1();
    for (;;) {
      std::string op;
      if (peek().type == Tok::kOp &&
          (peek().s == "==" || peek().s == "!=" || peek().s == "<" || peek().s == ">" ||
           peek().s == "<=" || peek().s == ">=")) {
        op = next().s;
      } else if (is_name("in")) {
        next();
        op = "in";
      } else if (is_name("not") && is_name("in", 1)) {
        next();
        next();
        op = "not in";
      } else {
        break;
      }
      e = binary(Op::kBinary, op, std::move(e), parse_math1());
    }
    return e;
  }
  ExprPtr parse_math1() {
    ExprPtr e = parse_concat();
    while (is_op("+") || is_op("-")) {
      const std::string op = next().s;
      e = binary(Op::kBinary, op, std::move(e), parse_concat());
    }
    return e;
  }
  ExprPtr parse_concat() {
    ExprPtr e = parse_math2();
    while (is_op("~")) { next(); e = binary(Op::kBinary, "~", std::move(e), parse_math2()); }
    return e;
  }
  ExprPtr parse_math2() {
    ExprPtr e = parse_pow();
    while (is_op("*") || is_op("/") || is_op("//") || is_op("%")) {
      const std::string op = next().s;
      e = binary(Op::kBinary, op, std::move(e), parse_pow());
    }
    return e;
  }
  ExprPtr parse_pow() {
    ExprPtr e = parse_unary();
    while (is_op("**")) { next(); e = binary(Op::kBinary, "**", std::move(e), parse_unary()); }
    return e;
  }
  ExprPtr parse_unary() {
    if (is_op("-") || is_op("+")) {
      const bool neg = next().s == "-";
      ExprPtr e = make(neg ? Op::kNeg : Op::kPos);
      e->args.push_back(parse_unary());
      return e;
    }
    ExprPtr e = postfix(primary());
    return filters(std::move(e));
  }

  void call_arguments(Expr& into) {
    expect_op("(");
    while (!is_op(")")) {
      if (peek().type == Tok::kName && is_op("=", 1)) {
        std::string key = next().s;
        next();
        into.kwargs.emplace_back(std::move(key), expression());
      } else {
        into.args.push_back(expression());
      }
      if (!is_op(",")) break;
      next();
    }
    expect_op(")");
  }

  ExprPtr primary() {
    const Tok t = peek();
    if (t.type == Tok::kName) {
      next();
      ExprPtr e = make(Op::kLiteral);
      if (t.s == "true" || t.s == "True") { e->literal = true; return e; }
      if (t.s == "false" || t.s == "False") { e->literal = false; return e; }
      if (t.s == "none" || t.s == "None") { e->literal = nullptr; return e; }
      e->op = Op::kName;
      e->name = t.s;
      return e;
    }
    if (t.type == Tok::kString) {
      std::string value;
      while (peek().type == Tok::kString) value += next().s;
      ExprPtr e = make(Op::kLiteral);
      e->literal = value;
      return e;
    }
    if (t.type == Tok::kInt || t.type == Tok::kFloat) {
      next();
      ExprPtr e = make(Op::kLiteral);
      if (t.type == Tok::kInt) e->literal = static_cast<std::int64_t>(std::stoll(t.s));
      else e->literal = std::stod(t.s);
      return e;
    }
    if (is_op("(")) {
      next();
      if (is_op(")")) { next(); return make(Op::kTuple); }
      ExprPtr first = expression();
      if (!is_op(",")) {
        expect_op(")");
        return first;
      }
      ExprPtr tuple = make(Op::kTuple);
      tuple->args.push_back(std::move(first));
      while (is_op(",")) {
        next();
        if (is_op(")")) break;
        tuple->args.push_back(expression());
      }
      expect_op(")");
      return tuple;
    }
    if (is_op("[")) {
      next();
      ExprPtr list = make(Op::kList);
      while (!is_op("]")) {
        list->args.push_back(expression());
        if (!is_op(",")) break;
        next();
      }
      expect_op("]");
      return list;
    }
    if (is_op("{")) {
      next();
      ExprPtr dict = make(Op::kDict);
      while (!is_op("}")) {
        dict->args.push_back(expression());
        expect_op(":");
        dict->args.push_back(expression());
        if (!is_op(",")) break;
        next();
      }
      expect_op("}");
      return dict;
    }
    fail(line_, "unexpected '" + t.s + "' in an expression");
  }

  ExprPtr postfix(ExprPtr e) {
    for (;;) {
      if (is_op(".")) {
        next();
        ExprPtr a = make(Op::kAttr);
        if (peek().type == Tok::kInt) {
          // x.0 is x[0].
          a->op = Op::kItem;
          a->args.push_back(std::move(e));
          ExprPtr index = make(Op::kLiteral);
          index->literal = static_cast<std::int64_t>(std::stoll(next().s));
          a->args.push_back(std::move(index));
        } else {
          a->name = expect_name();
          a->args.push_back(std::move(e));
        }
        e = std::move(a);
      } else if (is_op("[")) {
        next();
        ExprPtr start;
        if (!is_op(":")) start = expression();
        if (is_op(":")) {
          next();
          ExprPtr s = make(Op::kSlice);
          s->args.push_back(std::move(e));
          s->args.push_back(std::move(start));
          ExprPtr stop, step;
          if (!is_op(":") && !is_op("]")) stop = expression();
          if (is_op(":")) {
            next();
            if (!is_op("]")) step = expression();
          }
          s->args.push_back(std::move(stop));
          s->args.push_back(std::move(step));
          e = std::move(s);
        } else {
          ExprPtr item = make(Op::kItem);
          item->args.push_back(std::move(e));
          item->args.push_back(std::move(start));
          e = std::move(item);
        }
        expect_op("]");
      } else if (is_op("(")) {
        ExprPtr call = make(Op::kCall);
        call->args.push_back(std::move(e));
        call_arguments(*call);
        e = std::move(call);
      } else {
        return e;
      }
    }
  }

  ExprPtr filters(ExprPtr e) {
    for (;;) {
      if (is_op("|")) {
        next();
        ExprPtr f = make(Op::kFilter);
        f->name = expect_name();
        f->args.push_back(std::move(e));
        if (is_op("(")) call_arguments(*f);
        e = std::move(f);
      } else if (is_name("is")) {
        next();
        ExprPtr t = make(Op::kTest);
        if (is_name("not")) { next(); t->negated = true; }
        if (peek().type != Tok::kName) fail(line_, "expected a test name after 'is'");
        t->name = next().s;
        t->args.push_back(std::move(e));
        if (is_op("(")) {
          call_arguments(*t);
        } else {
          const Tok& n = peek();
          const bool argument =
              n.type == Tok::kString || n.type == Tok::kInt || n.type == Tok::kFloat ||
              (n.type == Tok::kName && n.s != "and" && n.s != "or" && n.s != "else" &&
               n.s != "if" && n.s != "not" && n.s != "in" && n.s != "is");
          if (argument) t->args.push_back(postfix(primary()));
        }
        e = std::move(t);
      } else {
        return e;
      }
    }
  }

  std::vector<Tok> toks_;
  std::size_t pos_ = 0;
  int line_;
};

// --- statements ----------------------------------------------------------------

class Parser {
 public:
  explicit Parser(std::vector<Segment> segments) : segs_(std::move(segments)) {}

  Body parse() {
    std::string end;
    Body body = block({}, end);
    if (!end.empty()) fail(last_line_, "unexpected '" + end + "'");
    return body;
  }

 private:
  // Parses until one of `ends` opens a tag; that keyword goes to `end` and the
  // tag's remaining tokens stay in `pending_`.
  Body block(const std::set<std::string>& ends, std::string& end) {
    Body out;
    while (pos_ < segs_.size()) {
      Segment& s = segs_[pos_++];
      last_line_ = s.line;
      if (s.kind == SegKind::kText) {
        auto n = std::make_unique<Node>();
        n->kind = NodeKind::kText;
        n->line = s.line;
        n->text = std::move(s.text);
        out.push_back(std::move(n));
        continue;
      }
      if (s.kind == SegKind::kExpr) {
        ExprParser p(tokenize(s.text, s.line), s.line);
        auto n = std::make_unique<Node>();
        n->kind = NodeKind::kOutput;
        n->line = s.line;
        n->expr = p.expression();
        p.expect_end();
        out.push_back(std::move(n));
        continue;
      }
      pending_ = std::make_unique<ExprParser>(tokenize(s.text, s.line), s.line);
      const std::string keyword = pending_->expect_name();
      if (ends.contains(keyword)) {
        end = keyword;
        return out;
      }
      out.push_back(statement(keyword, s.line));
    }
    end.clear();
    if (!ends.empty()) fail(last_line_, "missing {% " + *ends.begin() + " %}");
    return out;
  }

  NodePtr statement(const std::string& keyword, int line) {
    auto n = std::make_unique<Node>();
    n->line = line;
    ExprParser& p = *pending_;
    std::string end;
    if (keyword == "if") {
      n->kind = NodeKind::kIf;
      ExprPtr cond = p.expression();
      p.expect_end();
      for (;;) {
        Body body = block({"elif", "else", "endif"}, end);
        n->branches.emplace_back(std::move(cond), std::move(body));
        if (end == "elif") {
          cond = pending_->expression();
          pending_->expect_end();
          continue;
        }
        if (end == "else") {
          pending_->expect_end();
          n->else_body = block({"endif"}, end);
        }
        pending_->expect_end();
        return n;
      }
    }
    if (keyword == "for") {
      n->kind = NodeKind::kFor;
      n->targets.push_back(p.expect_name());
      while (p.is_op(",")) {
        p.next();
        n->targets.push_back(p.expect_name());
      }
      if (!p.is_name("in")) fail(line, "expected 'in' in a for statement");
      p.next();
      n->expr = p.expression(false);
      if (p.is_name("if")) {
        p.next();
        n->filter = p.expression(false);
      }
      if (p.is_name("recursive")) fail(line, "recursive for loops are not supported");
      p.expect_end();
      n->body = block({"else", "endfor"}, end);
      if (end == "else") {
        pending_->expect_end();
        n->else_body = block({"endfor"}, end);
      }
      pending_->expect_end();
      return n;
    }
    if (keyword == "set") {
      std::string target = p.expect_name();
      if (p.is_op(".")) {
        p.next();
        n->attribute = p.expect_name();
      }
      n->targets.push_back(std::move(target));
      while (n->attribute.empty() && p.is_op(",")) {
        p.next();
        n->targets.push_back(p.expect_name());
      }
      if (p.at_end()) {
        if (n->targets.size() != 1 || !n->attribute.empty()) fail(line, "invalid block set");
        n->kind = NodeKind::kSetBlock;
        n->body = block({"endset"}, end);
        pending_->expect_end();
        return n;
      }
      n->kind = NodeKind::kSet;
      p.expect_op("=");
      n->expr = p.tuple_or_expression();
      p.expect_end();
      return n;
    }
    if (keyword == "macro") {
      n->kind = NodeKind::kMacro;
      n->text = p.expect_name();
      p.expect_op("(");
      while (!p.is_op(")")) {
        n->params.push_back(p.expect_name());
        if (p.is_op("=")) {
          p.next();
          n->defaults.push_back(p.expression());
        } else {
          n->defaults.push_back(nullptr);
        }
        if (!p.is_op(",")) break;
        p.next();
      }
      p.expect_op(")");
      p.expect_end();
      n->body = block({"endmacro"}, end);
      return n;
    }
    if (keyword == "break" || keyword == "continue") {
      p.expect_end();
      n->kind = keyword == "break" ? NodeKind::kBreak : NodeKind::kContinue;
      return n;
    }
    if (keyword == "generation") {
      // transformers' assistant-mask marker renders its body unchanged.
      p.expect_end();
      n->kind = NodeKind::kBlock;
      n->body = block({"endgeneration"}, end);
      return n;
    }
    fail(line, "unsupported statement '{% " + keyword + " %}'");
  }

  std::vector<Segment> segs_;
  std::size_t pos_ = 0;
  int last_line_ = 1;
  std::unique_ptr<ExprParser> pending_;
};

// --- values --------------------------------------------------------------------

struct Value {
  enum class Kind { kUndefined, kData, kNamespace, kMacro, kFunction };
  Kind kind = Kind::kUndefined;
  ojson data;
  std::shared_ptr<ojson> ns;
  const Node* macro = nullptr;
  std::string function;

  static Value undefined() { return {}; }
  static Value of(ojson j) {
    Value v;
    v.kind = Kind::kData;
    v.data = std::move(j);
    return v;
  }
  bool defined() const { return kind != Kind::kUndefined; }
  bool is_data() const { return kind == Kind::kData; }
  bool is_string() const { return is_data() && data.is_string(); }
};

std::string python_float(double d) {
  if (std::isnan(d)) return "nan";
  if (std::isinf(d)) return d > 0 ? "inf" : "-inf";
  ojson j = d;
  std::string s = j.dump();
  if (s.find_first_of(".eE") == std::string::npos) s += ".0";
  return s;
}

std::string python_repr(const ojson& j);

std::string python_str(const ojson& j) {
  if (j.is_string()) return j.get<std::string>();
  if (j.is_null()) return "None";
  if (j.is_boolean()) return j.get<bool>() ? "True" : "False";
  if (j.is_number_float()) return python_float(j.get<double>());
  if (j.is_number()) return j.dump();
  return python_repr(j);
}

std::string python_repr(const ojson& j) {
  if (j.is_string()) {
    const std::string s = j.get<std::string>();
    const char quote = s.find('\'') != std::string::npos && s.find('"') == std::string::npos ? '"' : '\'';
    std::string out(1, quote);
    for (char c : s) {
      if (c == '\\') out += "\\\\";
      else if (c == '\n') out += "\\n";
      else if (c == '\t') out += "\\t";
      else if (c == '\r') out += "\\r";
      else if (c == quote) { out += '\\'; out += c; }
      else out += c;
    }
    return out + quote;
  }
  if (j.is_array()) {
    std::string out = "[";
    for (std::size_t i = 0; i < j.size(); ++i) {
      if (i) out += ", ";
      out += python_repr(j[i]);
    }
    return out + "]";
  }
  if (j.is_object()) {
    std::string out = "{";
    bool first = true;
    for (const auto& [k, v] : j.items()) {
      if (!first) out += ", ";
      first = false;
      out += python_repr(ojson(k)) + ": " + python_repr(v);
    }
    return out + "}";
  }
  return python_str(j);
}

// json.dumps with transformers' defaults: ensure_ascii off, ", " and ": ".
void to_json_text(const ojson& j, std::string& out, int indent, int depth) {
  const auto newline = [&](int level) {
    out += '\n';
    out.append(static_cast<std::size_t>(indent * level), ' ');
  };
  if (j.is_array()) {
    if (j.empty()) { out += "[]"; return; }
    out += '[';
    for (std::size_t i = 0; i < j.size(); ++i) {
      if (i) out += indent >= 0 ? "," : ", ";
      if (indent >= 0) newline(depth + 1);
      to_json_text(j[i], out, indent, depth + 1);
    }
    if (indent >= 0) newline(depth);
    out += ']';
    return;
  }
  if (j.is_object()) {
    if (j.empty()) { out += "{}"; return; }
    out += '{';
    bool first = true;
    for (const auto& [k, v] : j.items()) {
      if (!first) out += indent >= 0 ? "," : ", ";
      first = false;
      if (indent >= 0) newline(depth + 1);
      out += ojson(k).dump(-1, ' ', false, ojson::error_handler_t::replace);
      out += ": ";
      to_json_text(v, out, indent, depth + 1);
    }
    if (indent >= 0) newline(depth);
    out += '}';
    return;
  }
  if (j.is_number_float()) {
    out += python_float(j.get<double>());
    return;
  }
  out += j.dump(-1, ' ', false, ojson::error_handler_t::replace);
}

std::string stringify(const Value& v) {
  switch (v.kind) {
    case Value::Kind::kUndefined: return "";
    case Value::Kind::kData: return python_str(v.data);
    case Value::Kind::kNamespace: return "<Namespace>";
    case Value::Kind::kMacro: return "<Macro " + v.macro->text + ">";
    case Value::Kind::kFunction: return "<function " + v.function + ">";
  }
  return "";
}

bool truthy(const Value& v) {
  if (v.kind == Value::Kind::kUndefined) return false;
  if (v.kind != Value::Kind::kData) return true;
  const ojson& j = v.data;
  if (j.is_null()) return false;
  if (j.is_boolean()) return j.get<bool>();
  if (j.is_number_integer()) return j.get<std::int64_t>() != 0;
  if (j.is_number()) return j.get<double>() != 0.0;
  // json::empty() is false for every string; a string is falsy when it has no characters.
  if (j.is_string()) return !j.get_ref<const std::string&>().empty();
  if (j.is_array() || j.is_object()) return !j.empty();
  return true;
}

bool is_number(const Value& v) { return v.is_data() && v.data.is_number(); }
bool is_integer(const Value& v) {
  return v.is_data() && (v.data.is_number_integer() || v.data.is_boolean());
}
double number(const Value& v, int line) {
  if (v.is_data() && v.data.is_boolean()) return v.data.get<bool>() ? 1.0 : 0.0;
  if (!is_number(v)) fail(line, "expected a number, got '" + stringify(v) + "'");
  return v.data.get<double>();
}
std::int64_t integer(const Value& v, int line) {
  if (v.is_data() && v.data.is_boolean()) return v.data.get<bool>() ? 1 : 0;
  if (v.is_data() && v.data.is_number_integer()) return v.data.get<std::int64_t>();
  if (is_number(v)) return static_cast<std::int64_t>(v.data.get<double>());
  fail(line, "expected an integer, got '" + stringify(v) + "'");
}
const std::string& text_of(const Value& v, int line, const char* what) {
  if (!v.is_string()) fail(line, std::string(what) + " needs a string, got '" + stringify(v) + "'");
  return v.data.get_ref<const std::string&>();
}

bool equal(const Value& a, const Value& b) {
  if (!a.defined() || !b.defined()) return !a.defined() && !b.defined();
  if (a.kind != b.kind) return false;
  if (a.kind == Value::Kind::kNamespace) return a.ns == b.ns;
  if (a.kind == Value::Kind::kMacro) return a.macro == b.macro;
  if (a.kind == Value::Kind::kFunction) return a.function == b.function;
  const bool an = a.data.is_number() || a.data.is_boolean();
  const bool bn = b.data.is_number() || b.data.is_boolean();
  if (an && bn) return number(a, 0) == number(b, 0);
  return a.data == b.data;
}

// The sequence a for loop or a filter walks: list items, mapping keys, string
// characters.
std::vector<ojson> items_of(const Value& v, int line) {
  if (!v.defined()) return {};
  if (!v.is_data()) fail(line, "cannot iterate over '" + stringify(v) + "'");
  const ojson& j = v.data;
  std::vector<ojson> out;
  if (j.is_array()) {
    for (const auto& e : j) out.push_back(e);
  } else if (j.is_object()) {
    for (const auto& [k, e] : j.items()) out.push_back(k);
  } else if (j.is_string()) {
    for (std::uint32_t cp : decode_utf8(j.get<std::string>())) out.push_back(encode_utf8({cp}));
  } else if (j.is_null()) {
    fail(line, "cannot iterate over None");
  } else {
    fail(line, "cannot iterate over '" + stringify(v) + "'");
  }
  return out;
}

std::int64_t normalize_index(std::int64_t i, std::int64_t size) { return i < 0 ? i + size : i; }

// Python slice semantics over n elements.
std::vector<std::int64_t> slice_indices(std::int64_t n, std::optional<std::int64_t> start,
                                        std::optional<std::int64_t> stop,
                                        std::optional<std::int64_t> step_in, int line) {
  const std::int64_t step = step_in.value_or(1);
  if (step == 0) fail(line, "slice step cannot be zero");
  std::vector<std::int64_t> out;
  auto clamp = [&](std::int64_t v, std::int64_t lo, std::int64_t hi) { return std::max(lo, std::min(v, hi)); };
  if (step > 0) {
    std::int64_t b = start ? normalize_index(*start, n) : 0;
    std::int64_t e = stop ? normalize_index(*stop, n) : n;
    b = clamp(b, 0, n);
    e = clamp(e, 0, n);
    for (std::int64_t i = b; i < e; i += step) out.push_back(i);
  } else {
    std::int64_t b = start ? normalize_index(*start, n) : n - 1;
    std::int64_t e = stop ? normalize_index(*stop, n) : -1;
    b = clamp(b, -1, n - 1);
    e = clamp(e, -1, n - 1);
    for (std::int64_t i = b; i > e; i += step) out.push_back(i);
  }
  return out;
}

std::string strip(const std::string& s, const std::optional<std::string>& chars, bool left, bool right) {
  const std::string set = chars ? *chars : std::string(kWhitespace);
  std::size_t b = 0, e = s.size();
  if (left) while (b < e && set.find(s[b]) != std::string::npos) ++b;
  if (right) while (e > b && set.find(s[e - 1]) != std::string::npos) --e;
  return s.substr(b, e - b);
}

std::string replace_all(std::string s, const std::string& from, const std::string& to, std::int64_t count) {
  if (from.empty()) return s;
  std::size_t at = 0;
  for (std::int64_t done = 0; count < 0 || done < count; ++done) {
    at = s.find(from, at);
    if (at == std::string::npos) break;
    s.replace(at, from.size(), to);
    at += to.size();
  }
  return s;
}

ojson split(const std::string& s, const std::optional<std::string>& sep, std::int64_t max_split) {
  ojson out = ojson::array();
  if (!sep) {
    std::size_t i = 0;
    std::int64_t splits = 0;
    while (i < s.size()) {
      while (i < s.size() && kWhitespace.find(s[i]) != std::string_view::npos) ++i;
      if (i >= s.size()) break;
      if (max_split >= 0 && splits >= max_split) {
        out.push_back(strip(s.substr(i), std::nullopt, false, true));
        return out;
      }
      std::size_t j = i;
      while (j < s.size() && kWhitespace.find(s[j]) == std::string_view::npos) ++j;
      out.push_back(s.substr(i, j - i));
      ++splits;
      i = j;
    }
    return out;
  }
  if (sep->empty()) throw Error("empty separator");
  std::size_t i = 0;
  std::int64_t splits = 0;
  for (;;) {
    const std::size_t j = (max_split >= 0 && splits >= max_split) ? std::string::npos : s.find(*sep, i);
    if (j == std::string::npos) {
      out.push_back(s.substr(i));
      return out;
    }
    out.push_back(s.substr(i, j - i));
    ++splits;
    i = j + sep->size();
  }
}

// --- evaluation ----------------------------------------------------------------

struct Frame {
  std::unordered_map<std::string, Value> vars;
  Frame* parent = nullptr;
  Value lookup(const std::string& name) const {
    for (const Frame* f = this; f != nullptr; f = f->parent) {
      const auto it = f->vars.find(name);
      if (it != f->vars.end()) return it->second;
    }
    return Value::undefined();
  }
};

enum class Flow { kNormal, kBreak, kContinue };

struct Args {
  std::vector<Value> positional;
  std::vector<std::pair<std::string, Value>> named;
  const Value* get(std::size_t index, std::string_view name) const {
    for (const auto& [k, v] : named)
      if (k == name) return &v;
    return index < positional.size() ? &positional[index] : nullptr;
  }
};

class Renderer {
 public:
  explicit Renderer(Frame& root) : root_(root) {}

  Flow exec(const Body& body, Frame& frame, std::string& out) {
    for (const NodePtr& n : body) {
      const Flow f = exec(*n, frame, out);
      if (f != Flow::kNormal) return f;
    }
    return Flow::kNormal;
  }

 private:
  Flow exec(const Node& n, Frame& frame, std::string& out) {
    switch (n.kind) {
      case NodeKind::kText:
        out += n.text;
        return Flow::kNormal;
      case NodeKind::kOutput:
        out += stringify(eval(*n.expr, frame));
        return Flow::kNormal;
      case NodeKind::kBlock:
        return exec(n.body, frame, out);
      case NodeKind::kIf:
        for (const auto& [cond, body] : n.branches)
          if (truthy(eval(*cond, frame))) return exec(body, frame, out);
        return exec(n.else_body, frame, out);
      case NodeKind::kFor:
        return loop(n, frame, out);
      case NodeKind::kSet: {
        Value v = eval(*n.expr, frame);
        if (!n.attribute.empty()) {
          Value target = frame.lookup(n.targets[0]);
          if (target.kind != Value::Kind::kNamespace)
            fail(n.line, "cannot set an attribute of '" + n.targets[0] + "', which is not a namespace");
          if (v.kind == Value::Kind::kUndefined) (*target.ns)[n.attribute] = nullptr;
          else if (v.is_data()) (*target.ns)[n.attribute] = v.data;
          else fail(n.line, "a namespace attribute holds data only");
          return Flow::kNormal;
        }
        assign(n.targets, std::move(v), frame, n.line);
        return Flow::kNormal;
      }
      case NodeKind::kSetBlock: {
        std::string captured;
        Frame inner;
        inner.parent = &frame;
        (void)exec(n.body, inner, captured);
        frame.vars[n.targets[0]] = Value::of(captured);
        return Flow::kNormal;
      }
      case NodeKind::kMacro: {
        Value m;
        m.kind = Value::Kind::kMacro;
        m.macro = &n;
        frame.vars[n.text] = m;
        return Flow::kNormal;
      }
      case NodeKind::kBreak: return Flow::kBreak;
      case NodeKind::kContinue: return Flow::kContinue;
    }
    return Flow::kNormal;
  }

  void assign(const std::vector<std::string>& targets, Value v, Frame& frame, int line) {
    if (targets.size() == 1) {
      frame.vars[targets[0]] = std::move(v);
      return;
    }
    const auto items = items_of(v, line);
    if (items.size() != targets.size())
      fail(line, "cannot unpack " + std::to_string(items.size()) + " values into " +
                     std::to_string(targets.size()) + " names");
    for (std::size_t i = 0; i < targets.size(); ++i) frame.vars[targets[i]] = Value::of(items[i]);
  }

  Flow loop(const Node& n, Frame& frame, std::string& out) {
    const Value iterable = eval(*n.expr, frame);
    std::vector<ojson> all = items_of(iterable, n.line);
    if (n.filter) {
      std::vector<ojson> kept;
      for (auto& item : all) {
        Frame probe;
        probe.parent = &frame;
        assign(n.targets, Value::of(item), probe, n.line);
        if (truthy(eval(*n.filter, probe))) kept.push_back(std::move(item));
      }
      all = std::move(kept);
    }
    if (all.empty()) return exec(n.else_body, frame, out);
    const auto length = static_cast<std::int64_t>(all.size());
    for (std::int64_t i = 0; i < length; ++i) {
      Frame inner;
      inner.parent = &frame;
      assign(n.targets, Value::of(all[static_cast<std::size_t>(i)]), inner, n.line);
      ojson loop_var{{"index", i + 1}, {"index0", i}, {"revindex", length - i},
                     {"revindex0", length - i - 1}, {"first", i == 0},
                     {"last", i == length - 1}, {"length", length}};
      if (i > 0) loop_var["previtem"] = all[static_cast<std::size_t>(i - 1)];
      if (i + 1 < length) loop_var["nextitem"] = all[static_cast<std::size_t>(i + 1)];
      inner.vars["loop"] = Value::of(std::move(loop_var));
      const Flow f = exec(n.body, inner, out);
      if (f == Flow::kBreak) break;
    }
    return Flow::kNormal;
  }

  Args arguments(const Expr& e, std::size_t first, Frame& frame) {
    Args a;
    for (std::size_t i = first; i < e.args.size(); ++i) a.positional.push_back(eval(*e.args[i], frame));
    for (const auto& [k, v] : e.kwargs) a.named.emplace_back(k, eval(*v, frame));
    return a;
  }

  Value eval(const Expr& e, Frame& frame) {
    switch (e.op) {
      case Op::kLiteral: return Value::of(e.literal);
      case Op::kName: {
        Value v = frame.lookup(e.name);
        if (v.defined()) return v;
        static const std::set<std::string> kFunctions{"raise_exception", "namespace", "range",
                                                      "strftime_now", "dict"};
        if (kFunctions.contains(e.name)) {
          Value f;
          f.kind = Value::Kind::kFunction;
          f.function = e.name;
          return f;
        }
        return v;
      }
      case Op::kList:
      case Op::kTuple: {
        ojson list = ojson::array();
        for (const auto& item : e.args) list.push_back(data(eval(*item, frame), e.line));
        return Value::of(std::move(list));
      }
      case Op::kDict: {
        ojson dict = ojson::object();
        for (std::size_t i = 0; i + 1 < e.args.size(); i += 2) {
          const Value k = eval(*e.args[i], frame);
          dict[stringify(k)] = data(eval(*e.args[i + 1], frame), e.line);
        }
        return Value::of(std::move(dict));
      }
      case Op::kAttr: return attribute(eval(*e.args[0], frame), e.name);
      case Op::kItem: return item(eval(*e.args[0], frame), eval(*e.args[1], frame), e.line);
      case Op::kSlice: return slice(e, frame);
      case Op::kCall: return call(e, frame);
      case Op::kFilter: return filter(e, frame);
      case Op::kTest: {
        const bool r = test(e.name, eval(*e.args[0], frame), arguments(e, 1, frame), e.line);
        return Value::of(e.negated ? !r : r);
      }
      case Op::kNot: return Value::of(!truthy(eval(*e.args[0], frame)));
      case Op::kNeg:
      case Op::kPos: {
        const Value v = eval(*e.args[0], frame);
        if (e.op == Op::kPos) return v;
        if (is_integer(v)) return Value::of(-integer(v, e.line));
        return Value::of(-number(v, e.line));
      }
      case Op::kAnd: {
        Value a = eval(*e.args[0], frame);
        return truthy(a) ? eval(*e.args[1], frame) : a;
      }
      case Op::kOr: {
        Value a = eval(*e.args[0], frame);
        return truthy(a) ? a : eval(*e.args[1], frame);
      }
      case Op::kCond:
        if (truthy(eval(*e.args[1], frame))) return eval(*e.args[0], frame);
        return e.args[2] ? eval(*e.args[2], frame) : Value::undefined();
      case Op::kBinary: return binary(e, frame);
    }
    return Value::undefined();
  }

  static ojson data(const Value& v, int line) {
    if (v.kind == Value::Kind::kUndefined) return nullptr;
    if (v.kind == Value::Kind::kNamespace) return *v.ns;
    if (!v.is_data()) fail(line, "'" + stringify(v) + "' cannot be stored as data");
    return v.data;
  }

  static Value attribute(const Value& v, const std::string& name) {
    if (v.kind == Value::Kind::kNamespace) {
      const auto it = v.ns->find(name);
      return it == v.ns->end() ? Value::undefined() : Value::of(*it);
    }
    if (v.is_data() && v.data.is_object()) {
      const auto it = v.data.find(name);
      if (it != v.data.end()) return Value::of(*it);
    }
    return Value::undefined();
  }

  static Value item(const Value& v, const Value& key, int line) {
    if (v.kind == Value::Kind::kNamespace) return attribute(v, stringify(key));
    if (!v.is_data()) return Value::undefined();
    const ojson& j = v.data;
    if (j.is_object()) {
      const auto it = j.find(stringify(key));
      return it == j.end() ? Value::undefined() : Value::of(*it);
    }
    if (j.is_array() || j.is_string()) {
      if (!is_integer(key)) return Value::undefined();
      if (j.is_array()) {
        const auto n = static_cast<std::int64_t>(j.size());
        const std::int64_t i = normalize_index(integer(key, line), n);
        if (i < 0 || i >= n) return Value::undefined();
        return Value::of(j[static_cast<std::size_t>(i)]);
      }
      const auto cps = decode_utf8(j.get<std::string>());
      const auto n = static_cast<std::int64_t>(cps.size());
      const std::int64_t i = normalize_index(integer(key, line), n);
      if (i < 0 || i >= n) return Value::undefined();
      return Value::of(encode_utf8({cps[static_cast<std::size_t>(i)]}));
    }
    return Value::undefined();
  }

  Value slice(const Expr& e, Frame& frame) {
    const Value v = eval(*e.args[0], frame);
    std::optional<std::int64_t> bound[3];
    for (std::size_t k = 0; k < 3; ++k) {
      if (!e.args[k + 1]) continue;
      const Value b = eval(*e.args[k + 1], frame);
      if (b.is_data() && b.data.is_null()) continue;
      bound[k] = integer(b, e.line);
    }
    if (!v.is_data()) fail(e.line, "cannot slice '" + stringify(v) + "'");
    if (v.data.is_array()) {
      ojson out = ojson::array();
      for (std::int64_t i : slice_indices(static_cast<std::int64_t>(v.data.size()), bound[0], bound[1], bound[2], e.line))
        out.push_back(v.data[static_cast<std::size_t>(i)]);
      return Value::of(std::move(out));
    }
    if (v.data.is_string()) {
      const auto cps = decode_utf8(v.data.get<std::string>());
      std::vector<std::uint32_t> out;
      for (std::int64_t i : slice_indices(static_cast<std::int64_t>(cps.size()), bound[0], bound[1], bound[2], e.line))
        out.push_back(cps[static_cast<std::size_t>(i)]);
      return Value::of(encode_utf8(out));
    }
    fail(e.line, "cannot slice '" + stringify(v) + "'");
  }

  Value binary(const Expr& e, Frame& frame) {
    const Value a = eval(*e.args[0], frame);
    const Value b = eval(*e.args[1], frame);
    const std::string& op = e.name;
    if (op == "==") return Value::of(equal(a, b));
    if (op == "!=") return Value::of(!equal(a, b));
    if (op == "in" || op == "not in") {
      bool found = false;
      if (b.is_data()) {
        if (b.data.is_string())
          found = b.data.get<std::string>().find(text_of(a, e.line, "'in' over a string")) != std::string::npos;
        else if (b.data.is_array())
          found = std::any_of(b.data.begin(), b.data.end(), [&](const ojson& x) { return equal(a, Value::of(x)); });
        else if (b.data.is_object()) found = a.is_string() && b.data.contains(a.data.get<std::string>());
        else fail(e.line, "'in' needs a container on the right");
      } else if (b.kind == Value::Kind::kNamespace) {
        found = a.is_string() && b.ns->contains(a.data.get<std::string>());
      } else if (!b.defined()) {
        fail(e.line, "'in' over an undefined value");
      }
      return Value::of(op == "in" ? found : !found);
    }
    if (op == "~") return Value::of(stringify(a) + stringify(b));
    if (!a.defined() || !b.defined()) fail(e.line, "undefined value in '" + op + "'");
    if (op == "<" || op == ">" || op == "<=" || op == ">=") {
      int c = 0;
      if (a.is_string() && b.is_string()) {
        c = a.data.get<std::string>().compare(b.data.get<std::string>());
      } else {
        const double x = number(a, e.line), y = number(b, e.line);
        c = x < y ? -1 : x > y ? 1 : 0;
      }
      if (op == "<") return Value::of(c < 0);
      if (op == ">") return Value::of(c > 0);
      if (op == "<=") return Value::of(c <= 0);
      return Value::of(c >= 0);
    }
    if (op == "+") {
      if (a.is_string() && b.is_string()) return Value::of(a.data.get<std::string>() + b.data.get<std::string>());
      if (a.is_data() && b.is_data() && a.data.is_array() && b.data.is_array()) {
        ojson out = a.data;
        for (const auto& x : b.data) out.push_back(x);
        return Value::of(std::move(out));
      }
      if (a.is_string() || b.is_string())
        fail(e.line, "cannot add '" + stringify(a) + "' and '" + stringify(b) + "'");
    }
    const bool ints = is_integer(a) && is_integer(b);
    if (op == "+") return ints ? Value::of(integer(a, e.line) + integer(b, e.line)) : Value::of(number(a, e.line) + number(b, e.line));
    if (op == "-") return ints ? Value::of(integer(a, e.line) - integer(b, e.line)) : Value::of(number(a, e.line) - number(b, e.line));
    if (op == "*") {
      if (a.is_string() && is_integer(b)) {
        std::string out;
        for (std::int64_t i = 0; i < integer(b, e.line); ++i) out += a.data.get<std::string>();
        return Value::of(out);
      }
      return ints ? Value::of(integer(a, e.line) * integer(b, e.line)) : Value::of(number(a, e.line) * number(b, e.line));
    }
    if (op == "/") {
      const double y = number(b, e.line);
      if (y == 0.0) fail(e.line, "division by zero");
      return Value::of(number(a, e.line) / y);
    }
    if (op == "//" || op == "%") {
      if (ints) {
        const std::int64_t x = integer(a, e.line), y = integer(b, e.line);
        if (y == 0) fail(e.line, "division by zero");
        std::int64_t q = x / y, r = x % y;
        if (r != 0 && ((r < 0) != (y < 0))) { --q; r += y; }
        return Value::of(op == "//" ? q : r);
      }
      const double x = number(a, e.line), y = number(b, e.line);
      if (y == 0.0) fail(e.line, "division by zero");
      const double q = std::floor(x / y);
      return Value::of(op == "//" ? q : x - q * y);
    }
    if (op == "**") {
      if (ints && integer(b, e.line) >= 0) {
        std::int64_t r = 1;
        for (std::int64_t i = 0; i < integer(b, e.line); ++i) r *= integer(a, e.line);
        return Value::of(r);
      }
      return Value::of(std::pow(number(a, e.line), number(b, e.line)));
    }
    fail(e.line, "unsupported operator '" + op + "'");
  }

  Value call(const Expr& e, Frame& frame) {
    const Expr& callee = *e.args[0];
    if (callee.op == Op::kAttr) {
      const Value self = eval(*callee.args[0], frame);
      if (auto r = method(self, callee.name, arguments(e, 1, frame), e.line)) return *r;
      const Value f = attribute(self, callee.name);
      return invoke(f, arguments(e, 1, frame), e.line, callee.name);
    }
    const Value f = eval(callee, frame);
    return invoke(f, arguments(e, 1, frame), e.line, callee.op == Op::kName ? callee.name : "expression");
  }

  Value invoke(const Value& f, const Args& args, int line, const std::string& name) {
    if (f.kind == Value::Kind::kMacro) return call_macro(*f.macro, args, line);
    if (f.kind != Value::Kind::kFunction) fail(line, "'" + name + "' is not callable");
    if (f.function == "raise_exception") {
      const Value* m = args.get(0, "message");
      throw Raised(m != nullptr ? stringify(*m) : std::string("raise_exception"));
    }
    if (f.function == "namespace" || f.function == "dict") {
      auto ns = std::make_shared<ojson>(ojson::object());
      for (const Value& p : args.positional) {
        if (!p.is_data() || !p.data.is_object()) fail(line, f.function + "() takes a mapping");
        for (const auto& [k, v] : p.data.items()) (*ns)[k] = v;
      }
      for (const auto& [k, v] : args.named) (*ns)[k] = data(v, line);
      if (f.function == "dict") return Value::of(*ns);
      Value out;
      out.kind = Value::Kind::kNamespace;
      out.ns = std::move(ns);
      return out;
    }
    if (f.function == "range") {
      std::int64_t start = 0, stop = 0, step = 1;
      if (args.positional.size() == 1) {
        stop = integer(args.positional[0], line);
      } else if (args.positional.size() >= 2) {
        start = integer(args.positional[0], line);
        stop = integer(args.positional[1], line);
        if (args.positional.size() >= 3) step = integer(args.positional[2], line);
      } else {
        fail(line, "range() needs an argument");
      }
      if (step == 0) fail(line, "range() step cannot be zero");
      ojson out = ojson::array();
      for (std::int64_t i = start; step > 0 ? i < stop : i > stop; i += step) out.push_back(i);
      return Value::of(std::move(out));
    }
    if (f.function == "strftime_now") {
      const Value* fmt = args.get(0, "format");
      if (fmt == nullptr) fail(line, "strftime_now() needs a format");
      const std::string format = text_of(*fmt, line, "strftime_now");
      const std::time_t t = std::time(nullptr);
      std::tm tm{};
      localtime_r(&t, &tm);
      char buf[256];
      // The format is the template's, by design.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-nonliteral"
      const std::size_t n = std::strftime(buf, sizeof buf, format.c_str(), &tm);
#pragma GCC diagnostic pop
      return Value::of(std::string(buf, n));
    }
    fail(line, "unsupported function '" + f.function + "'");
  }

  Value call_macro(const Node& m, const Args& args, int line) {
    Frame inner;
    inner.parent = &root_;
    if (args.positional.size() > m.params.size())
      fail(line, "macro '" + m.text + "' takes " + std::to_string(m.params.size()) + " arguments");
    for (const auto& [k, v] : args.named)
      if (std::find(m.params.begin(), m.params.end(), k) == m.params.end())
        fail(line, "macro '" + m.text + "' has no parameter '" + k + "'");
    for (std::size_t i = 0; i < m.params.size(); ++i) {
      const Value* given = args.get(i, m.params[i]);
      if (given != nullptr) inner.vars[m.params[i]] = *given;
      else if (m.defaults[i]) inner.vars[m.params[i]] = eval(*m.defaults[i], inner);
      else inner.vars[m.params[i]] = Value::undefined();
    }
    std::string out;
    (void)exec(m.body, inner, out);
    return Value::of(out);
  }

  // String, mapping and loop methods. nullopt: not a method of this value.
  std::optional<Value> method(const Value& self, const std::string& name, const Args& args, int line) {
    const auto str_arg = [&](std::size_t i, std::string_view key) -> std::optional<std::string> {
      const Value* v = args.get(i, key);
      if (v == nullptr || !v->defined() || (v->is_data() && v->data.is_null())) return std::nullopt;
      return text_of(*v, line, name.c_str());
    };
    const auto int_arg = [&](std::size_t i, std::string_view key, std::int64_t fallback) {
      const Value* v = args.get(i, key);
      return v == nullptr ? fallback : integer(*v, line);
    };
    if (self.is_string()) {
      const std::string& s = self.data.get_ref<const std::string&>();
      if (name == "startswith" || name == "endswith") {
        const Value* p = args.get(0, "prefix");
        if (p == nullptr) fail(line, name + "() needs an argument");
        std::vector<std::string> options;
        if (p->is_data() && p->data.is_array())
          for (const auto& o : p->data) options.push_back(python_str(o));
        else
          options.push_back(text_of(*p, line, name.c_str()));
        for (const auto& o : options)
          if (name == "startswith" ? s.starts_with(o) : s.ends_with(o)) return Value::of(true);
        return Value::of(false);
      }
      if (name == "strip") return Value::of(strip(s, str_arg(0, "chars"), true, true));
      if (name == "lstrip") return Value::of(strip(s, str_arg(0, "chars"), true, false));
      if (name == "rstrip") return Value::of(strip(s, str_arg(0, "chars"), false, true));
      if (name == "split") return Value::of(split(s, str_arg(0, "sep"), int_arg(1, "maxsplit", -1)));
      if (name == "upper" || name == "lower") {
        std::string out = s;
        for (char& c : out) c = static_cast<char>(name == "upper" ? std::toupper(static_cast<unsigned char>(c))
                                                                  : std::tolower(static_cast<unsigned char>(c)));
        return Value::of(out);
      }
      if (name == "replace") {
        const auto from = str_arg(0, "old"), to = str_arg(1, "new");
        if (!from || !to) fail(line, "replace() needs two strings");
        return Value::of(replace_all(s, *from, *to, int_arg(2, "count", -1)));
      }
      if (name == "find" || name == "rfind" || name == "count") {
        const auto sub = str_arg(0, "sub");
        if (!sub) fail(line, name + "() needs a string");
        if (name == "count") {
          std::int64_t n = 0;
          if (!sub->empty())
            for (std::size_t at = s.find(*sub); at != std::string::npos; at = s.find(*sub, at + sub->size())) ++n;
          return Value::of(n);
        }
        const std::size_t at = name == "find" ? s.find(*sub) : s.rfind(*sub);
        if (at == std::string::npos) return Value::of(static_cast<std::int64_t>(-1));
        return Value::of(static_cast<std::int64_t>(decode_utf8(s.substr(0, at)).size()));
      }
      if (name == "join") {
        const Value* list = args.get(0, "iterable");
        if (list == nullptr) fail(line, "join() needs a list");
        std::string out;
        bool first = true;
        for (const auto& x : items_of(*list, line)) {
          if (!first) out += s;
          first = false;
          out += python_str(x);
        }
        return Value::of(out);
      }
      if (name == "title" || name == "capitalize") {
        std::string out = s;
        bool start = true;
        for (std::size_t i = 0; i < out.size(); ++i) {
          const auto c = static_cast<unsigned char>(out[i]);
          if (name == "capitalize") {
            out[i] = static_cast<char>(i == 0 ? std::toupper(c) : std::tolower(c));
          } else {
            out[i] = static_cast<char>(start ? std::toupper(c) : std::tolower(c));
            start = !std::isalpha(c);
          }
        }
        return Value::of(out);
      }
      if (name == "splitlines") {
        ojson out = ojson::array();
        std::size_t i = 0;
        while (i < s.size()) {
          const std::size_t j = s.find('\n', i);
          if (j == std::string::npos) { out.push_back(s.substr(i)); break; }
          std::string piece = s.substr(i, j - i);
          if (!piece.empty() && piece.back() == '\r') piece.pop_back();
          out.push_back(piece);
          i = j + 1;
        }
        return Value::of(std::move(out));
      }
      return std::nullopt;
    }
    if (self.is_data() && self.data.is_object()) {
      const ojson& j = self.data;
      if (name == "items") {
        ojson out = ojson::array();
        for (const auto& [k, v] : j.items()) out.push_back(ojson::array({k, v}));
        return Value::of(std::move(out));
      }
      if (name == "keys" || name == "values") {
        ojson out = ojson::array();
        for (const auto& [k, v] : j.items()) out.push_back(name == "keys" ? ojson(k) : v);
        return Value::of(std::move(out));
      }
      if (name == "get") {
        const auto key = str_arg(0, "key");
        if (!key) fail(line, "get() needs a key");
        const auto it = j.find(*key);
        if (it != j.end()) return Value::of(*it);
        const Value* fallback = args.get(1, "default");
        return fallback != nullptr ? *fallback : Value::of(nullptr);
      }
      if (name == "cycle" && j.contains("index0") && !j.contains("cycle")) {
        if (args.positional.empty()) fail(line, "loop.cycle() needs values");
        const auto i = static_cast<std::size_t>(j["index0"].get<std::int64_t>());
        return args.positional[i % args.positional.size()];
      }
    }
    return std::nullopt;
  }

  bool test(const std::string& name, const Value& v, const Args& args, int line) {
    const auto arg = [&](std::size_t i) -> const Value& {
      if (i >= args.positional.size()) fail(line, "test '" + name + "' needs an argument");
      return args.positional[i];
    };
    if (name == "defined") return v.defined();
    if (name == "undefined") return !v.defined();
    if (name == "none") return v.is_data() && v.data.is_null();
    if (name == "boolean") return v.is_data() && v.data.is_boolean();
    if (name == "true") return v.is_data() && v.data.is_boolean() && v.data.get<bool>();
    if (name == "false") return v.is_data() && v.data.is_boolean() && !v.data.get<bool>();
    if (name == "string") return v.is_string();
    if (name == "number") return v.is_data() && (v.data.is_number() || v.data.is_boolean());
    if (name == "integer") return v.is_data() && v.data.is_number_integer();
    if (name == "float") return v.is_data() && v.data.is_number_float();
    if (name == "mapping") return v.is_data() && v.data.is_object();
    if (name == "iterable" || name == "sequence")
      return v.is_data() && (v.data.is_array() || v.data.is_object() || v.data.is_string());
    if (name == "callable") return v.kind == Value::Kind::kMacro || v.kind == Value::Kind::kFunction;
    if (name == "odd") return integer(v, line) % 2 != 0;
    if (name == "even") return integer(v, line) % 2 == 0;
    if (name == "divisibleby") {
      const std::int64_t d = integer(arg(0), line);
      return d != 0 && integer(v, line) % d == 0;
    }
    if (name == "eq" || name == "equalto" || name == "sameas") return equal(v, arg(0));
    if (name == "ne") return !equal(v, arg(0));
    if (name == "lt" || name == "lessthan") return number(v, line) < number(arg(0), line);
    if (name == "gt" || name == "greaterthan") return number(v, line) > number(arg(0), line);
    if (name == "le") return number(v, line) <= number(arg(0), line);
    if (name == "ge") return number(v, line) >= number(arg(0), line);
    if (name == "in") {
      const Value& c = arg(0);
      if (c.is_data() && c.data.is_array())
        return std::any_of(c.data.begin(), c.data.end(), [&](const ojson& x) { return equal(v, Value::of(x)); });
      if (c.is_data() && c.data.is_object()) return v.is_string() && c.data.contains(v.data.get<std::string>());
      if (c.is_string()) return c.data.get<std::string>().find(text_of(v, line, "in")) != std::string::npos;
      return false;
    }
    if (name == "lower" || name == "upper") {
      const std::string& s = text_of(v, line, name.c_str());
      return std::all_of(s.begin(), s.end(), [&](char c) {
        return name == "lower" ? !std::isupper(static_cast<unsigned char>(c)) : !std::islower(static_cast<unsigned char>(c));
      });
    }
    fail(line, "unsupported test '" + name + "'");
  }

  Value filter(const Expr& e, Frame& frame) {
    const Value v = eval(*e.args[0], frame);
    return apply_filter(e.name, v, arguments(e, 1, frame), e.line);
  }

  Value apply_filter(const std::string& name, const Value& v, const Args& args, int line) {
    if (name == "default" || name == "d") {
      const Value* fallback = args.get(0, "default_value");
      const Value* boolean = args.get(1, "boolean");
      const bool use = !v.defined() || (boolean != nullptr && truthy(*boolean) && !truthy(v));
      if (!use) return v;
      return fallback != nullptr ? *fallback : Value::of("");
    }
    if (name == "safe" || name == "e" || name == "escape" || name == "forceescape") {
      if (name != "safe") fail(line, "unsupported filter '" + name + "' (autoescape is off in chat templates)");
      return v;
    }
    if (name == "string") return Value::of(stringify(v));
    if (name == "trim") {
      const Value* chars = args.get(0, "chars");
      std::optional<std::string> set;
      if (chars != nullptr && chars->defined()) set = text_of(*chars, line, "trim");
      return Value::of(strip(stringify(v), set, true, true));
    }
    if (name == "upper" || name == "lower" || name == "title" || name == "capitalize") {
      Args none;
      return *method(Value::of(stringify(v)), name, none, line);
    }
    if (name == "replace") {
      if (args.positional.size() < 2) fail(line, "replace needs two arguments");
      return Value::of(replace_all(stringify(v), stringify(args.positional[0]), stringify(args.positional[1]),
                                   args.positional.size() > 2 ? integer(args.positional[2], line) : -1));
    }
    if (name == "length" || name == "count") {
      if (!v.is_data()) {
        if (v.kind == Value::Kind::kNamespace) return Value::of(static_cast<std::int64_t>(v.ns->size()));
        if (!v.defined()) return Value::of(static_cast<std::int64_t>(0));
        fail(line, "length of '" + stringify(v) + "'");
      }
      if (v.data.is_string()) return Value::of(static_cast<std::int64_t>(decode_utf8(v.data.get<std::string>()).size()));
      if (v.data.is_array() || v.data.is_object()) return Value::of(static_cast<std::int64_t>(v.data.size()));
      fail(line, "length of '" + stringify(v) + "'");
    }
    if (name == "tojson") {
      const Value* indent = args.get(0, "indent");
      int width = -1;
      if (indent != nullptr && indent->is_data() && !indent->data.is_null())
        width = static_cast<int>(integer(*indent, line));
      std::string out;
      to_json_text(data(v, line), out, width, 0);
      return Value::of(out);
    }
    if (name == "items") {
      if (!v.defined()) return Value::of(ojson::array());
      if (v.kind == Value::Kind::kNamespace) return *method(Value::of(*v.ns), "items", {}, line);
      if (!v.is_data() || !v.data.is_object()) fail(line, "items needs a mapping, got '" + stringify(v) + "'");
      return *method(v, "items", {}, line);
    }
    if (name == "list") {
      ojson out = ojson::array();
      for (auto& x : items_of(v, line)) out.push_back(std::move(x));
      return Value::of(std::move(out));
    }
    if (name == "first" || name == "last") {
      const auto all = items_of(v, line);
      if (all.empty()) return Value::undefined();
      return Value::of(name == "first" ? all.front() : all.back());
    }
    if (name == "reverse") {
      if (v.is_string()) {
        auto cps = decode_utf8(v.data.get<std::string>());
        std::reverse(cps.begin(), cps.end());
        return Value::of(encode_utf8(cps));
      }
      auto all = items_of(v, line);
      std::reverse(all.begin(), all.end());
      ojson out = ojson::array();
      for (auto& x : all) out.push_back(std::move(x));
      return Value::of(std::move(out));
    }
    if (name == "join") {
      const Value* sep = args.get(0, "d");
      const Value* attr = args.get(1, "attribute");
      std::string out;
      bool first = true;
      for (const auto& x : items_of(v, line)) {
        if (!first && sep != nullptr) out += stringify(*sep);
        first = false;
        out += attr != nullptr ? stringify(attribute(Value::of(x), stringify(*attr))) : python_str(x);
      }
      return Value::of(out);
    }
    if (name == "int") {
      if (v.is_string()) {
        try { return Value::of(static_cast<std::int64_t>(std::stoll(strip(v.data.get<std::string>(), std::nullopt, true, true)))); }
        catch (const std::exception&) { return Value::of(static_cast<std::int64_t>(0)); }
      }
      return Value::of(is_number(v) || is_integer(v) ? integer(v, line) : 0);
    }
    if (name == "float") {
      if (v.is_string()) {
        try { return Value::of(std::stod(v.data.get<std::string>())); }
        catch (const std::exception&) { return Value::of(0.0); }
      }
      return Value::of(is_number(v) || is_integer(v) ? number(v, line) : 0.0);
    }
    if (name == "abs") return is_integer(v) ? Value::of(std::llabs(integer(v, line))) : Value::of(std::fabs(number(v, line)));
    if (name == "round") {
      const Value* precision = args.get(0, "precision");
      const double scale = std::pow(10.0, precision != nullptr ? number(*precision, line) : 0.0);
      return Value::of(std::round(number(v, line) * scale) / scale);
    }
    if (name == "indent") {
      const Value* width_arg = args.get(0, "width");
      const Value* first_arg = args.get(1, "first");
      const std::string pad = width_arg != nullptr && width_arg->is_string()
                                  ? width_arg->data.get<std::string>()
                                  : std::string(static_cast<std::size_t>(width_arg != nullptr ? integer(*width_arg, line) : 4), ' ');
      const bool first = first_arg != nullptr && truthy(*first_arg);
      const std::string s = stringify(v);
      std::string out;
      bool at_start = true, first_line = true;
      for (char c : s) {
        if (at_start && c != '\n' && (first || !first_line)) out += pad;
        out += c;
        at_start = c == '\n';
        if (c == '\n') first_line = false;
      }
      return Value::of(out);
    }
    if (name == "map") {
      const Value* attr = args.get(99, "attribute");
      ojson out = ojson::array();
      for (const auto& x : items_of(v, line)) {
        if (attr != nullptr) {
          out.push_back(data(attribute(Value::of(x), stringify(*attr)), line));
        } else {
          if (args.positional.empty()) fail(line, "map needs a filter or attribute=");
          Args rest;
          rest.positional.assign(args.positional.begin() + 1, args.positional.end());
          out.push_back(data(apply_filter(stringify(args.positional[0]), Value::of(x), rest, line), line));
        }
      }
      return Value::of(std::move(out));
    }
    if (name == "select" || name == "reject" || name == "selectattr" || name == "rejectattr") {
      const bool by_attr = name.ends_with("attr");
      const bool keep_true = name.starts_with("select");
      ojson out = ojson::array();
      for (const auto& x : items_of(v, line)) {
        std::size_t next = 0;
        Value subject = Value::of(x);
        if (by_attr) {
          if (args.positional.empty()) fail(line, name + " needs an attribute");
          subject = attribute(subject, stringify(args.positional[0]));
          next = 1;
        }
        bool r = false;
        if (next >= args.positional.size()) {
          r = truthy(subject);
        } else {
          Args rest;
          rest.positional.assign(args.positional.begin() + static_cast<std::ptrdiff_t>(next + 1), args.positional.end());
          r = test(stringify(args.positional[next]), subject, rest, line);
        }
        if (r == keep_true) out.push_back(x);
      }
      return Value::of(std::move(out));
    }
    if (name == "unique") {
      ojson out = ojson::array();
      for (const auto& x : items_of(v, line))
        if (std::none_of(out.begin(), out.end(), [&](const ojson& y) { return y == x; })) out.push_back(x);
      return Value::of(std::move(out));
    }
    if (name == "min" || name == "max" || name == "sum") {
      const auto all = items_of(v, line);
      if (name == "sum") {
        double total = 0;
        bool ints = true;
        for (const auto& x : all) { total += number(Value::of(x), line); ints = ints && x.is_number_integer(); }
        return ints ? Value::of(static_cast<std::int64_t>(total)) : Value::of(total);
      }
      if (all.empty()) return Value::undefined();
      ojson best = all.front();
      for (const auto& x : all)
        if (name == "min" ? x < best : best < x) best = x;
      return Value::of(best);
    }
    fail(line, "unsupported filter '" + name + "'");
  }

  Frame& root_;
};

// --- static reading of the program ---------------------------------------------

void walk_expr(const Expr& e, const std::function<void(const Expr&)>& visit) {
  visit(e);
  for (const auto& a : e.args)
    if (a) walk_expr(*a, visit);
  for (const auto& [k, v] : e.kwargs)
    if (v) walk_expr(*v, visit);
}

void walk(const Body& body, const std::function<void(const Node&)>& node,
          const std::function<void(const Expr&)>& expr) {
  for (const NodePtr& n : body) {
    node(*n);
    if (n->expr) walk_expr(*n->expr, expr);
    if (n->filter) walk_expr(*n->filter, expr);
    for (const auto& [cond, b] : n->branches) {
      if (cond) walk_expr(*cond, expr);
      walk(b, node, expr);
    }
    for (const auto& d : n->defaults)
      if (d) walk_expr(*d, expr);
    walk(n->body, node, expr);
    walk(n->else_body, node, expr);
  }
}

bool mentions(const Expr& e, const std::set<std::string>& names) {
  bool found = false;
  walk_expr(e, [&](const Expr& x) {
    if (x.op == Op::kName && names.contains(x.name)) found = true;
  });
  return found;
}

void string_literals(const Expr& e, std::vector<std::string>& out) {
  if (e.op == Op::kLiteral && e.literal.is_string()) {
    out.push_back(e.literal.get<std::string>());
  } else if (e.op == Op::kList || e.op == Op::kTuple) {
    for (const auto& a : e.args)
      if (a) string_literals(*a, out);
  }
}

}  // namespace
}  // namespace jinja

Result<ChatTemplate> ChatTemplate::parse(std::string_view source) {
  try {
    auto program = std::make_shared<jinja::Program>();
    jinja::Parser parser(jinja::lex(source));
    program->body = parser.parse();
    ChatTemplate t;
    t.program_ = std::move(program);
    return t;
  } catch (const std::exception& e) {
    return LSE_ERROR(kInvalidArgument, e.what());
  }
}

Result<std::string> ChatTemplate::render(const nlohmann::ordered_json& context) const {
  if (!program_) return LSE_ERROR(kInvalidArgument, "no chat template");
  try {
    jinja::Frame root;
    if (context.is_object())
      for (const auto& [k, v] : context.items()) root.vars[k] = jinja::Value::of(v);
    jinja::Renderer renderer(root);
    std::string out;
    (void)renderer.exec(program_->body, root, out);
    return out;
  } catch (const jinja::Raised& e) {
    return LSE_ERROR(kInvalidArgument, "the chat template raised: ", e.what());
  } catch (const std::exception& e) {
    return LSE_ERROR(kInvalidArgument, e.what());
  }
}

bool ChatTemplate::references(std::string_view name) const {
  if (!program_) return false;
  bool found = false;
  jinja::walk(program_->body, [](const jinja::Node&) {},
              [&](const jinja::Expr& e) {
                if (e.op == jinja::Op::kName && e.name == name) found = true;
              });
  return found;
}

std::vector<std::string> ChatTemplate::literals_compared_with(std::string_view name) const {
  std::vector<std::string> out;
  if (!program_) return out;
  std::set<std::string> tracked{std::string(name)};
  // Names assigned from an expression that reads a tracked name follow it,
  // e.g. `set effort = reasoning_effort|default('high')`.
  for (bool grew = true; grew;) {
    grew = false;
    jinja::walk(program_->body,
                [&](const jinja::Node& n) {
                  if (n.kind == jinja::NodeKind::kSet && n.attribute.empty() && n.expr &&
                      jinja::mentions(*n.expr, tracked))
                    for (const auto& t : n.targets) grew = tracked.insert(t).second || grew;
                },
                [](const jinja::Expr&) {});
  }
  jinja::walk(program_->body, [](const jinja::Node&) {}, [&](const jinja::Expr& e) {
    if (e.op == jinja::Op::kBinary &&
        (e.name == "==" || e.name == "!=" || e.name == "in" || e.name == "not in")) {
      if (jinja::mentions(*e.args[0], tracked)) jinja::string_literals(*e.args[1], out);
      else if (jinja::mentions(*e.args[1], tracked)) jinja::string_literals(*e.args[0], out);
    } else if (e.op == jinja::Op::kFilter && (e.name == "default" || e.name == "d") &&
               e.args.size() > 1 && jinja::mentions(*e.args[0], tracked)) {
      jinja::string_literals(*e.args[1], out);
    } else if (e.op == jinja::Op::kTest && (e.name == "eq" || e.name == "equalto" || e.name == "in") &&
               e.args.size() > 1 && jinja::mentions(*e.args[0], tracked)) {
      jinja::string_literals(*e.args[1], out);
    }
  });
  std::vector<std::string> unique;
  for (auto& s : out)
    if (std::find(unique.begin(), unique.end(), s) == unique.end()) unique.push_back(std::move(s));
  return unique;
}

namespace {
Result<std::string> read_text(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return LSE_ERROR(kIoError, "cannot read '", path.string(), "'");
  std::ostringstream text;
  text << in.rdbuf();
  if (in.bad()) return LSE_ERROR(kIoError, "cannot read '", path.string(), "'");
  return text.str();
}
}  // namespace

Result<ChatTemplateSource> find_chat_template(const std::string& model_dir) {
  namespace fs = std::filesystem;
  const fs::path dir(model_dir);
  std::error_code ec;
  if (const fs::path p = dir / "chat_template.jinja"; fs::exists(p, ec)) {
    LSE_ASSIGN_OR(std::string text, read_text(p));
    if (text.empty()) return LSE_ERROR(kInvalidArgument, "'", p.string(), "' is empty");
    return ChatTemplateSource{std::move(text), p.string()};
  }
  for (const char* file : {"tokenizer_config.json", "chat_template.json"}) {
    const fs::path p = dir / file;
    if (!fs::exists(p, ec)) continue;
    LSE_ASSIGN_OR(std::string text, read_text(p));
    nlohmann::json j;
    try {
      j = nlohmann::json::parse(text);
    } catch (const std::exception& e) {
      return LSE_ERROR(kInvalidArgument, "'", p.string(), "' is not valid JSON: ", e.what());
    }
    if (!j.is_object()) return LSE_ERROR(kInvalidArgument, "'", p.string(), "' must be a JSON object");
    const auto at = j.find("chat_template");
    if (at == j.end() || at->is_null()) continue;
    if (at->is_string()) {
      if (at->get<std::string>().empty()) return LSE_ERROR(kInvalidArgument, "'", p.string(), "' has an empty chat_template");
      return ChatTemplateSource{at->get<std::string>(), p.string()};
    }
    if (at->is_array()) {
      for (const auto& named : *at) {
        if (named.is_object() && named.value("name", "") == "default" && named.contains("template") &&
            named["template"].is_string())
          return ChatTemplateSource{named["template"].get<std::string>(), p.string() + "#default"};
      }
      return LSE_ERROR(kInvalidArgument, "'", p.string(), "' lists chat templates but none named \"default\"");
    }
    return LSE_ERROR(kInvalidArgument, "'", p.string(), "': chat_template must be a string or a list of named templates");
  }
  return ChatTemplateSource{};
}

}  // namespace lse::models
