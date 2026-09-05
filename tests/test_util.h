// Minimal assert helpers so the tests need no external framework.
#pragma once

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>

namespace test {

inline int& failure_count() {
  static int count = 0;
  return count;
}

inline void report(bool ok, const char* file, int line, const std::string& what) {
  if (ok) return;
  std::fprintf(stderr, "FAIL %s:%d: %s\n", file, line, what.c_str());
  ++failure_count();
}

#define CHECK(cond) \
  ::test::report((cond), __FILE__, __LINE__, "expected " #cond)

#define CHECK_EQ(a, b)                                                    \
  do {                                                                    \
    auto lhs_ = (a);                                                      \
    auto rhs_ = (b);                                                      \
    if (!(lhs_ == rhs_)) {                                                \
      std::ostringstream oss_;                                            \
      oss_ << #a " == " #b " (" << lhs_ << " vs " << rhs_ << ")";         \
      ::test::report(false, __FILE__, __LINE__, oss_.str());              \
    }                                                                     \
  } while (0)

#define CHECK_NEAR(a, b, tol)                                             \
  do {                                                                    \
    double lhs_ = static_cast<double>(a);                                 \
    double rhs_ = static_cast<double>(b);                                 \
    if (!(std::fabs(lhs_ - rhs_) <= (tol))) {                             \
      std::ostringstream oss_;                                            \
      oss_ << #a " near " #b " (" << lhs_ << " vs " << rhs_               \
           << ", tol " << (tol) << ")";                                   \
      ::test::report(false, __FILE__, __LINE__, oss_.str());              \
    }                                                                     \
  } while (0)

// Asserts that `body` throws, and that the message mentions `needle`.
inline void expect_throws(const char* file, int line, const char* label,
                          const std::string& needle,
                          const std::function<void()>& body) {
  try {
    body();
  } catch (const std::exception& e) {
    const std::string what = e.what();
    if (needle.empty() || what.find(needle) != std::string::npos) return;
    report(false, file, line,
           std::string(label) + " threw, but the message did not mention \"" +
               needle + "\": " + what);
    return;
  }
  report(false, file, line, std::string(label) + " did not throw");
}

#define CHECK_THROWS(needle, body) \
  ::test::expect_throws(__FILE__, __LINE__, #body, needle, [&]() body)

inline int finish(const char* suite) {
  if (failure_count() == 0) {
    std::printf("%s: all checks passed\n", suite);
    return 0;
  }
  std::fprintf(stderr, "%s: %d check(s) failed\n", suite, failure_count());
  return 1;
}

}  // namespace test
