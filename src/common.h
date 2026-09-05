// Shared host-side utilities: error reporting, element types, small helpers.
#pragma once

#include <cstddef>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>

namespace lcr {

class Error : public std::runtime_error {
 public:
  explicit Error(const std::string& what) : std::runtime_error(what) {}
};

namespace detail {
[[noreturn]] void throw_failure(const char* file, int line, const char* cond,
                                const std::string& msg);
}  // namespace detail

// Checks that stay in release builds. Failures throw lcr::Error carrying the
// file, line, and the source text of the condition.
//   LCR_CHECK(n > 0, "expected a positive length, got " << n);
#define LCR_CHECK(cond, msg)                                              \
  do {                                                                    \
    if (!(cond)) {                                                        \
      std::ostringstream lcr_oss_;                                        \
      lcr_oss_ << msg;                                                    \
      ::lcr::detail::throw_failure(__FILE__, __LINE__, #cond,             \
                                   lcr_oss_.str());                       \
    }                                                                     \
  } while (0)

#define LCR_FAIL(msg) LCR_CHECK(false, msg)

// Element types that appear in Llama safetensors checkpoints. The runtime only
// computes in BF16, F16, and F32; the integer types show up in auxiliary
// tensors and are here so the loader can describe them without failing.
enum class DType { kF32, kF16, kBF16, kI64, kI32, kI8, kU8, kBool };

size_t dtype_size(DType dt);
const char* dtype_name(DType dt);
DType dtype_from_string(const std::string& s);

// Rounds up to the next multiple of `alignment`, which must be a power of two.
inline size_t align_up(size_t value, size_t alignment) {
  return (value + alignment - 1) & ~(alignment - 1);
}

// Human-readable byte counts for logs and the benchmark table.
std::string format_bytes(size_t bytes);

}  // namespace lcr
