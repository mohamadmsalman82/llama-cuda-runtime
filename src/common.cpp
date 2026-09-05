#include "common.h"

#include <array>
#include <cmath>
#include <cstdio>

namespace lcr {
namespace detail {

void throw_failure(const char* file, int line, const char* cond,
                   const std::string& msg) {
  std::ostringstream oss;
  oss << file << ":" << line << ": check failed: " << cond;
  if (!msg.empty()) oss << "\n  " << msg;
  throw Error(oss.str());
}

}  // namespace detail

size_t dtype_size(DType dt) {
  switch (dt) {
    case DType::kF32:
    case DType::kI32:
      return 4;
    case DType::kF16:
    case DType::kBF16:
      return 2;
    case DType::kI64:
      return 8;
    case DType::kI8:
    case DType::kU8:
    case DType::kBool:
      return 1;
  }
  LCR_FAIL("unhandled dtype in dtype_size");
}

const char* dtype_name(DType dt) {
  switch (dt) {
    case DType::kF32: return "F32";
    case DType::kF16: return "F16";
    case DType::kBF16: return "BF16";
    case DType::kI64: return "I64";
    case DType::kI32: return "I32";
    case DType::kI8: return "I8";
    case DType::kU8: return "U8";
    case DType::kBool: return "BOOL";
  }
  LCR_FAIL("unhandled dtype in dtype_name");
}

DType dtype_from_string(const std::string& s) {
  if (s == "F32" || s == "F64") {
    LCR_CHECK(s == "F32", "F64 tensors are not supported");
    return DType::kF32;
  }
  if (s == "F16") return DType::kF16;
  if (s == "BF16") return DType::kBF16;
  if (s == "I64") return DType::kI64;
  if (s == "I32") return DType::kI32;
  if (s == "I8") return DType::kI8;
  if (s == "U8") return DType::kU8;
  if (s == "BOOL") return DType::kBool;
  LCR_FAIL("unknown safetensors dtype \"" << s << "\"");
}

std::string format_bytes(size_t bytes) {
  static const std::array<const char*, 5> kUnits{"B", "KiB", "MiB", "GiB",
                                                 "TiB"};
  double value = static_cast<double>(bytes);
  size_t unit = 0;
  while (value >= 1024.0 && unit + 1 < kUnits.size()) {
    value /= 1024.0;
    ++unit;
  }
  char buf[64];
  std::snprintf(buf, sizeof(buf), unit == 0 ? "%.0f %s" : "%.2f %s", value,
                kUnits[unit]);
  return std::string(buf);
}

}  // namespace lcr
