#include "safetensors.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <fstream>
#include <set>

#include <nlohmann/json.hpp>

namespace lcr {
namespace {

using json = nlohmann::json;

// The header length prefix is a fixed 8-byte little-endian u64 regardless of
// host endianness, so read it byte by byte instead of memcpy-ing a u64.
uint64_t read_u64_le(const uint8_t* p) {
  uint64_t v = 0;
  for (int i = 0; i < 8; ++i) v |= static_cast<uint64_t>(p[i]) << (8 * i);
  return v;
}

bool file_exists(const std::string& path) {
  struct stat st;
  return ::stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

std::string join_path(const std::string& dir, const std::string& name) {
  if (dir.empty()) return name;
  if (dir.back() == '/') return dir + name;
  return dir + "/" + name;
}

// A 32 MiB header would already be pathological for a 1B model; anything larger
// is a corrupt or hostile file and is rejected before the JSON parser sees it.
constexpr uint64_t kMaxHeaderBytes = 32ull << 20;

}  // namespace

int64_t TensorView::numel() const {
  int64_t n = 1;
  for (int64_t d : shape) n *= d;
  return n;
}

int64_t TensorView::dim(int index) const {
  LCR_CHECK(index >= 0 && index < rank(),
            "dim " << index << " out of range for tensor \"" << name
                   << "\" of rank " << rank());
  return shape[static_cast<size_t>(index)];
}

std::string TensorView::shape_string() const {
  std::ostringstream oss;
  oss << "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i) oss << ", ";
    oss << shape[i];
  }
  oss << "]";
  return oss.str();
}

void TensorView::expect_shape(std::initializer_list<int64_t> expected) const {
  bool ok = expected.size() == shape.size();
  if (ok) {
    size_t i = 0;
    for (int64_t e : expected) {
      if (shape[i++] != e) {
        ok = false;
        break;
      }
    }
  }
  if (!ok) {
    std::ostringstream want;
    want << "[";
    size_t i = 0;
    for (int64_t e : expected) {
      if (i++) want << ", ";
      want << e;
    }
    want << "]";
    LCR_FAIL("tensor \"" << name << "\" has shape " << shape_string()
                         << ", expected " << want.str());
  }
}

MappedFile::MappedFile(const std::string& path) : path_(path) {
  fd_ = ::open(path.c_str(), O_RDONLY);
  LCR_CHECK(fd_ >= 0, "cannot open " << path << ": " << std::strerror(errno));

  struct stat st;
  if (::fstat(fd_, &st) != 0) {
    int err = errno;
    ::close(fd_);
    fd_ = -1;
    LCR_FAIL("cannot stat " << path << ": " << std::strerror(err));
  }
  size_ = static_cast<size_t>(st.st_size);
  LCR_CHECK(size_ > 8, path << " is too small to be a safetensors file ("
                            << size_ << " bytes)");

  void* map = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
  if (map == MAP_FAILED) {
    int err = errno;
    ::close(fd_);
    fd_ = -1;
    LCR_FAIL("cannot mmap " << path << ": " << std::strerror(err));
  }
  base_ = static_cast<uint8_t*>(map);
}

MappedFile::~MappedFile() {
  if (base_ != nullptr) ::munmap(base_, size_);
  if (fd_ >= 0) ::close(fd_);
}

SafetensorsArchive::SafetensorsArchive(const std::string& model_dir)
    : dir_(model_dir) {
  const std::string single = join_path(dir_, "model.safetensors");
  const std::string index = join_path(dir_, "model.safetensors.index.json");

  if (file_exists(single)) {
    open_file(single);
  } else if (file_exists(index)) {
    // Sharded checkpoint. The index maps every tensor name to the shard that
    // holds it; open each distinct shard once.
    std::ifstream in(index);
    LCR_CHECK(in.good(), "cannot read " << index);
    json doc = json::parse(in, nullptr, false);
    LCR_CHECK(!doc.is_discarded(), index << " is not valid JSON");
    LCR_CHECK(doc.contains("weight_map") && doc["weight_map"].is_object(),
              index << " has no weight_map object");

    std::set<std::string> shards;
    for (const auto& entry : doc["weight_map"].items()) {
      LCR_CHECK(entry.value().is_string(),
                "weight_map entry for \"" << entry.key() << "\" is not a "
                                          << "filename");
      shards.insert(entry.value().get<std::string>());
    }
    for (const std::string& shard : shards) {
      open_file(join_path(dir_, shard));
    }
  } else {
    LCR_FAIL("no model.safetensors or model.safetensors.index.json in "
             << dir_);
  }

  LCR_CHECK(!tensors_.empty(), "checkpoint in " << dir_ << " has no tensors");
}

void SafetensorsArchive::open_file(const std::string& path) {
  files_.push_back(std::make_unique<MappedFile>(path));
  const MappedFile& file = *files_.back();
  const uint8_t* base = file.data();
  const size_t size = file.size();

  const uint64_t header_len = read_u64_le(base);
  LCR_CHECK(header_len > 0 && header_len <= kMaxHeaderBytes,
            path << " declares an implausible header length of " << header_len
                 << " bytes");
  LCR_CHECK(8 + header_len <= size,
            path << " header length " << header_len << " exceeds the file size "
                 << size);

  json header = json::parse(base + 8, base + 8 + header_len, nullptr, false);
  LCR_CHECK(!header.is_discarded(), path << " has a malformed JSON header");
  LCR_CHECK(header.is_object(), path << " header is not a JSON object");

  const uint8_t* payload = base + 8 + header_len;
  const size_t payload_size = size - 8 - header_len;

  for (const auto& entry : header.items()) {
    const std::string& name = entry.key();
    if (name == "__metadata__") continue;  // free-form string map, not a tensor

    const json& spec = entry.value();
    LCR_CHECK(spec.is_object(), "tensor \"" << name << "\" in " << path
                                            << " is not described by an object");
    LCR_CHECK(spec.contains("dtype") && spec.contains("shape") &&
                  spec.contains("data_offsets"),
              "tensor \"" << name << "\" in " << path
                          << " is missing dtype, shape, or data_offsets");

    TensorView view;
    view.name = name;
    view.dtype = dtype_from_string(spec["dtype"].get<std::string>());
    view.shape = spec["shape"].get<std::vector<int64_t>>();
    for (int64_t d : view.shape) {
      LCR_CHECK(d >= 0, "tensor \"" << name << "\" has negative extent " << d);
    }

    const auto offsets = spec["data_offsets"].get<std::vector<int64_t>>();
    LCR_CHECK(offsets.size() == 2, "tensor \"" << name
                                               << "\" has malformed "
                                                  "data_offsets");
    const int64_t begin = offsets[0];
    const int64_t end = offsets[1];
    LCR_CHECK(begin >= 0 && end >= begin,
              "tensor \"" << name << "\" has data_offsets [" << begin << ", "
                          << end << "]");
    LCR_CHECK(static_cast<size_t>(end) <= payload_size,
              "tensor \"" << name << "\" ends at " << end
                          << " but the data buffer of " << path << " is only "
                          << payload_size << " bytes");

    view.nbytes = static_cast<size_t>(end - begin);
    const size_t expected =
        static_cast<size_t>(view.numel()) * dtype_size(view.dtype);
    LCR_CHECK(view.nbytes == expected,
              "tensor \"" << name << "\" spans " << view.nbytes
                          << " bytes but its shape " << view.shape_string()
                          << " of " << dtype_name(view.dtype) << " needs "
                          << expected);
    view.data = payload + begin;

    auto [it, inserted] = tensors_.emplace(name, std::move(view));
    LCR_CHECK(inserted, "tensor \"" << name
                                    << "\" appears in more than one shard");
    (void)it;
  }
}

const TensorView* SafetensorsArchive::find(const std::string& name) const {
  auto it = tensors_.find(name);
  return it == tensors_.end() ? nullptr : &it->second;
}

const TensorView& SafetensorsArchive::get(const std::string& name) const {
  const TensorView* view = find(name);
  LCR_CHECK(view != nullptr,
            "checkpoint has no tensor named \"" << name << "\"");
  return *view;
}

std::vector<std::string> SafetensorsArchive::names() const {
  std::vector<std::string> out;
  out.reserve(tensors_.size());
  for (const auto& [name, unused] : tensors_) out.push_back(name);
  std::sort(out.begin(), out.end());
  return out;
}

size_t SafetensorsArchive::total_bytes() const {
  size_t total = 0;
  for (const auto& [unused, view] : tensors_) total += view.nbytes;
  return total;
}

}  // namespace lcr
