// Reader for the safetensors checkpoint format.
//
// A safetensors file is a little-endian u64 header length, a JSON header of that
// many bytes describing every tensor, then one contiguous byte buffer holding
// the tensor data. Offsets in the header are relative to the start of that
// buffer. Files are memory-mapped, so opening a 2.5 GB checkpoint costs nothing
// until the pages are actually touched by the upload to the GPU.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "common.h"

namespace lcr {

// A borrowed view of one tensor inside a mapped file. The data pointer stays
// valid for as long as the owning SafetensorsArchive is alive.
struct TensorView {
  std::string name;
  DType dtype = DType::kF32;
  std::vector<int64_t> shape;
  const void* data = nullptr;
  size_t nbytes = 0;

  int64_t numel() const;
  int64_t dim(int index) const;
  int rank() const { return static_cast<int>(shape.size()); }
  std::string shape_string() const;

  // Throws unless the tensor has exactly this shape, naming the offender. Used
  // by the weight loader so a mismatched checkpoint fails with a useful message
  // instead of a silent out-of-bounds read on the GPU.
  void expect_shape(std::initializer_list<int64_t> expected) const;
};

// One memory-mapped safetensors file.
class MappedFile {
 public:
  explicit MappedFile(const std::string& path);
  ~MappedFile();

  MappedFile(const MappedFile&) = delete;
  MappedFile& operator=(const MappedFile&) = delete;

  const uint8_t* data() const { return base_; }
  size_t size() const { return size_; }
  const std::string& path() const { return path_; }

 private:
  std::string path_;
  int fd_ = -1;
  uint8_t* base_ = nullptr;
  size_t size_ = 0;
};

// A whole checkpoint: either a single model.safetensors or a sharded set
// described by model.safetensors.index.json.
class SafetensorsArchive {
 public:
  // `model_dir` is the directory holding the checkpoint files.
  explicit SafetensorsArchive(const std::string& model_dir);

  // Throws if the tensor is absent.
  const TensorView& get(const std::string& name) const;
  // Returns nullptr if the tensor is absent.
  const TensorView* find(const std::string& name) const;
  bool has(const std::string& name) const { return find(name) != nullptr; }

  std::vector<std::string> names() const;
  size_t tensor_count() const { return tensors_.size(); }
  // Sum of every tensor's byte size, which is what has to cross PCIe at load
  // and what has to be read from HBM per decoded token.
  size_t total_bytes() const;

 private:
  void open_file(const std::string& path);

  std::string dir_;
  std::vector<std::unique_ptr<MappedFile>> files_;
  std::unordered_map<std::string, TensorView> tensors_;
};

}  // namespace lcr
