// Builds synthetic safetensors files on disk and checks that the loader reads
// the good ones and rejects the malformed ones.
#include <stdlib.h>
#include <unistd.h>

#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "safetensors.h"
#include "test_util.h"

namespace {

using lcr::DType;
using lcr::SafetensorsArchive;

std::string temp_dir() {
  char pattern[] = "/tmp/lcr_safetensors_XXXXXX";
  const char* dir = ::mkdtemp(pattern);
  CHECK(dir != nullptr);
  return std::string(dir);
}

// Writes an 8-byte little-endian header length, the header, then the payload.
void write_safetensors(const std::string& path, const std::string& header,
                       const std::vector<uint8_t>& payload,
                       uint64_t declared_header_len_override = 0) {
  std::ofstream out(path, std::ios::binary);
  CHECK(out.good());
  const uint64_t len = declared_header_len_override != 0
                           ? declared_header_len_override
                           : header.size();
  uint8_t prefix[8];
  for (int i = 0; i < 8; ++i) prefix[i] = static_cast<uint8_t>((len >> (8 * i)) & 0xff);
  out.write(reinterpret_cast<const char*>(prefix), 8);
  out.write(header.data(), static_cast<std::streamsize>(header.size()));
  if (!payload.empty()) {
    out.write(reinterpret_cast<const char*>(payload.data()),
              static_cast<std::streamsize>(payload.size()));
  }
}

std::vector<uint8_t> bytes_of(const std::vector<float>& values) {
  std::vector<uint8_t> out(values.size() * sizeof(float));
  std::memcpy(out.data(), values.data(), out.size());
  return out;
}

void test_reads_a_well_formed_file() {
  const std::string dir = temp_dir();
  // Two tensors: six floats, then four bf16 values that are only ever
  // inspected as raw bytes.
  const std::vector<float> a{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  std::vector<uint8_t> payload = bytes_of(a);
  for (uint8_t b : {0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08}) {
    payload.push_back(b);
  }
  const std::string header =
      R"({"__metadata__":{"format":"pt"},)"
      R"("model.embed_tokens.weight":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]},)"
      R"("model.norm.weight":{"dtype":"BF16","shape":[4],"data_offsets":[24,32]}})";
  write_safetensors(dir + "/model.safetensors", header, payload);

  SafetensorsArchive archive(dir);
  CHECK_EQ(archive.tensor_count(), static_cast<size_t>(2));
  CHECK_EQ(archive.total_bytes(), static_cast<size_t>(32));
  CHECK(archive.has("model.embed_tokens.weight"));
  CHECK(!archive.has("model.layers.0.mlp.gate_proj.weight"));

  const lcr::TensorView& embed = archive.get("model.embed_tokens.weight");
  CHECK(embed.dtype == DType::kF32);
  CHECK_EQ(embed.rank(), 2);
  CHECK_EQ(embed.dim(0), static_cast<int64_t>(2));
  CHECK_EQ(embed.dim(1), static_cast<int64_t>(3));
  CHECK_EQ(embed.numel(), static_cast<int64_t>(6));
  CHECK_EQ(embed.nbytes, static_cast<size_t>(24));
  CHECK_EQ(embed.shape_string(), std::string("[2, 3]"));

  // The mapped bytes must be the values that were written, in order.
  const float* values = static_cast<const float*>(embed.data);
  for (size_t i = 0; i < a.size(); ++i) CHECK_EQ(values[i], a[i]);

  // The second tensor starts exactly where the first one ended.
  const lcr::TensorView& norm = archive.get("model.norm.weight");
  CHECK(norm.dtype == DType::kBF16);
  CHECK_EQ(norm.nbytes, static_cast<size_t>(8));
  CHECK_EQ(static_cast<const uint8_t*>(norm.data)[0], static_cast<uint8_t>(0x01));

  // __metadata__ describes the file, it is not a tensor.
  CHECK(!archive.has("__metadata__"));

  // names() is sorted, which the weight-dump path relies on.
  const std::vector<std::string> names = archive.names();
  CHECK_EQ(names.size(), static_cast<size_t>(2));
  CHECK_EQ(names[0], std::string("model.embed_tokens.weight"));
  CHECK_EQ(names[1], std::string("model.norm.weight"));

  // expect_shape is the guard the weight loader uses.
  embed.expect_shape({2, 3});
  CHECK_THROWS("expected [3, 2]", { embed.expect_shape({3, 2}); });
  CHECK_THROWS("no tensor named", { archive.get("missing.weight"); });
}

void test_reads_a_sharded_checkpoint() {
  const std::string dir = temp_dir();
  const std::vector<float> first{1.0f, 2.0f};
  const std::vector<float> second{3.0f};
  write_safetensors(
      dir + "/model-00001-of-00002.safetensors",
      R"({"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}})",
      bytes_of(first));
  write_safetensors(
      dir + "/model-00002-of-00002.safetensors",
      R"({"b":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}})",
      bytes_of(second));
  std::ofstream index(dir + "/model.safetensors.index.json");
  index << R"({"metadata":{"total_size":12},"weight_map":{)"
        << R"("a":"model-00001-of-00002.safetensors",)"
        << R"("b":"model-00002-of-00002.safetensors"}})";
  index.close();

  SafetensorsArchive archive(dir);
  CHECK_EQ(archive.tensor_count(), static_cast<size_t>(2));
  CHECK_EQ(archive.total_bytes(), static_cast<size_t>(12));
  CHECK_EQ(static_cast<const float*>(archive.get("a").data)[1], 2.0f);
  CHECK_EQ(static_cast<const float*>(archive.get("b").data)[0], 3.0f);
}

void test_rejects_malformed_files() {
  // A tensor whose byte span does not match its shape and dtype. Letting this
  // through would read past the end of the mapping on the GPU upload.
  {
    const std::string dir = temp_dir();
    write_safetensors(
        dir + "/model.safetensors",
        R"({"a":{"dtype":"F32","shape":[4],"data_offsets":[0,8]}})",
        std::vector<uint8_t>(8, 0));
    CHECK_THROWS("needs 16", { SafetensorsArchive archive(dir); });
  }
  // A tensor that runs past the end of the data buffer.
  {
    const std::string dir = temp_dir();
    write_safetensors(
        dir + "/model.safetensors",
        R"({"a":{"dtype":"F32","shape":[8],"data_offsets":[0,32]}})",
        std::vector<uint8_t>(8, 0));
    CHECK_THROWS("data buffer", { SafetensorsArchive archive(dir); });
  }
  // A declared header length larger than the file.
  {
    const std::string dir = temp_dir();
    write_safetensors(dir + "/model.safetensors",
                      R"({"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}})",
                      std::vector<uint8_t>(4, 0), /*header_len_override=*/4096);
    CHECK_THROWS("exceeds the file size", { SafetensorsArchive archive(dir); });
  }
  // A header that is not JSON at all.
  {
    const std::string dir = temp_dir();
    write_safetensors(dir + "/model.safetensors", "not json at all{{{",
                      std::vector<uint8_t>(4, 0));
    CHECK_THROWS("malformed JSON header", { SafetensorsArchive archive(dir); });
  }
  // A dtype the runtime has never heard of.
  {
    const std::string dir = temp_dir();
    write_safetensors(
        dir + "/model.safetensors",
        R"({"a":{"dtype":"FP8_E4M3","shape":[4],"data_offsets":[0,4]}})",
        std::vector<uint8_t>(4, 0));
    CHECK_THROWS("unknown safetensors dtype", { SafetensorsArchive archive(dir); });
  }
  // A directory with no checkpoint in it.
  {
    const std::string dir = temp_dir();
    CHECK_THROWS("no model.safetensors", { SafetensorsArchive archive(dir); });
  }
}

}  // namespace

int main() {
  test_reads_a_well_formed_file();
  test_reads_a_sharded_checkpoint();
  test_rejects_malformed_files();
  return test::finish("test_safetensors");
}
