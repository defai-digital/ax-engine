#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <system_error>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include "mlx/io.h"
#include "mlx/version.h"

namespace mx = mlx::core;

// An independent stream position lets descriptor users retain their own cursor.
class DescriptorReader final : public mx::io::Reader {
 public:
  explicit DescriptorReader(int source) : fd_(fcntl(source, F_DUPFD_CLOEXEC, 0)) {
    if (fd_ < 0) {
      throw std::system_error(errno, std::generic_category(), "duplicate tensor file");
    }
  }

  DescriptorReader(const DescriptorReader&) = delete;
  DescriptorReader& operator=(const DescriptorReader&) = delete;
  ~DescriptorReader() override { close(fd_); }

  bool is_open() const override { return fd_ >= 0; }
  bool good() const override { return is_open(); }
  size_t tell() override { return static_cast<size_t>(position_); }
  std::string label() const override { return "opened tensor file"; }

  void seek(int64_t offset, std::ios_base::seekdir direction) override {
    int64_t origin = 0;
    if (direction == std::ios_base::end) {
      struct stat status {};
      if (fstat(fd_, &status) != 0) {
        throw std::system_error(errno, std::generic_category(), "stat tensor file");
      }
      origin = status.st_size;
    } else if (direction == std::ios_base::cur) {
      origin = position_;
    }
    int64_t next = 0;
    if (__builtin_add_overflow(origin, offset, &next) || next < 0) {
      throw std::out_of_range("invalid tensor file seek");
    }
    position_ = next;
  }

  void read(char* destination, size_t count) override {
    read(destination, count, tell());
    position_ += static_cast<int64_t>(count);
  }

  void read(char* destination, size_t count, size_t offset) override {
    constexpr size_t max_offset = std::numeric_limits<off_t>::max();
    if (offset > max_offset || count > max_offset - offset) {
      throw std::out_of_range("invalid tensor file read");
    }
    while (count > 0) {
      const auto length = std::min(count, size_t{64} << 20);
      const auto bytes = pread(fd_, destination, length, static_cast<off_t>(offset));
      if (bytes < 0 && errno == EINTR) {
        continue;
      }
      if (bytes < 0) {
        throw std::system_error(errno, std::generic_category(), "read tensor file");
      }
      if (bytes == 0) {
        throw std::runtime_error("short tensor file read");
      }
      destination += bytes;
      count -= static_cast<size_t>(bytes);
      offset += static_cast<size_t>(bytes);
    }
  }

 private:
  const int fd_;
  int64_t position_ = 0;
};

static int open_source(const std::filesystem::path& path) {
  const int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) {
    throw std::system_error(errno, std::generic_category(), "open tensor file");
  }
  return fd;
}

#include <chrono>
#include <iomanip>
#include <sstream>
#include <vector>

#include <CommonCrypto/CommonDigest.h>
#include <mach/mach.h>

#include "mlx/transforms.h"

static uint64_t rss() {
  struct mach_task_basic_info stats {};
  mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
  const auto code = task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                              reinterpret_cast<task_info_t>(&stats), &count);
  if (code != KERN_SUCCESS) {
    throw std::runtime_error("cannot sample process RSS");
  }
  return stats.resident_size;
}

static void hash_text(CC_SHA256_CTX& hash, const std::string& text) {
  const uint64_t length = text.size();
  CC_SHA256_Update(&hash, &length, sizeof(length));
  CC_SHA256_Update(&hash, text.data(), static_cast<CC_LONG>(length));
}

static std::string layout(const mx::array& tensor) {
  std::ostringstream shape;
  shape << "Bfloat16:[";
  const char* delimiter = "";
  for (auto dimension : tensor.shape()) {
    shape << delimiter << dimension;
    delimiter = ", ";
  }
  shape << ']';
  return shape.str();
}

int main(int argc, char** argv) {
  if (argc != 3) {
    return 2;
  }
  const std::filesystem::path path(argv[1]);
  const std::string mode(argv[2]);
  if (mode != "path-lazy" && mode != "descriptor-lazy") {
    return 2;
  }
  if (std::string(mx::version()) != "0.32.4") {
    throw std::runtime_error("unexpected MLX runtime");
  }
  const auto cpu = mx::Device(mx::Device::cpu);
  const auto before = rss();
  const auto start = std::chrono::steady_clock::now();
  auto tensors = [&]() {
    if (mode == "path-lazy") {
      return mx::load_safetensors(path.string(), cpu).first;
    }
    const int fd = open_source(path);
    auto reader = std::make_shared<DescriptorReader>(fd);
    close(fd);
    return mx::load_safetensors(std::move(reader), cpu).first;
  }();

  if (tensors.size() != 31) {
    throw std::runtime_error("unexpected sidecar tensor count");
  }
  std::vector<mx::array> arrays;
  for (const auto& [name, tensor] : tensors) {
    if (name.rfind("mtp.", 0) != 0 || tensor.dtype() != mx::bfloat16) {
      throw std::runtime_error("unexpected sidecar tensor layout");
    }
    arrays.push_back(tensor);
  }
  mx::eval(std::move(arrays));
  const std::chrono::duration<double> duration = std::chrono::steady_clock::now() - start;
  const auto after = rss();

  std::vector<std::string> names;
  for (const auto& entry : tensors) {
    names.push_back(entry.first);
  }
  std::sort(names.begin(), names.end());
  CC_SHA256_CTX context;
  CC_SHA256_Init(&context);
  uint64_t bytes = 0;
  for (const auto& name : names) {
    const auto& tensor = tensors.at(name);
    hash_text(context, name);
    hash_text(context, layout(tensor));
    const uint64_t length = tensor.nbytes();
    CC_SHA256_Update(&context, &length, sizeof(length));
    uint64_t offset = 0;
    while (offset < length) {
      const auto count = std::min(length - offset, uint64_t{64} << 20);
      CC_SHA256_Update(&context, tensor.data<uint8_t>() + offset, static_cast<CC_LONG>(count));
      offset += count;
    }
    bytes += length;
  }
  unsigned char digest[CC_SHA256_DIGEST_LENGTH];
  CC_SHA256_Final(digest, &context);
  std::ostringstream hex;
  hex << std::hex << std::setfill('0');
  for (auto value : digest) {
    hex << std::setw(2) << static_cast<unsigned int>(value);
  }
  std::cout << std::setprecision(17) << "{\"files\":[{\"tensors\":" << names.size()
            << ",\"selected_bytes\":" << bytes << ",\"sha256\":\"" << hex.str()
            << "\",\"load_eval_seconds\":" << duration.count()
            << ",\"rss_before_bytes\":" << before
            << ",\"rss_after_eval_bytes\":" << after << "}]}\n";
}
