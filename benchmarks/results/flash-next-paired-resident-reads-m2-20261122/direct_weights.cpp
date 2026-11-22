#include <CommonCrypto/CommonDigest.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <mlx/allocator.h>
#include <mlx/mlx.h>

namespace mx = mlx::core;

namespace {

struct Source {
    explicit Source(const char* path) : fd(open(path, O_RDONLY | O_CLOEXEC)) {
        if (fd < 0) {
            throw std::runtime_error("Could not open tensor shard");
        }
    }
    ~Source() { close(fd); }
    Source(const Source&) = delete;
    Source& operator=(const Source&) = delete;
    const int fd;
};

struct Mapping {
    Mapping(int fd, size_t size) : len(size), memory(mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0)) {
        if (memory == MAP_FAILED) {
            throw std::runtime_error("Could not map tensor shard");
        }
    }
    ~Mapping() { munmap(memory, len); }
    Mapping(const Mapping&) = delete;
    Mapping& operator=(const Mapping&) = delete;
    const size_t len;
    void* const memory;
};

struct Tensor {
    size_t offset;
    size_t len;
    mx::Dtype dtype;
    mx::Shape shape;
};

mx::Dtype type(const std::string& code) {
    if (code == "U8") return mx::uint8;
    if (code == "U32") return mx::uint32;
    if (code == "BF16") return mx::bfloat16;
    if (code == "F32") return mx::float32;
    throw std::runtime_error("Unsupported diagnostic dtype");
}

std::vector<Tensor> geometry(const char* path, size_t file_size) {
    std::ifstream plan(path);
    if (!plan) {
        throw std::runtime_error("Could not open validated tensor geometry");
    }
    std::vector<Tensor> tensors;
    size_t offset;
    while (plan >> offset) {
        size_t len;
        size_t dims;
        std::string code;
        if (!(plan >> len >> code >> dims) || dims > 8 || len == 0 ||
            offset > file_size || len > file_size - offset) {
            throw std::runtime_error("Tensor range exceeds validated file bounds");
        }
        const auto dtype = type(code);
        const size_t itemsize = mx::size_of(dtype);
        size_t expected = itemsize;
        mx::Shape shape;
        for (size_t dim = 0; dim < dims; ++dim) {
            int extent;
            if (!(plan >> extent) || extent <= 0 || expected > len / static_cast<size_t>(extent)) {
                throw std::runtime_error("Tensor shape overflow or mismatch");
            }
            shape.push_back(extent);
            expected *= static_cast<size_t>(extent);
        }
        if (expected != len) {
            throw std::runtime_error("Tensor length differs from its shape");
        }
        tensors.push_back({offset, len, dtype, std::move(shape)});
    }
    if (!plan.eof() || tensors.empty()) {
        throw std::runtime_error("Invalid or empty tensor geometry");
    }
    return tensors;
}

void read_exact(int fd, unsigned char* out, size_t len, size_t offset) {
    const size_t max_chunk = 4 * 1024 * 1024;
    size_t finished = 0;
    while (finished < len) {
        const auto count = pread(fd, out + finished, std::min(len - finished, max_chunk),
                                 static_cast<off_t>(offset + finished));
        if (count == 0) {
            throw std::runtime_error("Tensor read ended early");
        }
        if (count < 0) {
            if (errno == EINTR) {
                continue;
            }
            throw std::runtime_error("Positional tensor read failed");
        }
        finished += static_cast<size_t>(count);
    }
}

void read_pair(int fd, unsigned char* out, size_t len, size_t offset, size_t first_size) {
    auto first = std::async(std::launch::async, read_exact, fd, out, first_size, offset);
    read_exact(fd, out + first_size, len - first_size, offset + first_size);
    first.get();
}

constexpr size_t parallel_threshold = 64 * 1024 * 1024;
constexpr size_t read_alignment = 4 * 1024 * 1024;

mx::array materialized(int fd, const void* base, const Tensor& tensor, bool paired) {
    auto storage = mx::allocator::malloc(tensor.len);
    auto* destination = static_cast<unsigned char*>(storage.buffer().raw_ptr());
    if (!storage.is_owned() || destination == nullptr) {
        throw std::runtime_error("Allocator did not provide owned CPU-accessible storage");
    }
    if (base != nullptr) {
        std::memcpy(destination, static_cast<const unsigned char*>(base) + tensor.offset, tensor.len);
    } else if (paired && tensor.len >= parallel_threshold) {
        const size_t first_size = (tensor.len / 2 / read_alignment) * read_alignment;
        read_pair(fd, destination, tensor.len, tensor.offset, first_size);
    } else {
        read_exact(fd, destination, tensor.len, tensor.offset);
    }
    mx::array result(std::move(storage), tensor.shape, tensor.dtype);
    if (result.status() != mx::array::Status::available || !result.flags().row_contiguous ||
        result.buffer_size() < tensor.len || result.data<unsigned char>() != destination) {
        throw std::runtime_error("Array did not retain available owned storage");
    }
    return result;
}

class Digest {
  public:
    Digest() { CC_SHA256_Init(&ctx_); }
    void add(const void* bytes, size_t len) {
        auto* data = static_cast<const unsigned char*>(bytes);
        constexpr size_t limit = 64 * 1024 * 1024;
        while (len != 0) {
            const size_t chunk = std::min(len, limit);
            CC_SHA256_Update(&ctx_, data, static_cast<CC_LONG>(chunk));
            data += chunk;
            len -= chunk;
        }
    }
    std::string finish() {
        unsigned char raw[CC_SHA256_DIGEST_LENGTH];
        CC_SHA256_Final(raw, &ctx_);
        std::ostringstream text;
        for (unsigned char value : raw) {
            text << std::hex << std::setfill('0') << std::setw(2) << static_cast<int>(value);
        }
        return text.str();
    }
  private:
    CC_SHA256_CTX ctx_{};
};

void metal_probe(const mx::array& values, Digest& checksum) {
    auto flat = mx::reshape(values, {-1}, mx::Device::gpu);
    const int count = static_cast<int>(std::min<size_t>(values.size(), 16));
    auto portion = mx::slice(flat, {0}, {count}, mx::Device::gpu);
    auto widened = mx::astype(portion, mx::float32, mx::Device::gpu);
    auto transformed = mx::add(widened, mx::array(1.25f), mx::Device::gpu);
    mx::eval(transformed);
    checksum.add(transformed.data<float>(), transformed.nbytes());
}

void run(const char* file, const char* plan, const std::string& mode) {
    const bool mapped = mode == "mmap";
    const bool paired = mode == "pread-pair";
    if (!mapped && !paired && mode != "pread") {
        throw std::runtime_error("Unknown diagnostic mode");
    }
    Source input(file);
    struct stat header {};
    if (fstat(input.fd, &header) != 0 || header.st_size <= 0 || !S_ISREG(header.st_mode)) {
        throw std::runtime_error("Unexpected tensor source");
    }
    const size_t file_size = static_cast<size_t>(header.st_size);
    const auto tensors = geometry(plan, file_size);
    std::vector<mx::array> arrays;
    arrays.reserve(tensors.size());
    const auto started = std::chrono::steady_clock::now();
    if (mapped) {
        const Mapping map(input.fd, file_size);
        for (const auto& tensor : tensors) {
            arrays.push_back(materialized(input.fd, map.memory, tensor, false));
        }
    } else {
        for (const auto& tensor : tensors) {
            arrays.push_back(materialized(input.fd, nullptr, tensor, paired));
        }
    }
    const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    struct stat after {};
    if (fstat(input.fd, &after) != 0 || after.st_size != header.st_size ||
        after.st_mtimespec.tv_sec != header.st_mtimespec.tv_sec ||
        after.st_mtimespec.tv_nsec != header.st_mtimespec.tv_nsec ||
        after.st_ctimespec.tv_sec != header.st_ctimespec.tv_sec ||
        after.st_ctimespec.tv_nsec != header.st_ctimespec.tv_nsec) {
        throw std::runtime_error("Tensor source changed while reading");
    }
    Digest data;
    Digest metal;
    size_t selected = 0;
    size_t allocated = 0;
    for (const auto& array : arrays) {
        data.add(array.data<unsigned char>(), array.nbytes());
        metal_probe(array, metal);
        selected += array.nbytes();
        allocated += array.buffer_size();
    }
    const auto large_tensors = std::count_if(tensors.begin(), tensors.end(), [](const Tensor& tensor) {
        return tensor.len >= parallel_threshold;
    });
    std::cout << std::setprecision(9) << "{\"mode\":\"" << mode
              << "\",\"elapsed_seconds\":" << elapsed
              << ",\"tensors\":" << tensors.size()
              << ",\"paired_tensors\":" << (paired ? large_tensors : 0)
              << ",\"selected_bytes\":" << selected
              << ",\"allocated_bytes\":" << allocated
              << ",\"weights_sha256\":\"" << data.finish()
              << "\",\"metal_sha256\":\"" << metal.finish() << "\"}\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--eof-self-test") {
            std::unique_ptr<FILE, decltype(&fclose)> file(tmpfile(), fclose);
            if (!file || fwrite("abcd", 1, 4, file.get()) != 4 || fflush(file.get()) != 0) {
                throw std::runtime_error("Could not prepare owned EOF fixture");
            }
            auto rejects_partial = [&](bool paired) {
                unsigned char buffer[8]{};
                try {
                    if (paired) {
                        read_pair(fileno(file.get()), buffer, sizeof(buffer), 0, 4);
                    } else {
                        (void)materialized(fileno(file.get()), nullptr, Tensor{0, 8, mx::uint8, {8}}, false);
                    }
                } catch (const std::runtime_error& error) {
                    if (std::string(error.what()) != "Tensor read ended early") throw;
                    return;
                }
                throw std::runtime_error("Unexpected partial array result");
            };
            rejects_partial(false);
            rejects_partial(true);
            std::cout << "Serial and paired reads reject partial payloads.\n";
            return 0;
        }
        if (argc != 4) return 2;
        run(argv[1], argv[2], argv[3]);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
