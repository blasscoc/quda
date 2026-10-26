#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <openssl/evp.h>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

static void fadvise_dontneed_fd(int fd, off_t len = 0) {
#if defined(POSIX_FADV_DONTNEED)
  if (fd >= 0) {
    (void)posix_fadvise(fd, 0, len, POSIX_FADV_DONTNEED);
  }
#else
  (void)fd;
  (void)len;
#endif
}

static std::string to_hex(const unsigned char *digest, unsigned int len) {
  static const char *hex = "0123456789abcdef";
  std::string out;
  out.resize(len * 2);
  for (unsigned int i = 0; i < len; ++i) {
    out[2 * i] = hex[(digest[i] >> 4) & 0xF];
    out[2 * i + 1] = hex[digest[i] & 0xF];
  }
  return out;
}

int main(int argc, char **argv) {
  std::string src_path, dst_path;
  int src_dim[4] = {0, 0, 0, 0};
  int dst_dim[4] = {0, 0, 0, 0};

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--src" && i + 1 < argc) {
      src_path = argv[++i];
    } else if (arg == "--dst" && i + 1 < argc) {
      dst_path = argv[++i];
    } else if (arg == "--src-dim" && i + 4 < argc) {
      for (int d = 0; d < 4; ++d) src_dim[d] = std::atoi(argv[++i]);
    } else if (arg == "--dst-dim" && i + 4 < argc) {
      for (int d = 0; d < 4; ++d) dst_dim[d] = std::atoi(argv[++i]);
    } else if (arg == "--help" || arg == "-h") {
      std::printf("Usage: tile_gauge --src <in.bin> --src-dim Lx0 Ly0 Lz0 Lt0 --dst <out.bin> --dst-dim Lx1 Ly1 Lz1 Lt1\n");
      return 0;
    } else {
      std::fprintf(stderr, "Unknown or incomplete argument: %s\n", arg.c_str());
      return 2;
    }
  }

  if (src_path.empty() || dst_path.empty()) {
    std::fprintf(stderr, "Missing --src or --dst\n");
    return 2;
  }
  for (int d = 0; d < 4; ++d) {
    if (src_dim[d] <= 0 || dst_dim[d] <= 0 || (src_dim[d] % 2 != 0) || (dst_dim[d] % 2 != 0)) {
      std::fprintf(stderr, "Dimensions must be positive even integers\n");
      return 2;
    }
  }

  constexpr size_t LINK_BYTES = 144;
  size_t V0 = static_cast<size_t>(src_dim[0]) * src_dim[1] * src_dim[2] * src_dim[3];
  size_t V1 = static_cast<size_t>(dst_dim[0]) * dst_dim[1] * dst_dim[2] * dst_dim[3];
  size_t Vh0 = V0 / 2;
  size_t Vh1 = V1 / 2;
  size_t expected_src_bytes = V0 * 4 * LINK_BYTES;

  struct stat st;
  if (stat(src_path.c_str(), &st) != 0 || static_cast<size_t>(st.st_size) != expected_src_bytes) {
    std::fprintf(stderr, "Source gauge size mismatch for %s\n", src_path.c_str());
    return 1;
  }

  FILE *fs = std::fopen(src_path.c_str(), "rb");
  if (!fs) {
    std::fprintf(stderr, "Failed to open source %s\n", src_path.c_str());
    return 1;
  }

  std::string tmp_path = dst_path + ".tmp";
  FILE *fd = std::fopen(tmp_path.c_str(), "wb");
  if (!fd) {
    std::fclose(fs);
    std::fprintf(stderr, "Failed to open destination %s\n", tmp_path.c_str());
    return 1;
  }

  EVP_MD_CTX *mdctx = EVP_MD_CTX_new();
  EVP_DigestInit_ex(mdctx, EVP_sha256(), nullptr);

  std::vector<char> src_dir_buf(V0 * LINK_BYTES);
  constexpr size_t CHUNK_LINKS = 32768;
  std::vector<char> out_chunk(CHUNK_LINKS * LINK_BYTES);
  size_t written_since_flush = 0;
  int out_fd = fileno(fd);

  const int Lx0 = src_dim[0], Ly0 = src_dim[1], Lz0 = src_dim[2], Lt0 = src_dim[3];
  const int Lx1 = dst_dim[0], Ly1 = dst_dim[1], Lz1 = dst_dim[2], Lt1 = dst_dim[3];

  for (int mu = 0; mu < 4; ++mu) {
    if (std::fread(src_dir_buf.data(), 1, V0 * LINK_BYTES, fs) != V0 * LINK_BYTES) {
      std::fprintf(stderr, "Short read on mu=%d\n", mu);
      EVP_MD_CTX_free(mdctx);
      std::fclose(fs);
      std::fclose(fd);
      return 1;
    }

    for (int parity_dst = 0; parity_dst < 2; ++parity_dst) {
      size_t chunk_pos = 0;
      for (size_t idx_cb_dst = 0; idx_cb_dst < Vh1; ++idx_cb_dst) {
        size_t full_dst = 2 * idx_cb_dst + static_cast<size_t>(parity_dst);
        int x = static_cast<int>(full_dst % Lx1);
        size_t t_tmp = full_dst / Lx1;
        int y = static_cast<int>(t_tmp % Ly1);
        t_tmp /= Ly1;
        int z = static_cast<int>(t_tmp % Lz1);
        int t = static_cast<int>(t_tmp / Lz1);
        if (((x + y + z + t) & 1) != parity_dst) {
          x ^= 1;
        }
        int x0 = x % Lx0;
        int y0 = y % Ly0;
        int z0 = z % Lz0;
        int t0 = t % Lt0;
        int parity_src = (x0 + y0 + z0 + t0) & 1;
        size_t full_src = ((static_cast<size_t>(t0) * Lz0 + z0) * Ly0 + y0) * Lx0 + x0;
        size_t idx_cb_src = full_src / 2;
        size_t src_idx = static_cast<size_t>(parity_src) * Vh0 + idx_cb_src;

        const char *src_link = src_dir_buf.data() + src_idx * LINK_BYTES;
        char *dst_link = out_chunk.data() + chunk_pos * LINK_BYTES;
        std::memcpy(dst_link, src_link, LINK_BYTES);

        if (mu == 3 && ((t0 == (Lt0 - 1)) != (t == (Lt1 - 1)))) {
          double *vals = reinterpret_cast<double *>(dst_link);
          for (int k = 0; k < 18; ++k) {
            vals[k] = -vals[k];
          }
        }

        chunk_pos++;
        if (chunk_pos == CHUNK_LINKS) {
          size_t bytes = chunk_pos * LINK_BYTES;
          if (std::fwrite(out_chunk.data(), 1, bytes, fd) != bytes) return 1;
          EVP_DigestUpdate(mdctx, out_chunk.data(), bytes);
          written_since_flush += bytes;
          chunk_pos = 0;
          if (written_since_flush >= (64ULL << 20)) {
            std::fflush(fd);
            (void)fdatasync(out_fd);
            fadvise_dontneed_fd(out_fd, 0);
            written_since_flush = 0;
          }
        }
      }
      if (chunk_pos > 0) {
        size_t bytes = chunk_pos * LINK_BYTES;
        if (std::fwrite(out_chunk.data(), 1, bytes, fd) != bytes) return 1;
        EVP_DigestUpdate(mdctx, out_chunk.data(), bytes);
        written_since_flush += bytes;
      }
    }
  }

  std::fflush(fd);
  (void)fsync(out_fd);
  fadvise_dontneed_fd(out_fd, 0);
  fadvise_dontneed_fd(fileno(fs), 0);
  std::fclose(fd);
  std::fclose(fs);

  if (std::rename(tmp_path.c_str(), dst_path.c_str()) != 0) {
    std::fprintf(stderr, "Failed to rename %s -> %s\n", tmp_path.c_str(), dst_path.c_str());
    return 1;
  }

  unsigned char digest[EVP_MAX_MD_SIZE];
  unsigned int dlen = 0;
  EVP_DigestFinal_ex(mdctx, digest, &dlen);
  EVP_MD_CTX_free(mdctx);
  std::string sha_hex = to_hex(digest, dlen);

  std::printf("{\"src\": \"%s\", \"dst\": \"%s\", \"src_dim\": [%d, %d, %d, %d], \"dst_dim\": [%d, %d, %d, %d], \"bytes\": %zu, \"sha256\": \"%s\"}\n",
              src_path.c_str(), dst_path.c_str(), src_dim[0], src_dim[1], src_dim[2], src_dim[3],
              dst_dim[0], dst_dim[1], dst_dim[2], dst_dim[3], V1 * 4 * LINK_BYTES, sha_hex.c_str());
  return 0;
}
