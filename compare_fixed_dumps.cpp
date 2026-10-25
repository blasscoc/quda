// Compare the raw FP64 arrays emitted by QUDA_FIXED_DUMP_PREFIX.
// Build: c++ -O2 -std=c++17 src/compare_fixed_dumps.cpp -o build/compare_fixed_dumps
// Usage: build/compare_fixed_dumps reference.bin candidate.bin [--require-identical]
// Exit 0: finite comparison (and identical bits if requested); 1: failed check; 2: I/O/usage error.
// The caller must check the companion .meta files describe matching fields and seeds.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>

int main(int argc, char **argv)
{
  const bool require_identical = argc == 4 && std::string(argv[3]) == "--require-identical";
  if ((argc != 3 && argc != 4) || (argc == 4 && !require_identical)) {
    std::fprintf(stderr, "Usage: %s reference.bin candidate.bin [--require-identical]\n", argv[0]);
    return 2;
  }
  static_assert(sizeof(double) == sizeof(uint64_t) && std::numeric_limits<double>::is_iec559,
                "This tool requires IEEE-754 binary64 doubles");
  try {
    std::ifstream reference(argv[1], std::ios::binary | std::ios::ate);
    std::ifstream candidate(argv[2], std::ios::binary | std::ios::ate);
    if (!reference || !candidate) throw std::runtime_error("Cannot open both dump files");
    const auto bytes = reference.tellg();
    if (bytes <= 0 || bytes != candidate.tellg() || bytes % sizeof(double) != 0)
      throw std::runtime_error("Dump files must have equal, nonzero sizes divisible by sizeof(double)");
    reference.seekg(0);
    candidate.seekg(0);
    const uint64_t count = static_cast<uint64_t>(bytes) / sizeof(double);
    uint64_t finite_reference = 0, finite_candidate = 0, finite_pairs = 0, equal_values = 0, identical_bits = 0;
    long double reference_squared = 0, difference_squared = 0, max_absolute = 0;
    uint64_t max_absolute_index = 0;
    std::array<double, 8192> a, b;
    for (uint64_t offset = 0; offset < count;) {
      const size_t chunk = static_cast<size_t>(std::min<uint64_t>(a.size(), count - offset));
      const auto chunk_bytes = static_cast<std::streamsize>(chunk * sizeof(double));
      if (!reference.read(reinterpret_cast<char *>(a.data()), chunk_bytes)
          || !candidate.read(reinterpret_cast<char *>(b.data()), chunk_bytes))
        throw std::runtime_error("Short read or I/O error while reading dump files");
      for (size_t i = 0; i < chunk; i++) {
        const bool finite_a = std::isfinite(a[i]), finite_b = std::isfinite(b[i]);
        finite_reference += finite_a;
        finite_candidate += finite_b;
        equal_values += a[i] == b[i];
        identical_bits += std::memcmp(&a[i], &b[i], sizeof(double)) == 0;
        if (finite_a && finite_b) {
          finite_pairs++;
          const long double ref = a[i];
          const long double difference = static_cast<long double>(b[i]) - ref;
          reference_squared += ref * ref;
          difference_squared += difference * difference;
          if (std::fabs(difference) > max_absolute) {
            max_absolute = std::fabs(difference);
            max_absolute_index = offset + i;
          }
        }
      }
      offset += chunk;
    }
    const bool all_finite = finite_pairs == count;
    const long double nan = std::numeric_limits<long double>::quiet_NaN();
    const long double relative_l2 = !all_finite ? nan : reference_squared > 0
      ? std::sqrt(difference_squared / reference_squared) : difference_squared == 0
      ? 0 : std::numeric_limits<long double>::infinity();
    std::printf("count=%llu finite_reference=%llu finite_candidate=%llu finite_pairs=%llu "
                "equal_values=%llu identical_bits=%llu bitwise_identical=%d "
                "rel_l2=%.17Le max_abs=%.17Le max_abs_index=%llu\n",
                static_cast<unsigned long long>(count), static_cast<unsigned long long>(finite_reference),
                static_cast<unsigned long long>(finite_candidate), static_cast<unsigned long long>(finite_pairs),
                static_cast<unsigned long long>(equal_values), static_cast<unsigned long long>(identical_bits),
                identical_bits == count, relative_l2, all_finite ? max_absolute : nan,
                static_cast<unsigned long long>(max_absolute_index));
    return !all_finite || (require_identical && identical_bits != count) ? 1 : 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "compare_fixed_dumps: %s\n", error.what());
    return 2;
  }
}
