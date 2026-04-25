//
// Created by Elijah Crain on 12/31/25.
//

#ifndef POKER_HASHUTIL_HPP
#define POKER_HASHUTIL_HPP
#include <cstdint>
#include <cstring>

// Hashing utilities using boost::hash_combine style
namespace HashUtil {

constexpr uint64_t GOLDEN_RATIO = 0x9e3779b97f4a7c15ULL;  // 2^64 / phi

// Combine a running hash with a new value (boost::hash_combine style)
constexpr uint64_t combineHash(uint64_t seed, uint64_t value) {
    seed ^= value + GOLDEN_RATIO + (seed << 12) + (seed >> 4);
    return seed;
}

// Convert float to uint64_t bits for hashing
inline uint64_t floatBits(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(float));
    return static_cast<uint64_t>(bits);
}
}  // namespace HashUtil

#endif //POKER_HASHUTIL_HPP