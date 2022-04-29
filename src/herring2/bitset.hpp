#pragma once
#include <stdint.h>
#include <type_traits>
#include <herring2/detail/index_type.hpp>
#include <herring2/detail/raw_array.hpp>
#include <herring2/gpu_support.hpp>

namespace herring {
template<raw_index_t num_bits, typename storage_t>
struct bitset {
  using storage_type = storage_t;
  using index_type = detail::index_type<!GPU_ENABLED && DEBUG_ENABLED>;

  auto constexpr static const bin_width = raw_index_t{
    detail::index_type<false>{sizeof(storage_type) * 8}
  };
  auto constexpr static const bins = (
    num_bits / bin_width + (num_bits % bin_width != 0)
  );

  auto constexpr size() const {
    return num_bits;
  }

  HOST DEVICE bitset() {
    // Zero-initialize
    for (auto i = raw_index_t{}; i < bins; ++i) {
      data_[i] = storage_type{};
    }
  }

  // Standard bit-wise mutators and accessor
  auto& set(index_type index) {
    data_[bin_from_index(index)] |= mask_in_bin(index);
    return *this;
  }
  auto& clear(index_type index) {
    data_[bin_from_index(index)] &= ~mask_in_bin(index);
    return *this;
  }
  auto test(index_type index) {
    return ((data_[bin_from_index(index)] & mask_in_bin(index)) != 0);
  }
  auto& flip() {
    for (auto i = raw_index_t{}; i < bins; ++i) {
      data_[i] = ~data_[i];
    }
    return *this;
  }

  // Conversion operators for single-bin bitsets
  template <
    raw_index_t bin_count = bins, 
    typename = typename std::enable_if_t<bin_count == raw_index_t{1}>
  >
  operator storage_type() const noexcept { return data_[0]; }
  template <
    raw_index_t bin_count = bins, 
    typename = typename std::enable_if_t<bin_count == raw_index_t{1}>
  >
  operator storage_type&() noexcept { return data_[0]; }

  // Bit-wise boolean operations
  template<raw_index_t N, typename T>
  friend bitset<N, T> operator&(
    bitset<num_bits, storage_t> const& lhs,
    bitset<num_bits, storage_t> const& rhs
  );
  template<raw_index_t N, typename T>
  friend bitset<N, T> operator|(
    bitset<num_bits, storage_t> const& lhs,
    bitset<num_bits, storage_t> const& rhs
  );
  template<raw_index_t N, typename T>
  friend bitset<N, T> operator^(
    bitset<num_bits, storage_t> const& lhs,
    bitset<num_bits, storage_t> const& rhs
  );
  auto& operator&=(bitset<num_bits, storage_t> const& other) {
    for (auto i = raw_index_t{}; i < bins; ++i) {
      data_[i] &= other.data_[i];
    }
    return *this;
  }
  auto& operator|=(bitset<num_bits, storage_t> const& other) {
    for (auto i = raw_index_t{}; i < bins; ++i) {
      data_[i] |= other.data_[i];
    }
    return *this;
  }
  auto& operator^=(bitset<num_bits, storage_t> const& other) {
    for (auto i = raw_index_t{}; i < bins; ++i) {
      data_[i] ^= other.data_[i];
    }
    return *this;
  }
  auto operator~() const {
    return bitset<num_bits, storage_t>(*this).flip();
  }
 private:
  detail::raw_array<storage_t, bins> data_;

  auto mask_in_bin(raw_index_t index) {
    return storage_t{1} << (index % bin_width);
  }

  auto bin_from_index(raw_index_t index) {
    return index / bin_width;
  }
};

template <raw_index_t num_bits, typename storage_t>
auto operator&(
  bitset<num_bits, storage_t> const& lhs,
  bitset<num_bits, storage_t> const& rhs
) {
  auto result = bitset<num_bits, storage_t>(lhs);
  result &= rhs;
  return result;
}

template <raw_index_t num_bits, typename storage_t>
auto operator|(
  bitset<num_bits, storage_t> const& lhs,
  bitset<num_bits, storage_t> const& rhs
) {
  auto result = bitset<num_bits, storage_t>(lhs);
  result |= rhs;
  return result;
}

template <raw_index_t num_bits, typename storage_t>
auto operator^(
  bitset<num_bits, storage_t> const& lhs,
  bitset<num_bits, storage_t> const& rhs
) {
  auto result = bitset<num_bits, storage_t>(lhs);
  result |= rhs;
  return result;
}

}
