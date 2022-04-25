#pragma once
#include <memory>
#include <variant>
#include <herring2/detail/const_agnostic.hpp>
#include <herring2/detail/copy.hpp>
#include <herring2/cuda_stream.hpp>
#include <herring2/detail/non_owning_buffer.hpp>
#include <herring2/detail/owning_buffer.hpp>
#include <herring2/device_id.hpp>
#include <herring2/device_type.hpp>
#include <herring2/exceptions.hpp>
#include <herring2/gpu_support.hpp>

namespace herring {
/**
 * @brief A container which may or may not own its own data on host or device
 *
 */
template<typename T>
struct buffer {
  using size_type = index_t;
  using value_type = T;

  using data_store = std::variant<
    non_owning_buffer<device_type::cpu, T>, non_owning_buffer<device_type::gpu, T>, owning_buffer<device_type::cpu, T>, owning_buffer<device_type::gpu, T>
  >;

  buffer() : device_{}, data_{}, size_{} {}

  /** Construct non-initialized owning buffer */
  buffer(size_type size,
         device_type mem_type = device_type::cpu,
         int device = 0,
         cuda_stream stream = 0) 
    : device_{[mem_type, &device]() {
      auto result = device_id_variant{};
      switch (mem_type) {
        case device_type::cpu: result = device_id<device_type::cpu>{device}; break;
        case device_type::gpu: result = device_id<device_type::gpu>{device}; break;
      }
      return result;
    }()},
    data_{[this, mem_type, size, stream]() {
      auto result = data_store{};
      switch (mem_type) {
        case device_type::cpu:
          result = owning_buffer<device_type::cpu, T>{size};
          break;
        case device_type::gpu:
          result = owning_buffer<device_type::gpu, T>{std::get<1>(device_), size, stream};
          break;
      }
      return result;
    }()},
    size_{size},
    cached_ptr {[this](){
      auto result = static_cast<T*>(nullptr);
      switch(data_.index()) {
        case 0: result = std::get<0>(data_).get(); break;
        case 1: result = std::get<1>(data_).get(); break;
        case 2: result = std::get<2>(data_).get(); break;
        case 3: result = std::get<3>(data_).get(); break;
      }
      return result;
    }()}
  {
  }

  /** Construct non-owning buffer */
  buffer(T* input_data,
         size_type size,
         device_type mem_type = device_type::cpu,
         int device = 0)
    : device_{[mem_type, &device]() {
      auto result = device_id_variant{};
      switch (mem_type) {
        case device_type::cpu:
          result = device_id<device_type::cpu>{device};
          break;
        case device_type::gpu:
          result = device_id<device_type::gpu>{device};
          break;
      }
      return result;
    }()},
    data_{[this, input_data, mem_type]() {
      auto result = data_store{};
      switch (mem_type) {
        case device_type::cpu:
          result = non_owning_buffer<device_type::cpu, T>{input_data};
          break;
        case device_type::gpu:
          result = non_owning_buffer<device_type::gpu, T>{input_data};
          break;
      }
      return result;
    }()},
    size_{size},
    cached_ptr {[this](){
      auto result = static_cast<T*>(nullptr);
      switch(data_.index()) {
        case 0: result = std::get<0>(data_).get(); break;
        case 1: result = std::get<1>(data_).get(); break;
        case 2: result = std::get<2>(data_).get(); break;
        case 3: result = std::get<3>(data_).get(); break;
      }
      return result;
    }()}
  {
  }

  /**
   * @brief Construct one buffer from another in the given memory location
   * (either on host or on device)
   * A buffer constructed in this way is owning and will copy the data from
   * the original location
   */
  buffer(buffer<T> const& other, device_type mem_type, int device = 0, cuda_stream stream=cuda_stream{})
    : device_{[mem_type, &device]() {
      auto result = device_id_variant{};
      switch (mem_type) {
        case device_type::cpu:
          result = device_id<device_type::cpu>{device};
          break;
        case device_type::gpu:
          result = device_id<device_type::gpu>{device};
          break;
      }
      return result;
    }()},
    data_{[this, &other, mem_type, device, stream]() {
      auto result = data_store{};
      auto result_data = static_cast<T*>(nullptr);
      if (mem_type == device_type::cpu) {
        auto buf = owning_buffer<device_type::cpu, T>(other.size());
        result_data = buf.get();
        result = std::move(buf);
      } else if (mem_type==device_type::gpu) {
        auto buf = owning_buffer<device_type::gpu, T>(std::get<1>(device_), other.size(), stream);
        result_data = buf.get();
        result = std::move(buf);
      }
      copy(result_data, other.data(), other.size(), mem_type, other.memory_type(), stream);
      return result;
    }()},
    size_{other.size()},
    cached_ptr {[this](){
      auto result = static_cast<T*>(nullptr);
      switch(data_.index()) {
        case 0: result = std::get<0>(data_).get(); break;
        case 1: result = std::get<1>(data_).get(); break;
        case 2: result = std::get<2>(data_).get(); break;
        case 3: result = std::get<3>(data_).get(); break;
      }
      return result;
    }()}
  {
  }

  /**
   * @brief Create owning copy of existing buffer
   * The memory type of this new buffer will be the same as the original
   */
  buffer(buffer<T> const& other) : buffer(other, other.memory_type(), other.device_index()) {}

  /**
   * @brief Create owning copy of existing buffer with given stream
   * The memory type of this new buffer will be the same as the original
   */
  buffer(buffer<T> const& other, cuda_stream stream) : buffer(other, other.memory_type(), other.device_index(), stream) {}

  /**
   * @brief Move from existing buffer unless a copy is necessary based on
   * memory location
   */
  buffer(buffer<T>&& other, device_type mem_type, int device, cuda_stream stream)
    : device_{[mem_type, &device]() {
      auto result = device_id_variant{};
      switch (mem_type) {
        case device_type::cpu:
          result = device_id<device_type::cpu>{device};
          break;
        case device_type::gpu:
          result = device_id<device_type::gpu>{device};
          break;
      }
      return result;
    }()},
    data_{[&other, mem_type, device, stream]() {
      auto result = data_store{};
      if (mem_type == other.memory_type() && device == other.device_index()) {
        result  = std::move(other.data_);
      } else {
        auto* result_data = static_cast<T*>(nullptr);
        if (mem_type == device_type::cpu) {
          auto buf = owning_buffer<device_type::cpu, T>{other.size()};
          result_data = buf.get();
          result = std::move(buf);
        } else if (mem_type == device_type::gpu) {
          auto buf = owning_buffer<device_type::gpu, T>{device, other.size(), stream};
          result_data = buf.get();
          result = std::move(buf);
        }
        copy(result_data, other.data(), other.size(), mem_type, other.memory_type(), stream);
      }
      return result;
    }()},
    size_{other.size()},
    cached_ptr {[this](){
      auto result = static_cast<T*>(nullptr);
      switch(data_.index()) {
        case 0: result = std::get<0>(data_).get(); break;
        case 1: result = std::get<1>(data_).get(); break;
        case 2: result = std::get<2>(data_).get(); break;
        case 3: result = std::get<3>(data_).get(); break;
      }
      return result;
    }()}
  {
  }
  buffer(buffer<T>&& other, device_type mem_type, int device)
    : buffer{std::move(other), mem_type, device, cuda_stream{}}
  {
  }
  buffer(buffer<T>&& other, device_type mem_type)
    : buffer{std::move(other), mem_type, 0, cuda_stream{}}
  {
  }

  buffer(buffer<T>&& other) = default;
  buffer<T>& operator=(buffer<T>&& other) = default;

  auto size() const noexcept { return size_; }
  HOST DEVICE auto* data() const noexcept {
    return cached_ptr;
  }
  auto memory_type() const noexcept {
    auto result = device_type{};
    if (device_.index() == 0) {
      result = device_type::cpu;
    } else {
      result = device_type::gpu;
    }
    return result;
  }

  auto device_index() const noexcept {
    auto result = int{};
    switch(device_.index()) {
      case 0: result = std::get<0>(device_).value(); break;
      case 1: result = std::get<1>(device_).value(); break;
    }
    return result;
  }

 private:
  device_id_variant device_;
  data_store data_;
  size_type size_;
  T* cached_ptr;

};

template<bool bounds_check, typename T, typename U>
const_agnostic_same_t<T, U> copy(buffer<T>& dst, buffer<U> const& src, index_t dst_offset, index_t src_offset, index_t size, cuda_stream stream) {
  if constexpr (bounds_check) {
    if (src.size() - src_offset < size || dst.size() - dst_offset < size) {
      throw out_of_bounds("Attempted copy to or from buffer of inadequate size");
    }
  }
  copy(dst.data() + dst_offset, src.data() + src_offset, size, dst.memory_type(), src.memory_type(), stream);
}

template<bool bounds_check, typename T, typename U>
const_agnostic_same_t<T, U> copy(buffer<T>& dst, buffer<U> const& src, cuda_stream stream) {
  copy<bounds_check>(dst, src, 0, 0, src.size(), stream);
}
template<bool bounds_check, typename T, typename U>
const_agnostic_same_t<T, U> copy(buffer<T>& dst, buffer<U> const& src) {
  copy<bounds_check>(dst, src, 0, 0, src.size(), cuda_stream{});
}

}  // namespace herring
