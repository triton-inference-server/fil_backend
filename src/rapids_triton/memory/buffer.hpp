/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <variant>

#ifdef TRITON_ENABLE_GPU
#include <cuda_runtime_api.h>

#include <rapids_triton/memory/detail/gpu_only/copy.hpp>
#include <rapids_triton/memory/detail/gpu_only/owned_device_buffer.hpp>
#else
#include <rapids_triton/cpu_only/cuda_runtime_replacement.hpp>
#include <rapids_triton/memory/detail/cpu_only/copy.hpp>
#include <rapids_triton/memory/detail/cpu_only/owned_device_buffer.hpp>
#endif

#include <rapids_triton/build_control.hpp>
#include <rapids_triton/exceptions.hpp>
#include <rapids_triton/memory/resource.hpp>
#include <rapids_triton/memory/types.hpp>
#include <rapids_triton/triton/device.hpp>
#include <rapids_triton/triton/logging.hpp>

namespace triton { namespace backend { namespace rapids {
template <typename T>
struct Buffer {
  using size_type = std::size_t;
  using value_type = T;

  using h_buffer = T*;
  using d_buffer = T*;
  using owned_h_buffer = std::unique_ptr<T[]>;
  using owned_d_buffer = detail::owned_device_buffer<T, IS_GPU_BUILD>;
  using data_store =
      std::variant<h_buffer, d_buffer, owned_h_buffer, owned_d_buffer>;

  Buffer() noexcept
      : device_{}, data_{std::in_place_index<0>, nullptr}, size_{}, stream_{}
  {
  }

  /**
   * @brief Construct buffer of given size in given memory location (either
   * on host or on device)
   * A buffer constructed in this way is owning and will release allocated
   * resources on deletion
   */
  Buffer(
      size_type size, MemoryType memory_type = DeviceMemory,
      device_id_t device = 0, cudaStream_t stream = 0)
      : device_{device}, data_{allocate(size, device, memory_type, stream)},
        size_{size}, stream_{stream}
  {
    if constexpr (!IS_GPU_BUILD) {
      if (memory_type == DeviceMemory) {
        throw TritonException(
            Error::Internal, "Cannot use device buffer in non-GPU build");
      }
    }
  }

  /**
   * @brief Construct buffer from given source in given memory location (either
   * on host or on device)
   * A buffer constructed in this way is non-owning; the caller is
   * responsible for freeing any resources associated with the input pointer
   */
  Buffer(
      T* input_data, size_type size, MemoryType memory_type = DeviceMemory,
      device_id_t device = 0, cudaStream_t stream = 0)
      : device_{device}, data_{[&memory_type, &input_data]() {
          auto result = data_store{};
          if (memory_type == HostMemory) {
            result = data_store{std::in_place_index<0>, input_data};
          } else {
            if constexpr (!IS_GPU_BUILD) {
              throw TritonException(
                  Error::Internal, "Cannot use device buffer in non-GPU build");
            }
            result = data_store{std::in_place_index<1>, input_data};
          }
          return result;
        }()},
        size_{size}, stream_{stream}
  {
  }

  /**
   * @brief Construct one buffer from another in the given memory location
   * (either on host or on device)
   * A buffer constructed in this way is owning and will copy the data from
   * the original location
   */
  Buffer(Buffer<T> const& other, MemoryType memory_type, device_id_t device = 0)
      : device_{device}, data_([&other, &memory_type, &device]() {
          auto result =
              allocate(other.size_, device, memory_type, other.stream_);
          copy(result, other.data_, other.size_, other.stream_);
          return result;
        }()),
        size_{other.size_}, stream_{other.stream_}
  {
  }

  /**
   * @brief Create owning copy of existing buffer
   * The memory type of this new buffer will be the same as the original
   */
  Buffer(Buffer<T> const& other)
      : Buffer(other, other.mem_type(), other.device())
  {
  }

  Buffer(Buffer<T>&& other, MemoryType memory_type)
      : device_{other.device()}, data_{[&other, memory_type]() {
          data_store result;
          if (memory_type == other.mem_type()) {
            result = std::move(other.data_);
          } else {
            result = allocate(
                other.size_, memory_type, other.device(), other.stream());
            copy(result, other.data_, other.size_, other.stream_);
          }
          return result;
        }()},
        size_{other.size_}, stream_{other.stream_}
  {
  }

  Buffer(Buffer<T>&& other) = default;

  Buffer<T>& operator=(Buffer<T>&& other) = default;

  ~Buffer() {}

  /**
   * @brief Return where memory for this buffer is located (host or device)
   */
  auto mem_type() const noexcept
  {
    return data_.index() % 2 == 0 ? HostMemory : DeviceMemory;
  }

  /**
   * @brief Return number of elements in buffer
   */
  auto size() const noexcept { return size_; }

  /**
   * @brief Return pointer to data stored in buffer
   */
  auto* data() const noexcept { return get_raw_ptr(data_); }

  auto device() const noexcept { return device_; }

  /**
   * @brief Return CUDA stream associated with this buffer
   */
  auto stream() const noexcept { return stream_; }

  void stream_synchronize() const
  {
    if constexpr (IS_GPU_BUILD) {
      if (mem_type() == DeviceMemory) {
        cuda_check(cudaStreamSynchronize(stream_));
      }
    }
  }

  /**
   * @brief Set CUDA stream for this buffer to new value
   *
   * @warning This method calls cudaStreamSynchronize on the old stream
   * before updating. Be aware of performance implications and try to avoid
   * interactions between buffers on different streams where possible.
   */
  void set_stream(cudaStream_t new_stream)
  {
    stream_synchronize();
    stream_ = new_stream;
  }

 private:
  device_id_t device_;
  data_store data_;
  size_type size_;
  cudaStream_t stream_;

  // Helper function for accessing raw pointer to underlying data of
  // data_store
  static auto* get_raw_ptr(data_store const& ptr) noexcept
  {
    /* Switch statement is an optimization relative to std::visit to avoid
     * vtable overhead for a small number of alternatives */
    auto* result = static_cast<T*>(nullptr);
    switch (ptr.index()) {
      case 0:
        result = std::get<0>(ptr);
        break;
      case 1:
        result = std::get<1>(ptr);
        break;
      case 2:
        result = std::get<2>(ptr).get();
        break;
      case 3:
        result = std::get<3>(ptr).get();
        break;
    }
    return result;
  }

  // Helper function for allocating memory in constructors
  static auto allocate(
      size_type size, device_id_t device = 0,
      MemoryType memory_type = DeviceMemory, cudaStream_t stream = 0)
  {
    auto result = data_store{};
    if (memory_type == DeviceMemory) {
      if constexpr (IS_GPU_BUILD) {
        result = data_store{owned_d_buffer{
            device,
            size,
            stream,
        }};
      } else {
        throw TritonException(
            Error::Internal,
            "DeviceMemory requested in CPU-only build of FIL backend");
      }
    } else {
      result = std::make_unique<T[]>(size);
    }
    return result;
  }

  // Helper function for copying memory in constructors, where there are
  // stronger guarantees on conditions that would otherwise need to be
  // checked
  static void copy(
      data_store const& dst, data_store const& src, size_type len,
      cudaStream_t stream)
  {
    // This function will only be called in constructors, so we allow a
    // const_cast here to perform the initial copy of data from a
    // Buffer<T const> to a newly-created Buffer<T const>
    auto raw_dst = const_cast<std::remove_const_t<T>*>(get_raw_ptr(dst));
    auto raw_src = get_raw_ptr(src);

    auto dst_mem_type = dst.index() % 2 == 0 ? HostMemory : DeviceMemory;
    auto src_mem_type = src.index() % 2 == 0 ? HostMemory : DeviceMemory;

    detail::copy(raw_dst, raw_src, len, stream, dst_mem_type, src_mem_type);
  }
};

/**
 * @brief Copy data from one Buffer to another
 *
 * @param dst The destination buffer
 * @param src The source buffer
 * @param dst_begin The offset from the beginning of the destination buffer
 * at which to begin copying to.
 * @param src_begin The offset from the beginning of the source buffer
 * at which to begin copying from.
 * @param src_end The offset from the beginning of the source buffer
 * before which to end copying from.
 */
template <typename T, typename U>
void
copy(
    Buffer<T>& dst, Buffer<U> const& src,
    typename Buffer<T>::size_type dst_begin,
    typename Buffer<U>::size_type src_begin,
    typename Buffer<U>::size_type src_end)
{
  if (dst.stream() != src.stream()) {
    dst.set_stream(src.stream());
  }
  auto len = src_end - src_begin;
  if (len < 0 || src_end > src.size() || len > dst.size() - dst_begin) {
    throw TritonException(Error::Internal, "bad copy between buffers");
  }

  auto raw_dst = dst.data() + dst_begin;
  auto raw_src = src.data() + src_begin;

  detail::copy(
      raw_dst, raw_src, len, dst.stream(), dst.mem_type(), src.mem_type());
}

template <typename T, typename U>
void
copy(Buffer<T>& dst, Buffer<U> const& src)
{
  copy(dst, src, 0, 0, src.size());
}

template <typename T, typename U>
void
copy(
    Buffer<T>& dst, Buffer<U> const& src,
    typename Buffer<T>::size_type dst_begin)
{
  copy(dst, src, dst_begin, 0, src.size());
}

template <typename T, typename U>
void
copy(
    Buffer<T>& dst, Buffer<U> const& src,
    typename Buffer<U>::size_type src_begin,
    typename Buffer<U>::size_type src_end)
{
  copy(dst, src, 0, src_begin, src_end);
}
}}}  // namespace triton::backend::rapids
