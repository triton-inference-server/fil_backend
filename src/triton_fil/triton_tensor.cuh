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
#include <string>
#include <type_traits>
#include <vector>
#include <raft/cudart_utils.h>
#include <triton/backend/backend_common.h>
#include <triton/core/tritonserver.h>
#include <triton_fil/exceptions.h>
#include <triton_fil/buffers.cuh>

namespace triton { namespace backend { namespace fil {

using byte = char; // C++17: Use std::byte

/**
 * @brief Representation of a tensor constructed from data provided by Triton
 * TritonTensors are either constructed from input buffers provided by Triton
 * or wrap output buffers where Triton eventually expects data from this tensor
 * to end up. TritonTensors are constructed so as to minimize data copying
 * and new allocations where possible. Any newly-allocated buffers will be
 * freed on destruction.
 */
template<typename T>
class TritonTensor {
  using non_const_T = typename std::remove_const<T>::type;

 private:
  std::string name_;  //!< Name reported by Triton for these data
  std::vector<int64_t> shape_;  //!< Shape of array represented by buffer
  TRITONSERVER_DataType dtype_;  //!< Data type contained by this buffer
  uint64_t size_bytes_;  //!< Size of buffer in bytes
  bool is_owner_;  //!< Whether or not buffer is responsible for deallocation
  cudaStream_t stream_;  //!< Cuda stream for any needed synchronization
  T* buffer;  //!< Pointer to underlying data
  std::vector<RawOutputBuffer> final_buffers;  //!< Where data should be copied
                                              //!< back to on "sync" calls if
                                              //!< needed
  

 public:
  TritonTensor() : name_{},
                   shape_{},
                   dtype_{TRITONSERVER_TYPE_FP32},
                   size_bytes_{0},
                   is_owner_{false},
                   stream_{0},
                   buffer{nullptr},
                   final_buffers{} {}

  template<typename U=T>
  TritonTensor(
    const std::vector<RawInputBuffer>& buffers,
    const std::string& name,
    const std::vector<int64_t>& shape,
    TRITONSERVER_DataType dtype,
    TRITONSERVER_MemoryType target_memory,
    cudaStream_t stream
  ) : name_{name},
      shape_{shape},
      dtype_{dtype_},
      size_bytes_{
        sizeof(T) *
        product(shape_)
      },
      is_owner_{
        buffers.size() > 1  // non-contiguous
        || buffers[0].memory_type != target_memory  // non-device memory
      },
      stream_{stream},
      buffer{[&] {
        if (is_owner_) {
          auto ptr_d = allocate_device_memory<byte>(size_bytes_);
          auto cur_head = ptr_d;
          for (auto& buffer_ : buffers) {
            try {
              raft::copy(
                cur_head,
                reinterpret_cast<const byte *>(buffer_.data),
                buffer_.size_bytes,
                stream_
              );
            } catch (const raft::cuda_error& err) {
              throw TritonException(
                TRITONSERVER_errorcode_enum::TRITONSERVER_ERROR_INTERNAL,
                err.what()
              );
            }
            cur_head += buffer_.size_bytes;
          }
          return reinterpret_cast<T*>(ptr_d);
        } else {
          return reinterpret_cast<T*>(buffers[0].data);
        }
      }()},
      final_buffers{} {}

  template<typename U=T>
  TritonTensor(
    std::vector<RawOutputBuffer>&& buffers,
    const std::string& name,
    const std::vector<int64_t>& shape,
    const TRITONSERVER_DataType dtype,
    TRITONSERVER_MemoryType target_memory,
    raft::handle_t& handle
  ) : name_{name},
      shape_{shape},
      dtype_{dtype_},
      size_bytes_{
        sizeof(T) *
        product(shape_)
      },
      is_owner_{
        buffers.size() > 1  // non-contiguous
        || buffers[0].memory_type != target_memory  // non-device memory
      },
      stream_{handle.get_stream()},
      buffer{[&] {
        if (is_owner_) {
          return allocate_device_memory<non_const_T>(size_bytes_ /
              sizeof(non_const_T));
        } else {
          return reinterpret_cast<non_const_T*>(buffers[0].data);
        }
      }()},
      final_buffers{buffers} {}

  ~TritonTensor() {
    if (is_owner_) {
      // Allowing const_cast here because if this object owns the buffer, we
      // originally allocated it non-const then cast to const for consistency
      // with Triton-provided buffers. Since this is happening in the
      // destructor, removing const at this point should be safe.
      cudaFree(reinterpret_cast<void*>(const_cast<non_const_T*>(buffer)));
    }
  }

  TritonTensor(
    const TritonTensor<T>& other
  ) : name_{other.name_},
      shape_{other.shape_},
      dtype_{other.dtype_},
      size_bytes_{other.size_bytes_},
      is_owner_{true},
      stream_{other.stream_},
      buffer{[&] {
        non_const_T * ptr_d =
          allocate_device_memory<non_const_T>(other.size());
        try {
          raft::copy(ptr_d, other.buffer, other.size(), stream_);
        } catch (const raft::cuda_error& err) {
          throw TritonException(
            TRITONSERVER_errorcode_enum::TRITONSERVER_ERROR_INTERNAL,
            err.what()
          );
        }
        return ptr_d;
      }()},
      final_buffers{other.final_buffers} {}

  TritonTensor(TritonTensor<T>&& other) noexcept : name_{other.name_},
                                                   shape_{other.shape_},
                                                   dtype_{other.dtype_},
                                                   size_bytes_{other.size_bytes_},
                                                   is_owner_{other.is_owner_},
                                                   stream_{other.stream_},
                                                   buffer{other.buffer},
                                                   final_buffers{std::move(other.final_buffers)} {
    other.is_owner_ = false;
    other.buffer = nullptr;
  }

  TritonTensor<T>& operator=(const TritonTensor<T>& other) {
    return *this = TritonTensor<T>(other);
  }

  TritonTensor<T>& operator=(TritonTensor<T>&& other) noexcept {
    if (this != &other) {
      return *this = TritonTensor<T>(other);
    } else {
      return *this;
    }
  }

  /**
   * @brief Copy data from this tensor back to underlying output buffers (if
   * any)
   */
  void sync() {
    auto head = reinterpret_cast<
      typename std::conditional<
        std::is_const<T>::value, const byte*, byte*
      >::type
    >(buffer);
    for (auto& out_buffer : final_buffers) {
      try {
        raft::copy(
          reinterpret_cast<byte*>(out_buffer.data),
          head,
          out_buffer.size_bytes,
          stream_
        );
      } catch (const raft::cuda_error& err) {
        throw TritonException(
          TRITONSERVER_errorcode_enum::TRITONSERVER_ERROR_INTERNAL,
          err.what()
        );
      }
      head += out_buffer.size_bytes;
    }
  }

  /**
   * @brief Get pointer to underlying data
   */
  T* data() {
    return buffer;
  }

  /**
   * @brief Get the name of this input/output as reported by Triton
   */
  const std::string& name() {
    return name_;
  }
  /**
   * @brief Get the dimensions of this tensor
   */
  const std::vector<int64_t>& shape() {
    return shape_;
  }
  /**
   * @brief Get an enum representing the datatype stored in this tensor
   */
  TRITONSERVER_DataType dtype() {
    return dtype_;
  }
  /**
   * @brief Get the total number of elements in this tensor
   */
  int64_t size() const {
    return product(shape_);
  }
  /**
   * @brief Get the size of the data stored by this tensor in bytes
   */
  uint64_t size_bytes() const {
    return size_bytes_;
  }
  /**
   * @brief Get the CUDA stream used for operations on this tensor
   */
  const cudaStream_t& get_stream() const {
    return stream_;
  }
  /**
   * @brief Set the CUDA stream used for operations on this tensor
   */
  void set_stream(cudaStream_t new_stream) {
    cudaStreamSynchronize(stream_);
    stream_ = new_stream;
  }
};

}}}
