// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#pragma once
#include <iostream>
#include <numeric>
#include <string>
#include <type_traits>
#include <vector>
#include <raft/cudart_utils.h>
#include <triton/backend/backend_common.h>
#include <triton/core/tritonserver.h>
#include <triton_fil/exceptions.h>

namespace triton { namespace backend { namespace fil {

namespace {
  template<typename T>
  T product_of_elems(const std::vector<T>& array) {
    return std::accumulate(std::begin(array),
                           std::end(array),
                           1,
                           std::multiplies<>());
  }

  template<typename T>
  T* allocate_device_memory(size_t bytes) {
    T* ptr_d;
    raft::allocate(ptr_d, bytes);
    return ptr_d;
  }
}

template<typename T>
class TritonBuffer {
  using non_const_T = typename std::remove_const<T>::type;

 private:
  std::string name_;  //!< Name reported by Triton for these data
  std::vector<int64_t> shape_;  //!< Shape of array represented by buffer
  TRITONSERVER_DataType dtype_;  //!< Data type contained by this buffer
  uint64_t size_bytes_;  //!< Size of buffer in bytes
  bool is_owner_;  //!< Whether or not buffer is responsible for deallocation
  cudaStream_t stream_;  //!< Cuda stream for any needed synchronization
  T* buffer;  //!< Pointer to underlying data
  non_const_T* final_buffer;  //!< Where data should be copied back to on
                              //!< "sync" calls if needed
  

 public:
  TritonBuffer() : name_{},
                   shape_{},
                   dtype_{TRITONSERVER_TYPE_FP32},
                   size_bytes_{0},
                   is_owner_{false},
                   stream_{0},
                   buffer{nullptr},
                   final_buffer{nullptr} {}

  TritonBuffer(
    T* buffer,
    const std::string& name,
    const std::vector<int64_t>& shape,
    TRITONSERVER_DataType dtype
  ) : name_{name},
      shape_{shape},
      dtype_{dtype_},
      size_bytes_{
        sizeof(T) *
        product_of_elems(shape_)
      },
      is_owner_{false},
      stream_{0},
      buffer{buffer},
      final_buffer{nullptr} {}

  TritonBuffer(
    const std::string& name,
    const std::vector<int64_t>& shape,
    TRITONSERVER_DataType dtype,
    cudaStream_t stream,
    non_const_T* final_buffer = nullptr
  ) : name_{name},
      shape_{shape},
      dtype_{dtype_},
      size_bytes_{
        sizeof(T) *
        product_of_elems(shape_)
      },
      is_owner_{true},
      stream_{stream},
      buffer{
        allocate_device_memory<non_const_T>(size_bytes_)
      },
      final_buffer{final_buffer} {}

  TritonBuffer(
    T* input_buffer,
    const std::string& name,
    const std::vector<int64_t>& shape,
    TRITONSERVER_DataType dtype,
    cudaStream_t stream,
    non_const_T* final_buffer = nullptr
  ) : name_{name},
      shape_{shape},
      dtype_{dtype_},
      size_bytes_{
        sizeof(T) *
        product_of_elems(shape_)
      },
      is_owner_{true},
      stream_{stream},
      buffer{[&] {
        non_const_T * ptr_d = allocate_device_memory<non_const_T>(size_bytes_);
        try {
          raft::copy(ptr_d, input_buffer, product_of_elems(shape_), stream_);
        } catch (const raft::cuda_error& err) {
          throw TritonException(
            TRITONSERVER_errorcode_enum::TRITONSERVER_ERROR_INTERNAL,
            err.what()
          );
        }
        return ptr_d;
      }()},
      final_buffer{final_buffer} {}

  ~TritonBuffer() {
    if (is_owner_) {
      // Allowing const_cast here because if this object owns the buffer, we
      // originally allocated it non-const then cast to const for consistency
      // with Triton-provided buffers. Since this is happening in the
      // destructor, removing const at this point should be safe.
      cudaFree(reinterpret_cast<void*>(const_cast<non_const_T*>(buffer)));
    }
  }

  TritonBuffer(
    const TritonBuffer<T>& other
  ) : name_{other.name_},
      shape_{other.shape_},
      dtype_{other.dtype_},
      size_bytes_{other.size_bytes_},
      is_owner_{other.is_owner_},
      stream_{other.stream_},
      buffer{[&] {
        non_const_T * ptr_d =
          allocate_device_memory<non_const_T>(size_bytes_);
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
      final_buffer{other.final_buffer} {}

  TritonBuffer(TritonBuffer<T>&& other) noexcept : name_{other.name_},
                                                   shape_{other.shape_},
                                                   dtype_{other.dtype_},
                                                   size_bytes_{other.size_bytes_},
                                                   is_owner_{other.is_owner_},
                                                   stream_{other.stream_},
                                                   buffer{other.buffer},
                                                   final_buffer{other.final_buffer} {
    other.is_owner_ = false;
    other.buffer = nullptr;
  }

  TritonBuffer<T>& operator=(const TritonBuffer<T>& other) {
    return *this = TritonBuffer<T>(other);
  }

  TritonBuffer<T>& operator=(TritonBuffer<T>&& other) noexcept {
    if (this != &other) {
      return *this = TritonBuffer<T>(other);
    } else {
      return *this;
    }
  }

  void sync() {
    if (final_buffer != nullptr) {
      try {
        raft::copy(final_buffer, buffer, size(), stream_);
      } catch (const raft::cuda_error& err) {
        throw TritonException(
          TRITONSERVER_errorcode_enum::TRITONSERVER_ERROR_INTERNAL,
          err.what()
        );
      }
    }
  }

  T* data() {
    return buffer;
  }

  const std::string& name() {
    return name_;
  }
  const std::vector<int64_t>& shape() {
    return shape_;
  }
  TRITONSERVER_DataType dtype() {
    return dtype_;
  }
  int64_t size() const {
    return product_of_elems(shape_);
  }
  uint64_t size_bytes() const {
    return size_bytes_;
  }
  const cudaStream_t& get_stream() const {
    return stream_;
  }
  void set_stream(cudaStream_t new_stream) {
    cudaStreamSynchronize(stream_);
    stream_ = new_stream;
  }
};

}}}
