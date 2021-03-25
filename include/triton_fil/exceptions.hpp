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
#include <exception>
#include <string>
#include <triton/core/tritonserver.h>

namespace triton { namespace backend { namespace fil {

class TritonException : public std::exception {
  private:
    TRITONSERVER_Error * error_;
  public:
    TritonException() : error_(TRITONSERVER_ErrorNew(
        TRITONSERVER_errorcode_enum::TRITONSERVER_ERROR_UNKNOWN,
        "encountered unknown error")) {}

    TritonException(TRITONSERVER_Error_Code code, const std::string& msg) :
      error_(TRITONSERVER_ErrorNew(code, msg.c_str())) {}

    TritonException(TRITONSERVER_Error_Code code, const char * msg) :
      error_{TRITONSERVER_ErrorNew(code, msg)} {}
    TritonException(TRITONSERVER_Error * prev_error) : error_(prev_error) {}

    const char * what() const noexcept {
      return TRITONSERVER_ErrorMessage(error_);
    }

    TRITONSERVER_Error * error() {
      return error_;
    }
};

void triton_check(TRITONSERVER_Error * err);

}}}
