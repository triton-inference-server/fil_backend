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
