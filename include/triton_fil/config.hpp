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

#include <sstream>
#include <type_traits>

#include <cuml/fil/fil.h>
#include <triton/backend/backend_common.h>
#include <triton/core/tritonserver.h>
#include <triton_fil/exceptions.hpp>

namespace triton { namespace backend { namespace fil {

template <typename out_type>
out_type retrieve_param(
    triton::common::TritonJson::Value& config,
    const std::string& param_name) {
  common::TritonJson::Value value;
  out_type output;
  if (config.Find(param_name.c_str(), &value)) {
    std::string string_rep;

    triton_check(value.MemberAsString("string_value", &string_rep));

    std::istringstream input_stream{string_rep};
    if (std::is_same<out_type, bool>::value) {
      input_stream >> std::boolalpha >> output;
    } else {
      input_stream >> output;
    }
    if (input_stream.fail()) {
      throw TritonException(
        TRITONSERVER_errorcode_enum::TRITONSERVER_ERROR_INVALID_ARG,
        ("Bad input for parameter " + param_name).c_str()
      );
    } else {
      return output;
    }
  } else {
    throw TritonException(
      TRITONSERVER_errorcode_enum::TRITONSERVER_ERROR_INVALID_ARG,
      ("Required parameter " + param_name + " not found in config").c_str()
    );
  }
}

ML::fil::treelite_params_t
tl_params_from_config(triton::common::TritonJson::Value& config);

}}}
