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

#include <cuml/fil/fil.h>
#include <triton/backend/backend_common.h>
#include <triton/core/tritonserver.h>

#include <exception>
#include <string>
#include <triton_fil/config.hpp>
#include <triton_fil/enum_conversions.hpp>

namespace triton { namespace backend { namespace fil {

ML::fil::treelite_params_t
tl_params_from_config(triton::common::TritonJson::Value& config)
{
  ML::fil::treelite_params_t out_params;
  out_params.algo =
      name_to_tl_algo(retrieve_param<std::string>(config, "algo"));
  out_params.storage_type =
      name_to_storage_type(retrieve_param<std::string>(config, "storage_type"));
  out_params.output_class = retrieve_param<bool>(config, "output_class");
  out_params.threshold = retrieve_param<float>(config, "threshold");
  out_params.blocks_per_sm = retrieve_param<int>(config, "blocks_per_sm");

  return out_params;
};

}}}  // namespace triton::backend::fil
