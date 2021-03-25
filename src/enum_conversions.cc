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
#include <triton/core/tritonserver.h>

#include <string>
#include <triton_fil/exceptions.hpp>

namespace triton { namespace backend { namespace fil {

TritonException bad_enum_exception(
    TRITONSERVER_errorcode_enum::TRITONSERVER_ERROR_INVALID_ARG,
    "Unknown enum name");

ML::fil::algo_t
name_to_tl_algo(std::string name)
{
  if (name == "ALGO_AUTO") {
    return ML::fil::algo_t::ALGO_AUTO;
  }
  if (name == "NAIVE") {
    return ML::fil::algo_t::NAIVE;
  }
  if (name == "TREE_REORG") {
    return ML::fil::algo_t::TREE_REORG;
  }
  if (name == "BATCH_TREE_REORG") {
    return ML::fil::algo_t::BATCH_TREE_REORG;
  }
  throw bad_enum_exception;
}

ML::fil::storage_type_t
name_to_storage_type(std::string name)
{
  if (name == "AUTO") {
    return ML::fil::storage_type_t::AUTO;
  }
  if (name == "DENSE") {
    return ML::fil::storage_type_t::DENSE;
  }
  if (name == "SPARSE") {
    return ML::fil::storage_type_t::SPARSE;
  }
  if (name == "SPARSE8") {
    return ML::fil::storage_type_t::SPARSE8;
  }
  throw bad_enum_exception;
}

}}}  // namespace triton::backend::fil
