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
  TRITONSERVER_Error * err;
  if (config.Find(param_name.c_str(), &value)) {
    std::string string_rep;

    err = value.MemberAsString("string_value", &string_rep);
    if (err != nullptr) {
      throw(TritonException(err));
    }

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
