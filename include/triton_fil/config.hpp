#pragma once

#include <sstream>
#include <type_traits>

#include <cuml/fil/fil.h>  // TODO: forward declaration
#include <triton/backend/backend_common.h>
#include <triton/core/tritonserver.h>

namespace triton { namespace backend { namespace fil {

// TODO: Use exceptions
template <typename out_type>
TRITONSERVER_Error* retrieve_param(
    triton::common::TritonJson::Value& config,
    const std::string& param_name,
    out_type& output) {
  common::TritonJson::Value value;
  if (config.Find(param_name.c_str(), &value)) {
    std::string string_rep;
    RETURN_IF_ERROR(
      value.MemberAsString("string_value", &string_rep)
    );
    std::istringstream input_stream{string_rep};
    if (std::is_same<out_type, bool>::value) {
      input_stream >> std::boolalpha >> output;
    } else {
      input_stream >> output;
    }
    if (input_stream.fail()) {
      return TRITONSERVER_ErrorNew(
        TRITONSERVER_errorcode_enum::TRITONSERVER_ERROR_INVALID_ARG,
        ("Bad input for parameter " + param_name).c_str()
      );
    } else {
      return nullptr;
    }
  } else {
    return TRITONSERVER_ErrorNew(
      TRITONSERVER_errorcode_enum::TRITONSERVER_ERROR_INVALID_ARG,
      ("Required parameter " + param_name + " not found in config").c_str()
    );
  }
}

// TODO: Use std::pair return
TRITONSERVER_Error* 
tl_params_from_config(triton::common::TritonJson::Value& config,
                      ML::fil::treelite_params_t& out_params);

}}}
