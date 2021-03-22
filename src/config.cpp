#include <cuml/fil/fil.h>  // TODO: forward declaration
#include <triton/backend/backend_common.h>
#include <triton/core/tritonserver.h>

#include <exception>
#include <string>
#include <triton_fil/config.hpp>
#include <triton_fil/enum_conversions.hpp>

namespace triton { namespace backend { namespace fil {

TRITONSERVER_Error*
tl_params_from_config(
    triton::common::TritonJson::Value& config,
    ML::fil::treelite_params_t& out_params)
{
  common::TritonJson::Value value;

  std::string algo_name;
  RETURN_IF_ERROR(retrieve_param(config, "algo", algo_name));

  std::string storage_type_name;
  RETURN_IF_ERROR(retrieve_param(config, "storage_type", storage_type_name));

  RETURN_IF_ERROR(
      retrieve_param(config, "output_class", out_params.output_class));

  RETURN_IF_ERROR(retrieve_param(config, "threshold", out_params.threshold));

  RETURN_IF_ERROR(
      retrieve_param(config, "blocks_per_sm", out_params.blocks_per_sm));

  try {
    out_params.algo = name_to_tl_algo(algo_name);
  }
  catch (const std::exception& err) {
    RETURN_IF_ERROR(TRITONSERVER_ErrorNew(
        TRITONSERVER_errorcode_enum::TRITONSERVER_ERROR_INVALID_ARG,
        err.what()));
  }
  try {
    out_params.storage_type = name_to_storage_type(storage_type_name);
  }
  catch (const std::exception& err) {
    RETURN_IF_ERROR(TRITONSERVER_ErrorNew(
        TRITONSERVER_errorcode_enum::TRITONSERVER_ERROR_INVALID_ARG,
        err.what()));
  }

  return nullptr;
};

}}}  // namespace triton::backend::fil
