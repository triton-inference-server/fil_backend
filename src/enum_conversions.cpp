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
