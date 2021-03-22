#include <cuml/fil/fil.h>

#include <exception>
#include <string>

namespace triton { namespace backend { namespace fil {

class BadEnumName : public std::exception {
  virtual const char* what() const throw() { return "Unknown enum name"; }
} bad_enum_exception;

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
  // TODO: Switch to "optional" return
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
