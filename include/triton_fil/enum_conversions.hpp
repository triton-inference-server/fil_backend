#pragma once

#include <cuml/fil/fil.h>

#include <string>

namespace triton { namespace backend { namespace fil {

ML::fil::algo_t name_to_tl_algo(std::string);
ML::fil::storage_type_t name_to_storage_type(std::string);

}}}
