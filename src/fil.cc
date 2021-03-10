#include <iostream>
#include <memory>
#include <thread>

#include "cuml/fil/fil.h"
#include "raft/cuda_utils.cuh"
#include "triton/backend/backend_common.h"



namespace triton { namespace backend { namespace fil {

int foobar()
{
  std::cout << "Hello CMake from inside!" << std::endl;
  common::TritonJson::Value parameters;
  ML::fil::algo_t foo;

  return 0;
}

}}}  // namespace triton::backend::fil

int main(int argc, char *argv[])
{
  std::cout << "Hello CMake from outside!" << std::endl;
  triton::backend::fil::foobar();

  return 0;
}
