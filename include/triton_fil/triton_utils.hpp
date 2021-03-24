#pragma once
#include <memory>
#include <string>
#include <triton/backend/backend_common.h>
#include <triton/core/tritonserver.h>
#include <triton_fil/exceptions.hpp>

namespace triton { namespace backend { namespace fil {

/** Get the name of the given backend */
std::string get_backend_name(TRITONBACKEND_Backend& backend);

/** Check if the backend version API that this backend was compiled against is
 * supported by Triton
 */
bool check_backend_version(TRITONBACKEND_Backend& backend);

/** Get the name of the given model */
std::string get_model_name(TRITONBACKEND_Model& model);

/** Get the version of the given model */
uint64_t get_model_version(TRITONBACKEND_Model& model);

/** Get JSON configuration for given model */
std::unique_ptr<common::TritonJson::Value> get_model_config(TRITONBACKEND_Model& model);

/** Get Triton server object for given model */
TRITONSERVER_Server* get_server(TRITONBACKEND_Model& model);

template <typename ModelStateType>
void set_model_state(TRITONBACKEND_Model& model,
                     std::unique_ptr<ModelStateType> model_state) {
  TRITONSERVER_Error * err = TRITONBACKEND_ModelSetState(
    &model,
    reinterpret_cast<void*>(model_state.release())
  );
  if (err != nullptr) {
    throw(TritonException(err));
  }
  return model;
}

}}}
