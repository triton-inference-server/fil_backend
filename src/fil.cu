// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
#include <raft/handle.hpp>
#include <treelite/c_api.h>

#include <exception>
#include <limits>
#include <memory>
#include <sstream>
#include <thread>
#include <type_traits>

#include "triton/backend/backend_common.h"
#include "triton/backend/backend_model.h"
#include "triton/backend/backend_model_instance.h"

namespace triton { namespace backend { namespace fil {


#define GUARDED_RESPOND_IF_ERROR(RESPONSES, IDX, X)                     \
  do {                                                                  \
    if ((RESPONSES)[IDX] != nullptr) {                                  \
      TRITONSERVER_Error* err__ = (X);                                  \
      if (err__ != nullptr) {                                           \
        LOG_IF_ERROR(                                                   \
            TRITONBACKEND_ResponseSend(                                 \
                (RESPONSES)[IDX], TRITONSERVER_RESPONSE_COMPLETE_FINAL, \
                err__),                                                 \
            "failed to send error response");                           \
        (RESPONSES)[IDX] = nullptr;                                     \
        TRITONSERVER_ErrorDelete(err__);                                \
      }                                                                 \
    }                                                                   \
  } while (false)


class BadEnumName: public std::exception {
  virtual const char* what() const throw() {
    return "Unknown enum name";
  }
} bad_enum_exception;

ML::fil::algo_t name_to_tl_algo(std::string name) {
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

ML::fil::storage_type_t name_to_storage_type(
  std::string name
) {
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

template <typename target_type, typename source_type>
target_type narrow_cast(source_type input) {
  if (input > std::numeric_limits<target_type>::max() ) {
      return std::numeric_limits<target_type>::max();
  } else if (input < std::numeric_limits<target_type>::min() ) {
      return std::numeric_limits<target_type>::min();
  } else {
      return static_cast<target_type>(input);
  }
}

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

//
// ModelState
//
// State associated with a model that is using this backend. An object
// of this class is created and associated with each
// TRITONBACKEND_Model.
//
class ModelState : public BackendModel {
 public:
  static TRITONSERVER_Error* Create(
      TRITONBACKEND_Model* triton_model, ModelState** state);

  TRITONSERVER_Error* LoadModel(
      std::string artifact_name,
      const TRITONSERVER_InstanceGroupKind instance_group_kind,
      const int32_t instance_group_device_id);
  TRITONSERVER_Error* UnloadModel();

  // Get the handle to the TRITONBACKEND model.
  TRITONBACKEND_Model* TritonModel() { return triton_model_; }

  // Get the name and version of the model.
  const std::string& Name() const { return name_; }
  uint64_t Version() const { return version_; }

  // Does this model support batching in the first dimension. This
  // function should not be called until after the model is completely
  // loaded.
  TRITONSERVER_Error* SupportsFirstDimBatching(bool* supports);

  // Validate that model configuration is supported by this backend.
  TRITONSERVER_Error* ValidateModelConfig();

  ML::fil::treelite_params_t tl_params;
  void* treelite_handle;

 private:
  ModelState(
      TRITONSERVER_Server* triton_server, TRITONBACKEND_Model* triton_model,
      const char* name, const uint64_t version,
      common::TritonJson::Value&& model_config);

  TRITONSERVER_Server* triton_server_;
  TRITONBACKEND_Model* triton_model_;
  const std::string name_;
  const uint64_t version_;
  common::TritonJson::Value model_config_;

  bool supports_batching_initialized_;
  bool supports_batching_;
};

TRITONSERVER_Error* 
tl_params_from_config(
    triton::common::TritonJson::Value& config,
    ML::fil::treelite_params_t& out_params)
{
  common::TritonJson::Value value;

  std::string algo_name;
  RETURN_IF_ERROR(
    retrieve_param(config, "algo", algo_name)
  );

  std::string storage_type_name;
  RETURN_IF_ERROR(
    retrieve_param(config, "storage_type", storage_type_name)
  );

  RETURN_IF_ERROR(
    retrieve_param(config, "output_class", out_params.output_class)
  );

  RETURN_IF_ERROR(
    retrieve_param(config, "threshold", out_params.threshold)
  );

  RETURN_IF_ERROR(
    retrieve_param(config, "blocks_per_sm", out_params.blocks_per_sm)
  );

  try {
    out_params.algo = name_to_tl_algo(algo_name);
  } catch (const std::exception& err) {
    RETURN_IF_ERROR(
      TRITONSERVER_ErrorNew(
        TRITONSERVER_errorcode_enum::TRITONSERVER_ERROR_INVALID_ARG,
        err.what()
      )
    );
  }
  try {
    out_params.storage_type = name_to_storage_type(storage_type_name);
  } catch (const std::exception& err) {
    RETURN_IF_ERROR(
      TRITONSERVER_ErrorNew(
        TRITONSERVER_errorcode_enum::TRITONSERVER_ERROR_INVALID_ARG,
        err.what()
      )
    );
  }

  return nullptr;
};

TRITONSERVER_Error*
ModelState::Create(TRITONBACKEND_Model* triton_model, ModelState** state)
{
  TRITONSERVER_Message* config_message;
  RETURN_IF_ERROR(TRITONBACKEND_ModelConfig(
      triton_model, 1 /* config_version */, &config_message));

  // We can get the model configuration as a json string from
  // config_message, parse it with our favorite json parser to create
  // DOM that we can access when we need to example the
  // configuration. We use TritonJson, which is a wrapper that returns
  // nice errors (currently the underlying implementation is
  // rapidjson... but others could be added). You can use any json
  // parser you prefer.
  const char* buffer;
  size_t byte_size;
  RETURN_IF_ERROR(
      TRITONSERVER_MessageSerializeToJson(config_message, &buffer, &byte_size));

  common::TritonJson::Value model_config;
  TRITONSERVER_Error* err = model_config.Parse(buffer, byte_size);
  RETURN_IF_ERROR(TRITONSERVER_MessageDelete(config_message));
  RETURN_IF_ERROR(err);

  triton::common::TritonJson::Value config;

  const char* model_name;
  RETURN_IF_ERROR(TRITONBACKEND_ModelName(triton_model, &model_name));

  uint64_t model_version;
  RETURN_IF_ERROR(TRITONBACKEND_ModelVersion(triton_model, &model_version));

  TRITONSERVER_Server* triton_server;
  RETURN_IF_ERROR(TRITONBACKEND_ModelServer(triton_model, &triton_server));

  *state = new ModelState(
      triton_server, triton_model, model_name, model_version,
      std::move(model_config));

  (*state)->ModelConfig().Find("parameters", &config);
  RETURN_IF_ERROR(tl_params_from_config(config, (*state)->tl_params));
  return nullptr;  // success
}

TRITONSERVER_Error*
ModelState::LoadModel(
    std::string artifact_name,
    const TRITONSERVER_InstanceGroupKind instance_group_kind,
    const int32_t instance_group_device_id)
{
  if (artifact_name.empty()) {
    artifact_name = "xgboost.model";
  }
  std::string model_path =
      JoinPath({RepositoryPath(), std::to_string(Version()), artifact_name});
  {
    bool is_dir;
    RETURN_IF_ERROR(IsDirectory(model_path, &is_dir));
    if (is_dir) {
      model_path = JoinPath({model_path, "xgboost.model"});
    }
  }

  {
    bool exists;
    RETURN_IF_ERROR(FileExists(model_path, &exists));
    RETURN_ERROR_IF_FALSE(
        exists, TRITONSERVER_ERROR_UNAVAILABLE,
        std::string("unable to find '") + model_path +
            "' for model instance '" + Name() + "'");
  }

  TreeliteLoadXGBoostModel(model_path.c_str(), &treelite_handle);
  return nullptr;
}

TRITONSERVER_Error*
ModelState::UnloadModel() {
  // TODO
  /* if (treelite_handle != nullptr) {
    TreeliteFreeModel(treelite_handle);
  } */
  return nullptr;
}

extern "C" TRITONSERVER_Error* unload_treelite_model(ModelState * state) {
  state->UnloadModel();
  return nullptr;
}

ModelState::ModelState(
    TRITONSERVER_Server* triton_server, TRITONBACKEND_Model* triton_model,
    const char* name, const uint64_t version,
    common::TritonJson::Value&& model_config)
    : BackendModel(triton_model), triton_model_(triton_model), name_(name),
      version_(version), model_config_(std::move(model_config)),
      supports_batching_initialized_(false), supports_batching_(false),
      treelite_handle(nullptr)
{
}

TRITONSERVER_Error*
ModelState::SupportsFirstDimBatching(bool* supports)
{
  // We can't determine this during model initialization because
  // TRITONSERVER_ServerModelBatchProperties can't be called until the
  // model is loaded. So we just cache it here.
  if (!supports_batching_initialized_) {
    uint32_t flags = 0;
    RETURN_IF_ERROR(TRITONSERVER_ServerModelBatchProperties(
        triton_server_, name_.c_str(), version_, &flags, nullptr /* voidp */));
    supports_batching_ = ((flags & TRITONSERVER_BATCH_FIRST_DIM) != 0);
    supports_batching_initialized_ = true;
  }

  *supports = supports_batching_;
  return nullptr;  // success
}

TRITONSERVER_Error*
ModelState::ValidateModelConfig()
{
  // TODO
  return nullptr;  // success
}

//
// ModelInstanceState
//
// State associated with a model instance. An object of this class is
// created and associated with each TRITONBACKEND_ModelInstance.
//
class ModelInstanceState : public BackendModelInstance {
 public:
  static TRITONSERVER_Error* Create(
      ModelState* model_state,
      TRITONBACKEND_ModelInstance* triton_model_instance,
      ModelInstanceState** state);

  // Get the handle to the TRITONBACKEND model instance.
  TRITONBACKEND_ModelInstance* TritonModelInstance()
  {
    return triton_model_instance_;
  }

  // Get the name, kind and device ID of the instance.
  const std::string& Name() const { return name_; }
  TRITONSERVER_InstanceGroupKind Kind() const { return kind_; }
  int32_t DeviceId() const { return device_id_; }

  // Get the state of the model that corresponds to this instance.
  ModelState* StateForModel() const { return model_state_; }
  TRITONSERVER_Error * predict(
      const float* data,
      float* preds,
      size_t num_rows,
      bool predict_proba = false);

 private:
  ModelInstanceState(
      ModelState* model_state,
      TRITONBACKEND_ModelInstance* triton_model_instance, const char* name,
      const TRITONSERVER_InstanceGroupKind kind, const int32_t device_id);

  ModelState* model_state_;
  TRITONBACKEND_ModelInstance* triton_model_instance_;
  const std::string name_;
  const TRITONSERVER_InstanceGroupKind kind_;
  const int32_t device_id_;

  ML::fil::forest_t fil_forest;
  std::unique_ptr<raft::handle_t> handle;
};

TRITONSERVER_Error*
ModelInstanceState::Create(
    ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance,
    ModelInstanceState** state)
{
  (*state)->handle = std::make_unique<raft::handle_t>();
  (*state)->handle->set_stream((*state)->stream_);

  const char* instance_name;
  RETURN_IF_ERROR(
      TRITONBACKEND_ModelInstanceName(triton_model_instance, &instance_name));

  TRITONSERVER_InstanceGroupKind instance_kind;
  RETURN_IF_ERROR(
      TRITONBACKEND_ModelInstanceKind(triton_model_instance, &instance_kind));

  int32_t instance_id;
  RETURN_IF_ERROR(
      TRITONBACKEND_ModelInstanceDeviceId(triton_model_instance, &instance_id));

  *state = new ModelInstanceState(
      model_state, triton_model_instance, instance_name, instance_kind,
      instance_id);
  return nullptr;  // success
}

TRITONSERVER_Error * ModelInstanceState::predict(
    const float* data,
    float* preds,
    size_t num_rows,
    bool predict_proba) {
  ML::fil::predict(*handle, fil_forest, preds, data, num_rows, predict_proba);
  return nullptr;
}

ModelInstanceState::ModelInstanceState(
    ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance,
    const char* name, const TRITONSERVER_InstanceGroupKind kind,
    const int32_t device_id)
    : BackendModelInstance(model_state, triton_model_instance),
      model_state_(model_state), triton_model_instance_(triton_model_instance),
      name_(name), kind_(kind), device_id_(device_id)
{
  THROW_IF_BACKEND_INSTANCE_ERROR(
      model_state_->LoadModel(ArtifactFilename(), Kind(), DeviceId()));
  ML::fil::from_treelite(
      *handle,
      &fil_forest,
      model_state_->treelite_handle,
      &(model_state_->tl_params));
}

extern "C" TRITONSERVER_Error* fil_predict(
    TRITONBACKEND_ModelInstance * instance,
    const float* input_buffer,
    float * output_buffer,
    size_t rows) {

  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(instance, &vstate));
  ModelInstanceState* instance_state =
      reinterpret_cast<ModelInstanceState*>(vstate);

  instance_state->predict(input_buffer, output_buffer, rows);

  return nullptr;
}

/////////////

extern "C" {

// Implementing TRITONBACKEND_Initialize is optional. The backend
// should initialize any global state that is intended to be shared
// across all models and model instances that use the backend.
TRITONSERVER_Error*
TRITONBACKEND_Initialize(TRITONBACKEND_Backend* backend)
{
  const char* cname;
  RETURN_IF_ERROR(TRITONBACKEND_BackendName(backend, &cname));
  std::string name(cname);

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("TRITONBACKEND_Initialize: ") + name).c_str());

  // We should check the backend API version that Triton supports
  // vs. what this backend was compiled against.
  uint32_t api_version_major, api_version_minor;
  RETURN_IF_ERROR(
      TRITONBACKEND_ApiVersion(&api_version_major, &api_version_minor));

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("Triton TRITONBACKEND API version: ") +
       std::to_string(api_version_major) + "." +
       std::to_string(api_version_minor))
          .c_str());
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("'") + name + "' TRITONBACKEND API version: " +
       std::to_string(TRITONBACKEND_API_VERSION_MAJOR) + "." +
       std::to_string(TRITONBACKEND_API_VERSION_MINOR))
          .c_str());

  if ((api_version_major != TRITONBACKEND_API_VERSION_MAJOR) ||
      (api_version_minor < TRITONBACKEND_API_VERSION_MINOR)) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_UNSUPPORTED,
        "triton backend API version does not support this backend");
  }

  // The backend configuration may contain information needed by the
  // backend, such a command-line arguments. This backend doesn't use
  // any such configuration but we print whatever is available.
  TRITONSERVER_Message* backend_config_message;
  RETURN_IF_ERROR(
      TRITONBACKEND_BackendConfig(backend, &backend_config_message));

  const char* buffer;
  size_t byte_size;
  RETURN_IF_ERROR(TRITONSERVER_MessageSerializeToJson(
      backend_config_message, &buffer, &byte_size));
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("backend configuration:\n") + buffer).c_str());

  // If we have any global backend state we create and set it here. We
  // don't need anything for this backend but for demonstration
  // purposes we just create something...
  std::string* state = new std::string("backend state");
  RETURN_IF_ERROR(
      TRITONBACKEND_BackendSetState(backend, reinterpret_cast<void*>(state)));

  return nullptr;  // success
}

// Implementing TRITONBACKEND_Finalize is optional unless state is set
// using TRITONBACKEND_BackendSetState. The backend must free this
// state and perform any other global cleanup.
TRITONSERVER_Error*
TRITONBACKEND_Finalize(TRITONBACKEND_Backend* backend)
{
  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_BackendState(backend, &vstate));
  std::string* state = reinterpret_cast<std::string*>(vstate);

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("TRITONBACKEND_Finalize: state is '") + *state + "'")
          .c_str());

  delete state;

  return nullptr;  // success
}

// Implementing TRITONBACKEND_ModelInitialize is optional. The backend
// should initialize any state that is intended to be shared across
// all instances of the model.
TRITONSERVER_Error*
TRITONBACKEND_ModelInitialize(TRITONBACKEND_Model* model)
{
  const char* cname;
  RETURN_IF_ERROR(TRITONBACKEND_ModelName(model, &cname));
  std::string name(cname);

  uint64_t version;
  RETURN_IF_ERROR(TRITONBACKEND_ModelVersion(model, &version));

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("TRITONBACKEND_ModelInitialize: ") + name + " (version " +
       std::to_string(version) + ")")
          .c_str());

  // Can get location of the model artifacts. Normally we would need
  // to check the artifact type to make sure it was something we can
  // handle... but we are just going to log the location so we don't
  // need the check. We would use the location if we wanted to load
  // something from the model's repo.
  TRITONBACKEND_ArtifactType artifact_type;
  const char* clocation;
  RETURN_IF_ERROR(
      TRITONBACKEND_ModelRepository(model, &artifact_type, &clocation));
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("Repository location: ") + clocation).c_str());

  // The model can access the backend as well... here we can access
  // the backend global state.
  TRITONBACKEND_Backend* backend;
  RETURN_IF_ERROR(TRITONBACKEND_ModelBackend(model, &backend));

  void* vbackendstate;
  RETURN_IF_ERROR(TRITONBACKEND_BackendState(backend, &vbackendstate));
  std::string* backend_state = reinterpret_cast<std::string*>(vbackendstate);

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("backend state is '") + *backend_state + "'").c_str());

  // With each model we create a ModelState object and associate it
  // with the TRITONBACKEND_Model.
  ModelState* model_state;
  RETURN_IF_ERROR(ModelState::Create(model, &model_state));
  RETURN_IF_ERROR(
      TRITONBACKEND_ModelSetState(model, reinterpret_cast<void*>(model_state)));

  // One of the primary things to do in ModelInitialize is to examine
  // the model configuration to ensure that it is something that this
  // backend can support. If not, returning an error from this
  // function will prevent the model from loading.
  RETURN_IF_ERROR(model_state->ValidateModelConfig());

  return nullptr;  // success
}

// Implementing TRITONBACKEND_ModelFinalize is optional unless state
// is set using TRITONBACKEND_ModelSetState. The backend must free
// this state and perform any other cleanup.
TRITONSERVER_Error*
TRITONBACKEND_ModelFinalize(TRITONBACKEND_Model* model)
{
  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vstate));
  ModelState* model_state = reinterpret_cast<ModelState*>(vstate);
  RETURN_IF_ERROR(unload_treelite_model(model_state));

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO, "TRITONBACKEND_ModelFinalize: delete model state");

  delete model_state;

  return nullptr;  // success
}

// Implementing TRITONBACKEND_ModelInstanceInitialize is optional. The
// backend should initialize any state that is required for a model
// instance.
TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceInitialize(TRITONBACKEND_ModelInstance* instance)
{
  return nullptr;  // success
}

// Implementing TRITONBACKEND_ModelInstanceFinalize is optional unless
// state is set using TRITONBACKEND_ModelInstanceSetState. The backend
// must free this state and perform any other cleanup.
TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceFinalize(TRITONBACKEND_ModelInstance* instance)
{
  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(instance, &vstate));
  ModelInstanceState* instance_state =
      reinterpret_cast<ModelInstanceState*>(vstate);

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      "TRITONBACKEND_ModelInstanceFinalize: delete instance state");

  delete instance_state;

  return nullptr;  // success
}

// Implementing TRITONBACKEND_ModelInstanceExecute is required.
TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceExecute(
    TRITONBACKEND_ModelInstance* instance, TRITONBACKEND_Request** requests,
    const uint32_t request_count)
{
  ModelInstanceState* instance_state;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(
      instance, reinterpret_cast<void**>(&instance_state)));
  ModelState* model_state = instance_state->StateForModel();

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("model ") + model_state->Name() + ", instance " +
       instance_state->Name() + ", executing " + std::to_string(request_count) +
       " requests")
          .c_str());

  bool supports_batching = false;
  RETURN_IF_ERROR(model_state->SupportsFirstDimBatching(&supports_batching));

  std::vector<TRITONBACKEND_Response*> responses;
  responses.reserve(request_count);

  for (uint32_t r = 0; r < request_count; ++r) {
    TRITONBACKEND_Request* request = requests[r];

    TRITONBACKEND_Response* response;
    RETURN_IF_ERROR(TRITONBACKEND_ResponseNew(&response, request));
    responses.push_back(response);
  }

  uint64_t total_batch_size = 0;

  // After this point we take ownership of 'requests', which means
  // that a response must be sent for every request. If something does
  // go wrong in processing a particular request then we send an error
  // response just for the specific request.

  // For simplicity we just process each request separately... in
  // general a backend should try to operate on the entire batch of
  // requests at the same time for improved performance.
  for (uint32_t r = 0; r < request_count; ++r) {

    TRITONBACKEND_Request* request = requests[r];

    const char* request_id = "";
    GUARDED_RESPOND_IF_ERROR(
        responses, r, TRITONBACKEND_RequestId(request, &request_id));

    const char* input_name;
    GUARDED_RESPOND_IF_ERROR(
        responses, r,
        TRITONBACKEND_RequestInputName(request, 0 /* index */, &input_name));

    TRITONBACKEND_Input* input = nullptr;
    GUARDED_RESPOND_IF_ERROR(
        responses, r, TRITONBACKEND_RequestInput(request, input_name, &input));

    // We also validated that the model configuration specifies only a
    // single output, but the request is not required to request any
    // output at all so we only produce an output if requested.
    const char* requested_output_name = nullptr;
    GUARDED_RESPOND_IF_ERROR(
        responses, r,
        TRITONBACKEND_RequestOutputName(
            request, 0 /* index */, &requested_output_name));

    // If an error response was sent while getting the input or
    // requested output name then display an error message and move on
    // to next request.
    if (responses[r] == nullptr) {
      LOG_MESSAGE(
          TRITONSERVER_LOG_ERROR,
          (std::string("request ") + std::to_string(r) +
           ": failed to read input or requested output name, error response "
           "sent")
              .c_str());
      continue;
    }

    TRITONSERVER_DataType input_datatype;
    const int64_t* input_shape;
    uint32_t input_dims_count;
    uint64_t input_byte_size;
    uint32_t input_buffer_count;
    GUARDED_RESPOND_IF_ERROR(
        responses, r,
        TRITONBACKEND_InputProperties(
            input, nullptr /* input_name */, &input_datatype, &input_shape,
            &input_dims_count, &input_byte_size, &input_buffer_count));
    if (responses[r] == nullptr) {
      LOG_MESSAGE(
          TRITONSERVER_LOG_ERROR,
          (std::string("request ") + std::to_string(r) +
           ": failed to read input properties, error response sent")
              .c_str());
      continue;
    }

    LOG_MESSAGE(
        TRITONSERVER_LOG_INFO,
        (std::string("\tinput ") + input_name +
         ": datatype = " + TRITONSERVER_DataTypeString(input_datatype) +
         ", shape = " + backend::ShapeToString(input_shape, input_dims_count) +
         ", byte_size = " + std::to_string(input_byte_size) +
         ", buffer_count = " + std::to_string(input_buffer_count))
            .c_str());
    LOG_MESSAGE(
        TRITONSERVER_LOG_INFO,
        (std::string("\trequested_output ") + requested_output_name).c_str());

    // For statistics we need to collect the total batch size of all
    // the requests. If the model doesn't support batching then each
    // request is necessarily batch-size 1. If the model does support
    // batching then the first dimension of the shape is the batch
    // size.
    if (supports_batching && (input_dims_count > 0)) {
      total_batch_size += input_shape[0];
    } else {
      total_batch_size++;
    }

    // This backend simply copies the input tensor to the output
    // tensor. The input tensor contents are available in one or
    // more contiguous buffers. To do the copy we:
    //
    //   1. Create an output tensor in the response.
    //
    //   2. Allocate appropriately sized buffer in the output
    //      tensor.
    //
    //   3. Iterate over the input tensor buffers and copy the
    //      contents into the output buffer.
    TRITONBACKEND_Response* response = responses[r];

    // Step 1 
    TRITONBACKEND_Output* output;
    GUARDED_RESPOND_IF_ERROR(
        responses, r,
        TRITONBACKEND_ResponseOutput(
            response, &output, requested_output_name, input_datatype,
            input_shape, input_dims_count - 1));
    if (responses[r] == nullptr) {
      LOG_MESSAGE(
          TRITONSERVER_LOG_ERROR,
          (std::string("request ") + std::to_string(r) +
           ": failed to create response output, error response sent")
              .c_str());
      continue;
    }

    // Step 2. Get the output buffer. We request a buffer in CPU
    // memory but we have to handle any returned type. If we get
    // back a buffer in GPU memory we just fail the request.
    void* output_buffer;
    TRITONSERVER_MemoryType output_memory_type = TRITONSERVER_MEMORY_CPU;
    int64_t output_memory_type_id = 0;
    GUARDED_RESPOND_IF_ERROR(
        responses, r,
        TRITONBACKEND_OutputBuffer(
            output,
            &output_buffer,
            input_byte_size / input_shape[input_dims_count - 1],
            &output_memory_type,
            &output_memory_type_id));
    if ((responses[r] == nullptr) ||
        (output_memory_type == TRITONSERVER_MEMORY_GPU)) {
      GUARDED_RESPOND_IF_ERROR(
          responses, r,
          TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_UNSUPPORTED,
              "failed to create output buffer in CPU memory"));
      LOG_MESSAGE(
          TRITONSERVER_LOG_ERROR,
          (std::string("request ") + std::to_string(r) +
           ": failed to create output buffer in CPU memory, error response "
           "sent")
              .c_str());
      continue;
    }

    // Step 3. Copy input -> output. We can only handle if the input
    // buffers are on CPU so fail otherwise.
    size_t output_buffer_offset = 0;
    for (uint32_t b = 0; b < input_buffer_count; ++b) {
      const void* input_buffer = nullptr;
      uint64_t buffer_byte_size = 0;
      TRITONSERVER_MemoryType input_memory_type = TRITONSERVER_MEMORY_CPU;
      int64_t input_memory_type_id = 0;
      GUARDED_RESPOND_IF_ERROR(
          responses, r,
          TRITONBACKEND_InputBuffer(
              input, b, &input_buffer, &buffer_byte_size, &input_memory_type,
              &input_memory_type_id));
      if ((responses[r] == nullptr) ||
          (input_memory_type == TRITONSERVER_MEMORY_GPU)) {
        GUARDED_RESPOND_IF_ERROR(
            responses, r,
            TRITONSERVER_ErrorNew(
                TRITONSERVER_ERROR_UNSUPPORTED,
                "failed to get input buffer in CPU memory"));
      }

      fil_predict(
        instance,
        reinterpret_cast<const float*>(input_buffer),
        reinterpret_cast<float*>(output_buffer) + output_buffer_offset,
        static_cast<size_t>(input_shape[0])
      );
      output_buffer_offset += buffer_byte_size;
    }

    if (responses[r] == nullptr) {
      LOG_MESSAGE(
          TRITONSERVER_LOG_ERROR,
          (std::string("request ") + std::to_string(r) +
           ": failed to get input buffer in CPU memory, error response "
           "sent")
              .c_str());
      continue;
    }

    // If we get to this point then there hasn't been any error and
    // the response is complete and we can send it. This is the last
    // (and only) response that we are sending for the request so we
    // must mark it FINAL. If there is an error when sending all we
    // can do is log it.
    LOG_IF_ERROR(
        TRITONBACKEND_ResponseSend(
            responses[r], TRITONSERVER_RESPONSE_COMPLETE_FINAL,
            nullptr /* success */),
        "failed sending response");
  }

  // Done with requests...
  // TODO: Release each request as follows
    /* LOG_IF_ERROR(
        TRITONBACKEND_RequestRelease(request, TRITONSERVER_REQUEST_RELEASE_ALL),
        "failed releasing request"); */

  return nullptr;  // success
}

}  // extern "C"

}}}  // namespace triton::backend::fil
