#pragma once

// #ifdef TRITON_ENABLE_GPU
#include <cuml/fil/fil.h>
#include <treelite/tree.h>

#include <cstddef>
#include <memory>
#include <raft/handle.hpp>

#include <matrix.hpp>

struct ForestModel {
  using device_id_t = int;
  ForestModel(std::unique_ptr<treelite::Model>& tl_model, bool sparse=false)
      : device_id_{}, raft_handle_{},
        fil_forest_{[this, &tl_model, sparse]() {
          auto result = ML::fil::forest_t{};
          auto config = ML::fil::treelite_params_t{
            ML::fil::algo_t::ALGO_AUTO,
            true,
            0.5,
            sparse ? ML::fil::storage_type_t::SPARSE : ML::fil::storage_type_t::DENSE,
            0,
            1,
            0,
            nullptr
          };
          ML::fil::from_treelite(
              raft_handle_, &result, static_cast<void*>(tl_model.get()), &config);
          return result;
        }()}
  {
  }

  ForestModel(ForestModel const& other) = default;
  ForestModel& operator=(ForestModel const& other) = default;
  ForestModel(ForestModel&& other) = default;
  ForestModel& operator=(ForestModel&& other) = default;

  ~ForestModel() noexcept { ML::fil::free(raft_handle_, fil_forest_); };

  /* void predict(float* output, matrix& input, bool predict_proba) const {
    ML::fil::predict(
        raft_handle_, fil_forest_, output, input.data, input.rows,
        predict_proba);
    raft_handle_.sync_stream();
  } */

  auto get_stream() {
    return raft_handle_.get_stream();
  }

 private:
  raft::handle_t raft_handle_;
  ML::fil::forest_t fil_forest_;
  device_id_t device_id_;
};
// #endif
