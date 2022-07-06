#pragma once
#include <exception>
#include <string>

namespace herring {

struct unusable_model_exception : std::exception {
  unusable_model_exception () : msg_{"Model is not compatible with Herring"}
  {
  }
  unusable_model_exception (std::string msg) : msg_{msg}
  {
  }
  unusable_model_exception (char const* msg) : msg_{msg}
  {
  }
  virtual char const* what() const noexcept { return msg_.c_str(); }
 private:
  std::string msg_;
};

}
