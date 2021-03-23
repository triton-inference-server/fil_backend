#pragma once
#include <exception>
#include <triton/core/tritonserver.h>

namespace triton { namespace backend { namespace fil {

class TritonException : public std::exception {
  private:
    TRITONSERVER_Error * error_;
  public:
    TritonException() : error_(TRITONSERVER_ErrorNew(
        TRITONSERVER_errorcode_enum::TRITONSERVER_ERROR_UNKNOWN,
        "encountered unknown error")) {}

    TritonException(TRITONSERVER_Error_Code code, const std::string& msg) :
      error_(TRITONSERVER_ErrorNew(code, msg.c_str())) {}

    TritonException(TRITONSERVER_Error_Code code, const char * msg) :
      error_{TRITONSERVER_ErrorNew(code, msg)} {}
    TritonException(TRITONSERVER_Error * prev_error) : error_(prev_error) {}

    const char * what() const noexcept {
      return TRITONSERVER_ErrorMessage(error_);
    }

    TRITONSERVER_Error * error() {
      return error_;
    }
};

}}}
