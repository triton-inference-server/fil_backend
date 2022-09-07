#include <exception>
#include <string>
struct not_implemented_exception : std::exception {
  not_implemented_exception () : msg_{"Feature not yet implemented"}
  {
  }
  not_implemented_exception (std::string msg) : msg_{msg}
  {
  }
  not_implemented_exception (char const* msg) : msg_{msg}
  {
  }
  virtual char const* what() const noexcept { return msg_.c_str(); }
 private:
  std::string msg_;
};

