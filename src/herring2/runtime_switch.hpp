#pragma once

template<bool compile_switch, bool fast_path, template<bool, typename...> typename functor, typename... fixed_params>
struct runtime_switch {
  template<typename... arg_types>
  auto operator() (bool runtime_switch, arg_types... args) {
    if constexpr (compile_switch == fast_path) {
      return functor<fast_path, fixed_params...>{}(args...);
    } else {
      if (runtime_switch == fast_path) {
        return functor<fast_path, fixed_params...>{}(args...);
      } else {
        return functor<!fast_path, fixed_params...>{}(args...);
      }
    } 
  }
};
