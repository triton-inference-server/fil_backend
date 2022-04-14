#pragma once

namespace herring {

template<typename forest_t, typename output_t>
void infer(forest_t const& forest, std::optional<std::vector<output_t>> leaf_outputs) {
};

namespace detail {

template<typename forest_t, bool missing_values_in_input, bool categorical_model>
void find_leaf(forest_t const& forest, std::size_t node_index) {
}

}

}
