/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#include <type_traits>

namespace herring {
/** A value representing possible stored data for a tree node
 *
 * A non-leaf node will simply store its threshold value in the value field. A
 * leaf node that has an output of the same type as the threshold values will
 * do the same. A leaf node with a different output type will store an index
 * which is used to look up the output in external storage _unless_
 * output_index_t is itself the output type, in which case the output can be
 * stored directly in the index field. A non-leaf node with a categorical test
 * will store the bitset representing its categories in the categories field
 * unless the number of categories results in a bitset type large than either
 * value_t or output_index_t. In such cases, the categories field is used as an
 * index to look up the categorical bitset in external storage. */
template<typename value_t, typename output_index_t, typename bitset_t>
union node_value {
  value_t value;
  output_index_t index;
  std::conditional_t<
    sizeof(bitset_t) <= sizeof(value_t) || sizeof(bitset_t) <= sizeof(output_index_t),
    bitset_t,
    output_index_t
  > categories;
};
}
