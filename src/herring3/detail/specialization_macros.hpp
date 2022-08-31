#pragma once
#include <variant>
#include <herring3/constants.hpp>
#include <herring3/detail/forest.hpp>
#include <herring3/detail/specialization_types.hpp>

#define HERRING_SPEC(variant_index) typename std::variant_alternative_t<variant_index, herring::detail::specialization_variant>

#define HERRING_FOREST(variant_index) forest<preferred_tree_layout, HERRING_SPEC(variant_index)::threshold_type, HERRING_SPEC(variant_index)::index_type, HERRING_SPEC(variant_index)::metadata_type, HERRING_SPEC(variant_index)::offset_type>

#define HERRING_SCALAR_LOCAL_ARGS(dev, variant_index)(\
  HERRING_FOREST(variant_index) const&,\
  postprocessor<HERRING_SPEC(variant_index)::threshold_type> const&,\
  HERRING_SPEC(variant_index)::threshold_type*,\
  HERRING_SPEC(variant_index)::threshold_type*,\
  std::size_t,\
  std::size_t,\
  std::size_t,\
  std::nullptr_t,\
  std::nullptr_t,\
  std::optional<std::size_t>,\
  kayak::device_id<dev>,\
  kayak::cuda_stream stream\
)

#define HERRING_VECTOR_LOCAL_ARGS(dev, variant_index)(\
  HERRING_FOREST(variant_index) const&,\
  postprocessor<HERRING_SPEC(variant_index)::threshold_type> const&,\
  HERRING_SPEC(variant_index)::threshold_type*,\
  HERRING_SPEC(variant_index)::threshold_type*,\
  std::size_t,\
  std::size_t,\
  std::size_t,\
  HERRING_SPEC(variant_index)::threshold_type*,\
  std::nullptr_t,\
  std::optional<std::size_t>,\
  kayak::device_id<dev>,\
  kayak::cuda_stream stream\
)

#define HERRING_SCALAR_NONLOCAL_ARGS(dev, variant_index)(\
  HERRING_FOREST(variant_index) const&,\
  postprocessor<HERRING_SPEC(variant_index)::threshold_type> const&,\
  HERRING_SPEC(variant_index)::threshold_type*,\
  HERRING_SPEC(variant_index)::threshold_type*,\
  std::size_t,\
  std::size_t,\
  std::size_t,\
  std::nullptr_t,\
  HERRING_SPEC(variant_index)::index_type*,\
  std::optional<std::size_t>,\
  kayak::device_id<dev>,\
  kayak::cuda_stream stream\
)

#define HERRING_VECTOR_NONLOCAL_ARGS(dev, variant_index)(\
  HERRING_FOREST(variant_index) const&,\
  postprocessor<HERRING_SPEC(variant_index)::threshold_type> const&,\
  HERRING_SPEC(variant_index)::threshold_type*,\
  HERRING_SPEC(variant_index)::threshold_type*,\
  std::size_t,\
  std::size_t,\
  std::size_t,\
  HERRING_SPEC(variant_index)::threshold_type*,\
  HERRING_SPEC(variant_index)::index_type*,\
  std::optional<std::size_t>,\
  kayak::device_id<dev>,\
  kayak::cuda_stream stream\
)

#define HERRING_INFER_TEMPLATE(template_type, dev, variant_index, categorical) template_type void infer<\
  dev, categorical, HERRING_FOREST(variant_index)>

#define HERRING_INFER_DEV_SCALAR_LEAF_NO_CAT(template_type, dev, variant_index) HERRING_INFER_TEMPLATE(template_type, dev, variant_index, false)HERRING_SCALAR_LOCAL_ARGS(dev, variant_index);

#define HERRING_INFER_DEV_SCALAR_LEAF_LOCAL_CAT(template_type, dev, variant_index) HERRING_INFER_TEMPLATE(template_type, dev, variant_index, true)HERRING_SCALAR_LOCAL_ARGS(dev, variant_index);

#define HERRING_INFER_DEV_SCALAR_LEAF_NONLOCAL_CAT(template_type, dev, variant_index) HERRING_INFER_TEMPLATE(template_type, dev, variant_index, true)HERRING_SCALAR_NONLOCAL_ARGS(dev, variant_index);

#define HERRING_INFER_DEV_VECTOR_LEAF_NO_CAT(template_type, dev, variant_index) HERRING_INFER_TEMPLATE(template_type, dev, variant_index, false)HERRING_VECTOR_LOCAL_ARGS(dev, variant_index);

#define HERRING_INFER_DEV_VECTOR_LEAF_LOCAL_CAT(template_type, dev, variant_index) HERRING_INFER_TEMPLATE(template_type, dev, variant_index, true)HERRING_VECTOR_LOCAL_ARGS(dev, variant_index);

#define HERRING_INFER_DEV_VECTOR_LEAF_NONLOCAL_CAT(template_type, dev, variant_index) HERRING_INFER_TEMPLATE(template_type, dev, variant_index, true)HERRING_VECTOR_NONLOCAL_ARGS(dev, variant_index);

#define HERRING_INFER_ALL(template_type, dev, variant_index) HERRING_INFER_DEV_SCALAR_LEAF_NO_CAT(template_type, dev, variant_index)\
  HERRING_INFER_DEV_SCALAR_LEAF_LOCAL_CAT(template_type, dev, variant_index)\
  HERRING_INFER_DEV_SCALAR_LEAF_NONLOCAL_CAT(template_type, dev, variant_index)\
  HERRING_INFER_DEV_VECTOR_LEAF_NO_CAT(template_type, dev, variant_index)\
  HERRING_INFER_DEV_VECTOR_LEAF_LOCAL_CAT(template_type, dev, variant_index)\
  HERRING_INFER_DEV_VECTOR_LEAF_NONLOCAL_CAT(template_type, dev, variant_index)
