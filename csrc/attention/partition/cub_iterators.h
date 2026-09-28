#pragma once

// CUB iterators on older toolkits; Thrust aliases on CCCL 3+.
#if __has_include(<cub/iterator/counting_input_iterator.cuh>) && \
    __has_include(<cub/iterator/transform_input_iterator.cuh>)
#include <cub/iterator/counting_input_iterator.cuh>
#include <cub/iterator/transform_input_iterator.cuh>
#else
#include <cub/util_namespace.cuh>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <cstddef>

// BOS iterator aliases with explicit value and difference types.
CUB_NAMESPACE_BEGIN
template <typename ValueType, typename OffsetT = std::ptrdiff_t>
using CountingInputIterator = thrust::counting_iterator<
    ValueType, thrust::use_default, thrust::use_default, OffsetT>;

template <typename ValueType, typename ConversionOp, typename InputIteratorT>
using TransformInputIterator = thrust::transform_iterator<
    ConversionOp, InputIteratorT, ValueType, ValueType>;
CUB_NAMESPACE_END
#endif
