#pragma once

// @generated by torchgen/gen.py from Function.h

#include <ATen/Context.h>
#include <ATen/DeviceGuard.h>
#include <ATen/TensorUtils.h>
#include <ATen/TracerMode.h>
#include <ATen/core/Generator.h>
#include <ATen/core/Reduction.h>
#include <ATen/core/Tensor.h>
#include <c10/core/Scalar.h>
#include <c10/core/Storage.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/Deprecated.h>
#include <c10/util/Optional.h>



#include <ATen/ops/_sparse_semi_structured_mm_ops.h>

namespace at {


// aten::_sparse_semi_structured_mm(Tensor mat1, Tensor mat1_meta, Tensor mat2, *, ScalarType? out_dtype=None) -> Tensor
inline at::Tensor _sparse_semi_structured_mm(const at::Tensor & mat1, const at::Tensor & mat1_meta, const at::Tensor & mat2, ::std::optional<at::ScalarType> out_dtype=::std::nullopt) {
    return at::_ops::_sparse_semi_structured_mm::call(mat1, mat1_meta, mat2, out_dtype);
}

}
