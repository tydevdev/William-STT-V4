#pragma once

// @generated by torchgen/gen.py from NativeFunction.h

#include <c10/core/Scalar.h>
#include <c10/core/Storage.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/Deprecated.h>
#include <c10/util/Optional.h>
#include <c10/core/QScheme.h>
#include <ATen/core/Reduction.h>
#include <ATen/core/Tensor.h>
#include <tuple>
#include <vector>
#include <ATen/ops/_convert_indices_from_csr_to_coo_meta.h>

namespace at {
namespace native {
struct TORCH_API structured__convert_indices_from_csr_to_coo_structured_cpu : public at::meta::structured__convert_indices_from_csr_to_coo {
void impl(const at::Tensor & crow_indices, const at::Tensor & col_indices, bool out_int32, bool transpose, const at::Tensor & out);
};
struct TORCH_API structured__convert_indices_from_csr_to_coo_structured_cuda : public at::meta::structured__convert_indices_from_csr_to_coo {
void impl(const at::Tensor & crow_indices, const at::Tensor & col_indices, bool out_int32, bool transpose, const at::Tensor & out);
};
} // namespace native
} // namespace at
