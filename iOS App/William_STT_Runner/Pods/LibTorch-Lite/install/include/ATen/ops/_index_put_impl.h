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



#include <ATen/ops/_index_put_impl_ops.h>

namespace at {


// aten::_index_put_impl_(Tensor(a!) self, Tensor?[] indices, Tensor values, bool accumulate=False, bool unsafe=False) -> Tensor(a!)
inline at::Tensor & _index_put_impl_(at::Tensor & self, const c10::List<::std::optional<at::Tensor>> & indices, const at::Tensor & values, bool accumulate=false, bool unsafe=false) {
    return at::_ops::_index_put_impl_::call(self, indices, values, accumulate, unsafe);
}

// aten::_index_put_impl.out(Tensor self, Tensor?[] indices, Tensor values, bool accumulate=False, bool unsafe=False, *, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & _index_put_impl_out(at::Tensor & out, const at::Tensor & self, const c10::List<::std::optional<at::Tensor>> & indices, const at::Tensor & values, bool accumulate=false, bool unsafe=false) {
    return at::_ops::_index_put_impl_out::call(self, indices, values, accumulate, unsafe, out);
}
// aten::_index_put_impl.out(Tensor self, Tensor?[] indices, Tensor values, bool accumulate=False, bool unsafe=False, *, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & _index_put_impl_outf(const at::Tensor & self, const c10::List<::std::optional<at::Tensor>> & indices, const at::Tensor & values, bool accumulate, bool unsafe, at::Tensor & out) {
    return at::_ops::_index_put_impl_out::call(self, indices, values, accumulate, unsafe, out);
}

// aten::_index_put_impl(Tensor self, Tensor?[] indices, Tensor values, bool accumulate=False, bool unsafe=False) -> Tensor
inline at::Tensor _index_put_impl(const at::Tensor & self, const c10::List<::std::optional<at::Tensor>> & indices, const at::Tensor & values, bool accumulate=false, bool unsafe=false) {
    return at::_ops::_index_put_impl::call(self, indices, values, accumulate, unsafe);
}

}
