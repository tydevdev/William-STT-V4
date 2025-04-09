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



#include <ATen/ops/atanh_ops.h>

namespace at {


// aten::atanh(Tensor self) -> Tensor
inline at::Tensor atanh(const at::Tensor & self) {
    return at::_ops::atanh::call(self);
}

// aten::atanh_(Tensor(a!) self) -> Tensor(a!)
inline at::Tensor & atanh_(at::Tensor & self) {
    return at::_ops::atanh_::call(self);
}

// aten::atanh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & atanh_out(at::Tensor & out, const at::Tensor & self) {
    return at::_ops::atanh_out::call(self, out);
}
// aten::atanh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & atanh_outf(const at::Tensor & self, at::Tensor & out) {
    return at::_ops::atanh_out::call(self, out);
}

}
