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



#include <ATen/ops/fft_ifftshift_ops.h>

namespace at {


// aten::fft_ifftshift(Tensor self, int[1]? dim=None) -> Tensor
inline at::Tensor fft_ifftshift(const at::Tensor & self, at::OptionalIntArrayRef dim=::std::nullopt) {
    return at::_ops::fft_ifftshift::call(self, dim);
}

}
