#pragma once
// @generated by torchgen/gen.py from DispatchKeyFunction.h

// NB: The implementing C++ file is RegisterDispatchKey.cpp

// The only #includes we need are for custom classes that have defaults in the C++ API
#include <c10/core/MemoryFormat.h>
#include <c10/core/Scalar.h>
#include <ATen/core/Reduction.h>

// Forward declarations of any types needed in the operator signatures.
// We can't directly include these classes because it will cause circular include dependencies.
// This file is included by TensorBody.h, which defines the Tensor class.
#include <ATen/core/ATen_fwd.h>

namespace at {

namespace compositeexplicitautograd {

TORCH_API at::Tensor & slow_conv_dilated2d_out(at::Tensor & out, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, const ::std::optional<at::Tensor> & bias={}, at::IntArrayRef stride=1, at::IntArrayRef padding=0, at::IntArrayRef dilation=1);
TORCH_API at::Tensor & slow_conv_dilated2d_outf(const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, const ::std::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, at::Tensor & out);
TORCH_API at::Tensor & slow_conv_dilated2d_symint_out(at::Tensor & out, const at::Tensor & self, const at::Tensor & weight, c10::SymIntArrayRef kernel_size, const ::std::optional<at::Tensor> & bias={}, c10::SymIntArrayRef stride=c10::SymInt(1), c10::SymIntArrayRef padding=c10::SymInt(0), c10::SymIntArrayRef dilation=c10::SymInt(1));
TORCH_API at::Tensor & slow_conv_dilated2d_symint_outf(const at::Tensor & self, const at::Tensor & weight, c10::SymIntArrayRef kernel_size, const ::std::optional<at::Tensor> & bias, c10::SymIntArrayRef stride, c10::SymIntArrayRef padding, c10::SymIntArrayRef dilation, at::Tensor & out);

} // namespace compositeexplicitautograd
} // namespace at
