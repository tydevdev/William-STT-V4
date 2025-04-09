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

namespace compositeimplicitautograd {

TORCH_API at::Tensor cross_entropy_loss(const at::Tensor & self, const at::Tensor & target, const ::std::optional<at::Tensor> & weight={}, int64_t reduction=at::Reduction::Mean, int64_t ignore_index=-100, double label_smoothing=0.0);
TORCH_API at::Tensor cross_entropy_loss_symint(const at::Tensor & self, const at::Tensor & target, const ::std::optional<at::Tensor> & weight={}, int64_t reduction=at::Reduction::Mean, c10::SymInt ignore_index=-100, double label_smoothing=0.0);

} // namespace compositeimplicitautograd
} // namespace at
