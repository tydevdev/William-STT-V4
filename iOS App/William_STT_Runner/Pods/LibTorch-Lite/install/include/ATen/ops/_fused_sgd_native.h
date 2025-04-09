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


namespace at {
namespace native {
TORCH_API ::std::tuple<::std::vector<at::Tensor>,::std::vector<at::Tensor>,::std::vector<at::Tensor>> _fused_sgd(at::TensorList self, at::TensorList grads, at::TensorList momentum_buffer_list, double weight_decay, double momentum, double lr, double dampening, bool nesterov, bool maximize, bool is_first_step, const ::std::optional<at::Tensor> & grad_scale={}, const ::std::optional<at::Tensor> & found_inf={});
TORCH_API void _fused_sgd_out(at::TensorList self, at::TensorList grads, at::TensorList momentum_buffer_list, double weight_decay, double momentum, double lr, double dampening, bool nesterov, bool maximize, bool is_first_step, const ::std::optional<at::Tensor> & grad_scale, const ::std::optional<at::Tensor> & found_inf, at::TensorList out);
TORCH_API void _fused_sgd_kernel_cpu_(at::TensorList self, at::TensorList grads, at::TensorList momentum_buffer_list, double weight_decay, double momentum, double lr, double dampening, bool nesterov, bool maximize, bool is_first_step, const ::std::optional<at::Tensor> & grad_scale={}, const ::std::optional<at::Tensor> & found_inf={});
TORCH_API void _fused_sgd_kernel_cuda_(at::TensorList self, at::TensorList grads, at::TensorList momentum_buffer_list, double weight_decay, double momentum, double lr, double dampening, bool nesterov, bool maximize, bool is_first_step, const ::std::optional<at::Tensor> & grad_scale={}, const ::std::optional<at::Tensor> & found_inf={});
TORCH_API ::std::tuple<::std::vector<at::Tensor>,::std::vector<at::Tensor>,::std::vector<at::Tensor>> _fused_sgd(at::TensorList self, at::TensorList grads, at::TensorList momentum_buffer_list, double weight_decay, double momentum, const at::Tensor & lr, double dampening, bool nesterov, bool maximize, bool is_first_step, const ::std::optional<at::Tensor> & grad_scale={}, const ::std::optional<at::Tensor> & found_inf={});
TORCH_API void _fused_sgd_tensor_lr_out(at::TensorList self, at::TensorList grads, at::TensorList momentum_buffer_list, double weight_decay, double momentum, const at::Tensor & lr, double dampening, bool nesterov, bool maximize, bool is_first_step, const ::std::optional<at::Tensor> & grad_scale, const ::std::optional<at::Tensor> & found_inf, at::TensorList out);
TORCH_API void _fused_sgd_kernel_cpu_(at::TensorList self, at::TensorList grads, at::TensorList momentum_buffer_list, double weight_decay, double momentum, const at::Tensor & lr, double dampening, bool nesterov, bool maximize, bool is_first_step, const ::std::optional<at::Tensor> & grad_scale={}, const ::std::optional<at::Tensor> & found_inf={});
TORCH_API void _fused_sgd_kernel_cuda_(at::TensorList self, at::TensorList grads, at::TensorList momentum_buffer_list, double weight_decay, double momentum, const at::Tensor & lr, double dampening, bool nesterov, bool maximize, bool is_first_step, const ::std::optional<at::Tensor> & grad_scale={}, const ::std::optional<at::Tensor> & found_inf={});
} // namespace native
} // namespace at
