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



#include <ATen/ops/linalg_lstsq_ops.h>

namespace at {


// aten::linalg_lstsq(Tensor self, Tensor b, float? rcond=None, *, str? driver=None) -> (Tensor solution, Tensor residuals, Tensor rank, Tensor singular_values)
inline ::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> linalg_lstsq(const at::Tensor & self, const at::Tensor & b, ::std::optional<double> rcond=::std::nullopt, ::std::optional<c10::string_view> driver=::std::nullopt) {
    return at::_ops::linalg_lstsq::call(self, b, rcond, driver);
}

// aten::linalg_lstsq.out(Tensor self, Tensor b, float? rcond=None, *, str? driver=None, Tensor(a!) solution, Tensor(b!) residuals, Tensor(c!) rank, Tensor(d!) singular_values) -> (Tensor(a!) solution, Tensor(b!) residuals, Tensor(c!) rank, Tensor(d!) singular_values)
inline ::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &,at::Tensor &> linalg_lstsq_out(at::Tensor & solution, at::Tensor & residuals, at::Tensor & rank, at::Tensor & singular_values, const at::Tensor & self, const at::Tensor & b, ::std::optional<double> rcond=::std::nullopt, ::std::optional<c10::string_view> driver=::std::nullopt) {
    return at::_ops::linalg_lstsq_out::call(self, b, rcond, driver, solution, residuals, rank, singular_values);
}
// aten::linalg_lstsq.out(Tensor self, Tensor b, float? rcond=None, *, str? driver=None, Tensor(a!) solution, Tensor(b!) residuals, Tensor(c!) rank, Tensor(d!) singular_values) -> (Tensor(a!) solution, Tensor(b!) residuals, Tensor(c!) rank, Tensor(d!) singular_values)
inline ::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &,at::Tensor &> linalg_lstsq_outf(const at::Tensor & self, const at::Tensor & b, ::std::optional<double> rcond, ::std::optional<c10::string_view> driver, at::Tensor & solution, at::Tensor & residuals, at::Tensor & rank, at::Tensor & singular_values) {
    return at::_ops::linalg_lstsq_out::call(self, b, rcond, driver, solution, residuals, rank, singular_values);
}

}
