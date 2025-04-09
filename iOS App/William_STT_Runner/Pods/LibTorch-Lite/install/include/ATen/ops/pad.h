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



#include <ATen/ops/pad_ops.h>

namespace at {


// aten::pad(Tensor self, SymInt[] pad, str mode="constant", float? value=None) -> Tensor
inline at::Tensor pad(const at::Tensor & self, at::IntArrayRef pad, c10::string_view mode="constant", ::std::optional<double> value=::std::nullopt) {
    return at::_ops::pad::call(self, c10::fromIntArrayRefSlow(pad), mode, value);
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same<T, int64_t>::value>>
  at::Tensor pad(const at::Tensor & self, at::IntArrayRef pad, c10::string_view mode="constant", ::std::optional<double> value=::std::nullopt) {
    return at::_ops::pad::call(self, c10::fromIntArrayRefSlow(pad), mode, value);
  }
}

// aten::pad(Tensor self, SymInt[] pad, str mode="constant", float? value=None) -> Tensor
inline at::Tensor pad_symint(const at::Tensor & self, c10::SymIntArrayRef pad, c10::string_view mode="constant", ::std::optional<double> value=::std::nullopt) {
    return at::_ops::pad::call(self, pad, mode, value);
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same<T, c10::SymInt>::value>>
  at::Tensor pad(const at::Tensor & self, c10::SymIntArrayRef pad, c10::string_view mode="constant", ::std::optional<double> value=::std::nullopt) {
    return at::_ops::pad::call(self, pad, mode, value);
  }
}

}
