#ifndef _LIBCUDACXX__FLOAT_EXTRACTION
#define _LIBCUDACXX__FLOAT_EXTRACTION

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__float/helpers.h>
#include <cuda/__float/traits.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_floating_point.h>
#include <cuda/std/cmath>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

// __cuda_float_get_representation: helper for normalizing float-like types into their storage types.
// For types not defined belo, this will generally just be an identity transformation. However, because of our layering
// of the PTX dispatch functions here, we do need a transformation to happen for floating_point defined below, so they
// will define overloads of this normalization hook.

template <class _Fp>
_LIBCUDACXX_INLINE_VISIBILITY constexpr _Fp __cuda_float_get_representation(const _Fp& __self)
{
  return __self;
}

// __cuda_float_get_{sign,exponent,mantissa}: helpers for extracting the bits of the given part of a float-like
// representation object.
// TODO(mdominiak): fuse these into one? Seeing how the default of get_mantissa uses get_exponent?

template <class _Repr, class... _Ts>
_LIBCUDACXX_INLINE_VISIBILITY constexpr __sign_val __cuda_float_get_sign(const _Repr& __repr, const _Ts&...)
{
  return _CUDA_VSTD::signbit(__repr) ? __sign_val::__negative : __sign_val::__positive;
}

template <class _Repr, class... _Ts>
_LIBCUDACXX_INLINE_VISIBILITY constexpr __int_least_t<__float_repr_traits<_Repr>::__exponent>
__cuda_float_get_exponent(const _Repr& __repr, const _Ts&...)
{
  if (!_CUDA_VSTD::isnormal(__repr))
  {
    return __cuda_float_unbias<__float_repr_traits<_Repr>::__exponent>(0);
  }
  return _CUDA_VSTD::ilogb(__repr);
}

template <class _Repr, class... _Ts>
_LIBCUDACXX_INLINE_VISIBILITY constexpr __uint_least_t<__float_repr_traits<_Repr>::__mantissa>
__cuda_float_get_mantissa(const _Repr& __repr, const _Ts&...)
{
  using _Traits = __float_repr_traits<_Repr>;
  auto __exp    = __cuda_float_get_exponent(__repr);
  return static_cast<__uint_least_t<_Traits::__mantissa>>(_CUDA_VSTD::abs(
    _CUDA_VSTD::scalbn(__repr, (_CUDA_VSTD::isnormal(__repr) ? _Traits::__mantissa - 1 - __exp : -__exp))));
}

// Customizations for built-in types with equivalent-length unsigned types.

template <class _Fp, class _Traits = __float_repr_traits<_Fp>>
struct __is_integral_representable
    : _CUDA_VSTD::bool_constant<_Traits::__radix == 2 && _CUDA_VSTD::is_floating_point<_Fp>::value
                                && _CUDA_VSTD::is_integral<__uint_least_t<sizeof(_Fp)>>::value>
{};

template <class _Fp, typename = _CUDA_VSTD::__enable_if_t<__is_integral_representable<_Fp>::value>>
_LIBCUDACXX_INLINE_VISIBILITY constexpr __uint_least_t<sizeof(_Fp) * CHAR_BIT> __cuda_float_raw_repr(const _Fp& __fp)
{
  __uint_least_t<sizeof(_Fp) * CHAR_BIT> __tmp = 0;
  __aligned_memcpy(__tmp, __fp);
  return __tmp;
}

template <class _Fp,
          typename _Traits = __float_repr_traits<_Fp>,
          typename         = _CUDA_VSTD::__enable_if_t<__is_integral_representable<_Fp>::value>>
_LIBCUDACXX_INLINE_VISIBILITY constexpr __int_least_t<_Traits::__exponent> __cuda_float_get_exponent(const _Fp& __fp)
{
  auto __repr = __cuda_float_raw_repr(__fp);
  return __cuda_float_unbias<_Traits::__exponent>(
    (__repr >> (_Traits::__mantissa - 1))
    & ((static_cast<__uint_least_t<_Traits::__exponent>>(1) << _Traits::__exponent) - 1));
}

template <class _Fp,
          typename _Traits = __float_repr_traits<_Fp>,
          typename         = _CUDA_VSTD::__enable_if_t<__is_integral_representable<_Fp>::value>>
_LIBCUDACXX_INLINE_VISIBILITY constexpr __uint_least_t<_Traits::__mantissa> __cuda_float_get_mantissa(const _Fp& __fp)
{
  auto __repr = __cuda_float_raw_repr(__fp);
  return __cuda_float_mantissa_with_implicit<_Traits::__mantissa>(
    __repr & (((static_cast<__uint_least_t<_Traits::__mantissa>>(1) << (_Traits::__mantissa - 1)) - 1)));
}

_LIBCUDACXX_END_NAMESPACE_CUDA

#endif
