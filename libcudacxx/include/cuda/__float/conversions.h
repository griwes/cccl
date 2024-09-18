#ifndef _LIBCUDACXX__FLOAT_CONVERSIONS
#define _LIBCUDACXX__FLOAT_CONVERSIONS

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
#include <cuda/std/__algorithm/min.h>
#include <cuda/std/cmath>
#include <cuda/std/type_traits>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

// Useful integer conversion helpers.

template <_CUDA_VSTD::size_t _TargetBits, _CUDA_VSTD::size_t _SourceBits, class _Source>
_LIBCUDACXX_INLINE_VISIBILITY constexpr __int_least_t<_TargetBits> __saturating_biased_convert_signed(_Source __val)
{
  static_assert(_CUDA_VSTD::is_signed<_Source>::value);

  constexpr _Source __max = (1ull << ((_CUDA_VSTD::min)(_TargetBits, sizeof(_Source) * CHAR_BIT) - 1)) - 1;
  constexpr _Source __min = -__max - 1;

  if (__val == __cuda_float_unbias<_SourceBits>(0))
  {
    return _TargetBits >= _SourceBits ? __val : __cuda_float_unbias<_TargetBits>(0);
  }

  if (__val == 0)
  {
    return 0;
  }

  if (__val > 0)
  {
    return static_cast<__int_least_t<_TargetBits>>(__val < __max ? __val : __max);
  }

  return static_cast<__int_least_t<_TargetBits>>(__val > __min ? __val : __min);
}

template <_CUDA_VSTD::size_t _TargetBits, _CUDA_VSTD::size_t _SourceBits, class _Source>
_LIBCUDACXX_INLINE_VISIBILITY constexpr __uint_least_t<_TargetBits> __shifting_convert_unsigned(_Source __val)
{
  static_assert(_CUDA_VSTD::is_unsigned<_Source>::value);

  if (_TargetBits == _SourceBits)
  {
    return __val;
  }

  if (_TargetBits > _SourceBits)
  {
    return static_cast<__uint_least_t<_TargetBits>>(__val)
        << (_TargetBits > _SourceBits ? _TargetBits - _SourceBits : 0);
  }

  constexpr auto _Shift = _SourceBits > _TargetBits ? _SourceBits - _TargetBits : 0;

  bool __least_significant_bit = _Shift ? (__val & (1ull << (_Shift - 1))) : 0;
  return static_cast<__uint_least_t<_TargetBits>>(__val >> _Shift) + __least_significant_bit;
}

template <class _Repr>
_LIBCUDACXX_INLINE_VISIBILITY constexpr bool __cuda_float_isinf(const _Repr& __repr)
{
  return _CUDA_VSTD::isinf(__repr);
}

template <class _Repr>
_LIBCUDACXX_INLINE_VISIBILITY constexpr bool __cuda_float_isnan(const _Repr& __repr)
{
  return _CUDA_VSTD::isnan(__repr);
}

// Generic constant generators.

template <class _Target>
_LIBCUDACXX_INLINE_VISIBILITY constexpr _Target
__cuda_float_infinity(_CUDA_VSTD::type_identity<_Target>, __sign_val __sign)
{
  return __sign == __sign_val::__positive
         ? _CUDA_VSTD::numeric_limits<_Target>::infinity()
         : -_CUDA_VSTD::numeric_limits<_Target>::infinity();
}

template <class _Target>
_LIBCUDACXX_INLINE_VISIBILITY constexpr _Target __cuda_float_nan(_CUDA_VSTD::type_identity<_Target>, __sign_val __sign)
{
  return __sign == __sign_val::__positive
         ? _CUDA_VSTD::numeric_limits<_Target>::quiet_NaN()
         : -_CUDA_VSTD::numeric_limits<_Target>::quiet_NaN();
}

// Generic converting functions.

template <class _Target>
_LIBCUDACXX_INLINE_VISIBILITY constexpr _Target __cuda_float_reconstruct(
  _CUDA_VSTD::type_identity<_Target>,
  __sign_val __sign,
  __int_least_t<__float_repr_traits<_Target>::__exponent> __exponent,
  __uint_least_t<__float_repr_traits<_Target>::__mantissa> __mantissa)
{
  if (__exponent == __cuda_float_unbias<__float_repr_traits<_Target>::__exponent>(0))
  {
    if (__mantissa == 0)
    {
      return static_cast<_Target>(0);
    }
    return (__sign == __sign_val::__positive ? 1 : -1)
         * _CUDA_VSTD::scalbn(static_cast<_Target>(__mantissa), __exponent);
  }
  return (__sign == __sign_val::__positive ? 1 : -1)
       * _CUDA_VSTD::scalbn(
           static_cast<_Target>(__mantissa
                                | (static_cast<__uint_least_t<__float_repr_traits<_Target>::__mantissa>>(1)
                                   << (__float_repr_traits<_Target>::__mantissa - 1))),
           __exponent - __float_repr_traits<_Target>::__mantissa + 1);
}

// NOT an ADL CPO, handles saturation and shifting of the constituents to the correct values.
template <class _Target,
          class _Source,
          class _Traits  = __float_repr_traits<_Target>,
          class _STraits = __float_repr_traits<_Source>>
_LIBCUDACXX_INLINE_VISIBILITY constexpr auto __cuda_float_convert_reconstruct(
  __sign_val __sign, __int_least_t<_STraits::__exponent> __exponent, __uint_least_t<_STraits::__mantissa> __mantissa)
{
  static_assert(_Traits::__radix == _STraits::__radix,
                "converting between floats of different radices is not supported yet");

  if (__exponent == __cuda_float_unbias<_STraits::__exponent>(0) && __mantissa == 0)
  {
    return __cuda_float_reconstruct(
      _CUDA_VSTD::type_identity<_Target>(), __sign, __cuda_float_unbias<_Traits::__exponent>(0), 0);
  }

  auto __converted_exponent = __saturating_biased_convert_signed<_Traits::__exponent, _STraits::__exponent>(__exponent);
  if (__converted_exponent != __exponent && __converted_exponent != __cuda_float_unbias<_Traits::__exponent>(0))
  {
    return __cuda_float_infinity(_CUDA_VSTD::type_identity<_Target>(), __sign);
  }

  return __cuda_float_reconstruct(
    _CUDA_VSTD::type_identity<_Target>(),
    __sign,
    __converted_exponent,
    __shifting_convert_unsigned<_Traits::__mantissa, _STraits::__mantissa>(__mantissa));
}

template <class _Target>
_LIBCUDACXX_INLINE_VISIBILITY constexpr auto
__cuda_float_convert_into(_CUDA_VSTD::type_identity<_Target>, _CUDA_VSTD::type_identity_t<_Target> __val)
{
  return __val;
}

template <class _Target,
          class _Source,
          typename _Traits  = __float_repr_traits<_Target>,
          typename _STraits = __float_repr_traits<_Source>>
struct __cuda_float_layout_compatible
    : _CUDA_VSTD::bool_constant<_Traits::__radix == _STraits::__radix && _Traits::__exponent == _STraits::__exponent
                                && _Traits::__mantissa == _STraits::__mantissa && sizeof(_Target) == sizeof(_Source)
                                && alignof(_Target) == alignof(_Source)>
{};

template <class _Target,
          class _Source,
          typename = _CUDA_VSTD::enable_if_t<!_CUDA_VSTD::is_same_v<_Target, _CUDA_VSTD::remove_cvref_t<_Source>>
                                             && __cuda_float_layout_compatible<_Target, _Source>::value>>
_LIBCUDACXX_INLINE_VISIBILITY constexpr auto
__cuda_float_convert_into(_CUDA_VSTD::type_identity<_Target>, const _Source& __src)
{
  _Target __ret;
  __aligned_memcpy(__ret, __src);
  return __ret;
}

template <class _Target, class _Source, class... _Ts>
_LIBCUDACXX_INLINE_VISIBILITY constexpr auto
__cuda_float_convert_into(_CUDA_VSTD::type_identity<_Target>, const _Source& __src, _Ts&&...)
{
  if (__cuda_float_isnan(__src))
  {
    return __cuda_float_nan(_CUDA_VSTD::type_identity<_Target>(), __cuda_float_get_sign(__src));
  }

  if (__cuda_float_isinf(__src))
  {
    return __cuda_float_infinity(_CUDA_VSTD::type_identity<_Target>(), __cuda_float_get_sign(__src));
  }

  return ::cuda::__cuda_float_convert_reconstruct<_Target, _Source>(
    __cuda_float_get_sign(__src), __cuda_float_get_exponent(__src), __cuda_float_get_mantissa(__src));
}

_LIBCUDACXX_END_NAMESPACE_CUDA

#endif
