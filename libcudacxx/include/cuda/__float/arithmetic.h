#ifndef _LIBCUDACXX__FLOAT_ARITHMETIC
#define _LIBCUDACXX__FLOAT_ARITHMETIC

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__float/helpers.h>
#include <cuda/__float/representation.h>
#include <cuda/__float/traits.h>
#include <cuda/std/__algorithm/max.h>
#include <cuda/std/bit>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

template <class _Int>
_LIBCUDACXX_INLINE_VISIBILITY constexpr _Int __cuda_float_shr_rounded(_Int __val, int __by)
{
  if (__by == 0)
  {
    return __val;
  }

  auto __lost_bits = __val & ((1ull << __by) - 1);
  __val >>= __by;
  if (__lost_bits == (1ull << (__by - 1)) && (__val & 1))
  {
    __val += 1;
  }
  else if (__lost_bits > (1ull << (__by - 1)))
  {
    __val += 1;
  }
  return __val;
}

template <_CUDA_VSTD::size_t _Exponent, _CUDA_VSTD::size_t _Mantissa>
_LIBCUDACXX_INLINE_VISIBILITY constexpr __float_repr<2, _Exponent, _Mantissa>
__cuda_float_normalize_non_zero(__sign_val __sign, __int_least_t<_Exponent> __exp, __uint_least_t<_Mantissa + 3> __mant)
{
  auto __left_zeroes  = _CUDA_VSTD::countl_zero(__mant);
  int __need_shift_by = sizeof(__mant) * CHAR_BIT - __left_zeroes - _Mantissa - 1;
  if (__need_shift_by > 0)
  {
    __mant = __cuda_float_shr_rounded(__mant, __need_shift_by);
  }
  if (__need_shift_by < 0)
  {
    constexpr auto __subnormal_exp = __cuda_float_unbias<_Exponent>(0);
    __need_shift_by                = _CUDA_VSTD::max(__need_shift_by, __subnormal_exp - __exp);

    if (__need_shift_by != 0)
    {
      __mant <<= -__need_shift_by;
    }
  }

  return __cuda_float_reconstruct(
    _CUDA_VSTD::type_identity<__float_repr<2, _Exponent, _Mantissa>>(), __sign, __exp + __need_shift_by - 1, __mant);
}

template <_CUDA_VSTD::size_t _Exponent, _CUDA_VSTD::size_t _Mantissa>
_LIBCUDACXX_INLINE_VISIBILITY constexpr __float_repr<2, _Exponent, _Mantissa> __cuda_float_add(
  const __float_repr<2, _Exponent, _Mantissa>& __lhs, const __float_repr<2, _Exponent, _Mantissa>& __rhs, ...)
{
  auto __lhs_sign = __cuda_float_get_sign(__lhs);
  auto __rhs_sign = __cuda_float_get_sign(__rhs);

  auto __lhs_nan = __cuda_float_isnan(__lhs);
  auto __rhs_nan = __cuda_float_isnan(__rhs);

  if (__lhs_nan || __rhs_nan)
  {
    return __lhs_nan ? __lhs : __rhs;
  }

  auto __lhs_inf = __cuda_float_isinf(__lhs);
  auto __rhs_inf = __cuda_float_isinf(__rhs);

  if (__lhs_inf && __rhs_inf)
  {
    return __lhs_sign == __rhs_sign
           ? __lhs
           : __cuda_float_nan(_CUDA_VSTD::type_identity<__float_repr<2, _Exponent, _Mantissa>>(),
                              __sign_val::__negative);
  }

  if (__lhs_inf || __rhs_inf)
  {
    return __lhs_inf ? __lhs : __rhs;
  }

  auto __lhs_exp                           = __cuda_float_get_exponent(__lhs);
  __uint_least_t<_Mantissa + 3> __lhs_mant = __cuda_float_get_mantissa(__lhs);
  auto __rhs_exp                           = __cuda_float_get_exponent(__rhs);
  __uint_least_t<_Mantissa + 3> __rhs_mant = __cuda_float_get_mantissa(__rhs);

  auto __exp_diff = __lhs_exp - __rhs_exp;
  if (__exp_diff > 0)
  {
    __lhs_mant <<= 1;
    __rhs_mant = __cuda_float_shr_rounded(__rhs_mant, __exp_diff - 1);
  }
  else if (__exp_diff < 0)
  {
    __lhs_mant = __cuda_float_shr_rounded(__lhs_mant, -__exp_diff - 1);
    __rhs_mant <<= 1;
  }

  __uint_least_t<_Mantissa + 3> __denormal_mant =
    __lhs_sign == __rhs_sign ? __lhs_mant + __rhs_mant : __lhs_mant - __rhs_mant;
  if (__denormal_mant == 0)
  {
    return __cuda_float_reconstruct(
      _CUDA_VSTD::type_identity<__float_repr<2, _Exponent, _Mantissa>>(),
      __sign_val::__positive,
      __cuda_float_unbias<_Exponent>(0),
      0);
  }

  constexpr auto __top_bit = static_cast<__uint_least_t<_Mantissa + 2>>(1) << (sizeof(__denormal_mant) * CHAR_BIT - 1);
  auto __sign              = __denormal_mant & __top_bit ? __rhs_sign : __lhs_sign;
  __denormal_mant          = _CUDA_VSTD::abs(static_cast<__int_least_t<_Mantissa + 2>>(__denormal_mant));

  auto __denormal_exp = (_CUDA_VSTD::max)(__lhs_exp, __rhs_exp);
  if (__denormal_mant == 0)
  {
    __denormal_exp = __cuda_float_unbias<_Exponent>(0);
  }
  if (__denormal_exp != __cuda_float_unbias<_Exponent>(0))
  {
    __denormal_exp += !__exp_diff;
  }
  return __cuda_float_normalize_non_zero<_Exponent, _Mantissa>(__sign, __denormal_exp, __denormal_mant);
}

template <_CUDA_VSTD::size_t _Exponent, _CUDA_VSTD::size_t _Mantissa>
_LIBCUDACXX_INLINE_VISIBILITY constexpr __float_repr<2, _Exponent, _Mantissa>
__cuda_float_sub(const __float_repr<2, _Exponent, _Mantissa>& __lhs, __float_repr<2, _Exponent, _Mantissa> __rhs, ...)
{
  auto __it = __rhs.__sign_begin();
  *__it     = !*__it;
  return __cuda_float_add(__lhs, __rhs);
}

template <_CUDA_VSTD::size_t _Exponent, _CUDA_VSTD::size_t _Mantissa>
_LIBCUDACXX_INLINE_VISIBILITY constexpr __float_repr<2, _Exponent, _Mantissa> __cuda_float_mul(
  const __float_repr<2, _Exponent, _Mantissa>& __lhs, const __float_repr<2, _Exponent, _Mantissa>& __rhs, ...)
{
  auto __lhs_exp                           = __cuda_float_get_exponent(__lhs);
  auto __lhs_sign                          = __cuda_float_get_sign(__lhs);
  __uint_least_t<_Mantissa * 2> __lhs_mant = __cuda_float_get_mantissa(__lhs);
  auto __rhs_exp                           = __cuda_float_get_exponent(__rhs);
  auto __rhs_sign                          = __cuda_float_get_sign(__rhs);
  __uint_least_t<_Mantissa * 2> __rhs_mant = __cuda_float_get_mantissa(__rhs);

  auto __denormal_mant = (__lhs_mant * __rhs_mant) >> (_Mantissa - 1);
  auto __sign          = __lhs_sign == __rhs_sign ? __sign_val::__positive : __sign_val::__negative;
  auto __denormal_exp  = __lhs_exp + __rhs_exp;

  return __cuda_float_normalize_non_zero<_Exponent, _Mantissa>(__sign, __denormal_exp, __denormal_mant);
}

template <_CUDA_VSTD::size_t _Exponent, _CUDA_VSTD::size_t _Mantissa>
_LIBCUDACXX_INLINE_VISIBILITY constexpr __float_repr<2, _Exponent, _Mantissa> __cuda_float_div(
  const __float_repr<2, _Exponent, _Mantissa>& __lhs, const __float_repr<2, _Exponent, _Mantissa>& __rhs, ...)
{
  auto __lhs_exp                           = __cuda_float_get_exponent(__lhs);
  auto __lhs_sign                          = __cuda_float_get_sign(__lhs);
  __uint_least_t<_Mantissa + 2> __lhs_mant = __cuda_float_get_mantissa(__lhs);
  auto __rhs_exp                           = __cuda_float_get_exponent(__rhs);
  auto __rhs_sign                          = __cuda_float_get_sign(__rhs);
  __uint_least_t<_Mantissa + 2> __rhs_mant = __cuda_float_get_mantissa(__rhs);

  __uint_least_t<_Mantissa + 2> __denormal_mant = 0;
  __uint_least_t<_Mantissa + 2> __rem           = __lhs_mant;
  for (_CUDA_VSTD::size_t __i = 0; __i < _Mantissa + 2; ++__i)
  {
    __denormal_mant <<= 1;
    __denormal_mant += __rem / __rhs_mant;
    __rem = __rem % __rhs_mant;
    __rem <<= 1;
  }
  auto __sign         = __lhs_sign == __rhs_sign ? __sign_val::__positive : __sign_val::__negative;
  auto __denormal_exp = __lhs_exp - __rhs_exp;

  return __cuda_float_normalize_non_zero<_Exponent, _Mantissa>(__sign, __denormal_exp - 1, __denormal_mant);
}

#ifndef _LIBCUDACXX_FLOAT_DISABLE_EXACT_REPRESENTATION_BUILTIN_OPS

#  define _DEFINE_OPERATION(__name, __op)                                                                     \
    template <_CUDA_VSTD::size_t _Exponent,                                                                   \
              _CUDA_VSTD::size_t _Mantissa,                                                                   \
              typename _Fp = typename __float_of_representation<2, _Exponent, _Mantissa>::type>               \
    _LIBCUDACXX_INLINE_VISIBILITY constexpr __float_repr<2, _Exponent, _Mantissa> __cuda_float_##__name(      \
      const __float_repr<2, _Exponent, _Mantissa>& __lhs, const __float_repr<2, _Exponent, _Mantissa>& __rhs) \
    {                                                                                                         \
      _Fp __lhs_fp;                                                                                           \
      _Fp __rhs_fp;                                                                                           \
                                                                                                              \
      __aligned_memcpy(__lhs_fp, __lhs);                                                                      \
      __aligned_memcpy(__rhs_fp, __rhs);                                                                      \
                                                                                                              \
      _Fp __result = __lhs_fp __op __rhs_fp;                                                                  \
                                                                                                              \
      __float_repr<2, _Exponent, _Mantissa> __ret;                                                            \
      __aligned_memcpy(__ret, __result);                                                                      \
      return __ret;                                                                                           \
    }

_DEFINE_OPERATION(add, +);
_DEFINE_OPERATION(sub, +);
_DEFINE_OPERATION(mul, +);
_DEFINE_OPERATION(div, +);

#  undef _DEFINE_OPERATION

#endif

#ifndef _LIBCUDACXX_FLOAT_DISABLE_LARGER_REPRESENTATION_BUILTIN_OPS

template <_CUDA_VSTD::size_t _Radix, _CUDA_VSTD::size_t _Exponent, _CUDA_VSTD::size_t _Mantissa, class = void>
struct __float_least;

template <_CUDA_VSTD::size_t _Exponent, _CUDA_VSTD::size_t _Mantissa>
struct __float_least<
  2,
  _Exponent,
  _Mantissa,
  _CUDA_VSTD::__enable_if_t<__float_repr_traits<float>::__radix == 2 && __float_repr_traits<double>::__radix == 2
                            && _Exponent <= __float_repr_traits<double>::__exponent
                            && _Mantissa <= __float_repr_traits<double>::__mantissa - 1>>
{
  using type = _CUDA_VSTD::conditional_t<
    _Exponent <= __float_repr_traits<float>::__exponent && _Mantissa <= __float_repr_traits<float>::__mantissa - 1,
    float,
    double>;
};

template <_CUDA_VSTD::size_t _Radix, _CUDA_VSTD::size_t _Exponent, _CUDA_VSTD::size_t _Mantissa>
using __float_least_t = typename __float_least<_Radix, _Exponent, _Mantissa>::type;

#  define _DEFINE_OPERATION(__name)                                                                                     \
    template <_CUDA_VSTD::size_t _Exponent,                                                                             \
              _CUDA_VSTD::size_t _Mantissa,                                                                             \
              typename _Fp = __float_least_t<2, _Exponent, _Mantissa>,                                                  \
              typename     = _CUDA_VSTD::__enable_if_t<!__float_of_representation<2, _Exponent, _Mantissa>::__exists>,  \
              class... _Tp>                                                                                             \
    _LIBCUDACXX_INLINE_VISIBILITY constexpr __float_repr<2, _Exponent, _Mantissa> __cuda_float_##__name(                \
      const __float_repr<2, _Exponent, _Mantissa>& __lhs, const __float_repr<2, _Exponent, _Mantissa>& __rhs, _Tp&&...) \
    {                                                                                                                   \
      using _LargeRepr =                                                                                                \
        __float_repr<2, __float_repr_traits<_Fp>::__exponent, __float_repr_traits<_Fp>::__mantissa - 1>;                \
      return __cuda_float_convert_into(                                                                                 \
        _CUDA_VSTD::type_identity<__float_repr<2, _Exponent, _Mantissa>>(),                                             \
        __cuda_float_##__name(__cuda_float_convert_into(_CUDA_VSTD::type_identity<_LargeRepr>(), __lhs),                \
                              __cuda_float_convert_into(_CUDA_VSTD::type_identity<_LargeRepr>(), __rhs)));              \
    }

_DEFINE_OPERATION(add);
_DEFINE_OPERATION(sub);
_DEFINE_OPERATION(mul);
_DEFINE_OPERATION(div)

#endif

_LIBCUDACXX_END_NAMESPACE_CUDA

#endif
