#ifndef _LIBCUDACXX__FLOAT_REPRESENTATION
#define _LIBCUDACXX__FLOAT_REPRESENTATION

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__float/extraction.h>
#include <cuda/__float/helpers.h>
#include <cuda/__float/traits.h>
#include <cuda/std/bitset>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

// __float_repr: storage for custom-exponent-and-mantissa float types

template <class _Tp>
struct __storage_iterator_defs
{
  using iterator                = _CUDA_VSTD::__bit_iterator<__storage_iterator_defs, false>;
  using const_iterator          = _CUDA_VSTD::__bit_iterator<__storage_iterator_defs, true>;
  using difference_type         = _CUDA_VSTD::ptrdiff_t;
  using __storage_type          = _Tp;
  using __storage_pointer       = __storage_type*;
  using __const_storage_pointer = const __storage_type*;
  using __self                  = __storage_iterator_defs;

  static const constexpr auto __bits_per_word = CHAR_BIT * sizeof(_Tp);
};

template <_CUDA_VSTD::size_t _Exponent, _CUDA_VSTD::size_t _Mantissa>
struct __float_layout_math
{
  static const constexpr _CUDA_VSTD::size_t __total_bits  = 1 + _Exponent + _Mantissa;
  static const constexpr _CUDA_VSTD::size_t __total_bytes = __total_bits / CHAR_BIT + (__total_bits % CHAR_BIT ? 1 : 0);
  static const constexpr _CUDA_VSTD::size_t __alignment   = _CUDA_VSTD::bit_ceil(__total_bytes);

  using __bitset = _CUDA_VSTD::__bitset<(__total_bytes - 1) / sizeof(_CUDA_VSTD::size_t) + 1, __total_bits>;
};

template <_CUDA_VSTD::size_t _Exponent, _CUDA_VSTD::size_t _Mantissa>
using __float_bitset_t = typename __float_layout_math<_Exponent, _Mantissa>::__bitset;

template <_CUDA_VSTD::size_t _Radix,
          _CUDA_VSTD::size_t _Exponent,
          _CUDA_VSTD::size_t _Mantissa,
          _CUDA_VSTD::float_round_style _RoundingMode = _CUDA_VSTD::round_to_nearest>
struct __float_repr;

template <_CUDA_VSTD::size_t _Exponent, _CUDA_VSTD::size_t _Mantissa>
struct alignas(__float_layout_math<_Exponent, _Mantissa>::__alignment) __float_repr<2, _Exponent, _Mantissa>
    : __float_bitset_t<_Exponent, _Mantissa>
{
  static_assert(_Exponent <= 128, "exponents over 128 bits are not currently supported");
  static_assert(_Mantissa <= 126, "mantissas over 126 bits are not currently supported");

  using __base         = __float_bitset_t<_Exponent, _Mantissa>;
  using iterator       = typename __base::iterator;
  using const_iterator = typename __base::const_iterator;

  using __exponent_access_t =
    _CUDA_VSTD::conditional_t<_CUDA_VSTD::is_same<typename __base::__storage_type, __uint_least_t<_Exponent>>::value,
                              typename __base::__storage_type,
                              unsigned char>;
  using __mantissa_access_t =
    _CUDA_VSTD::conditional_t<_CUDA_VSTD::is_same<typename __base::__storage_type, __uint_least_t<_Mantissa>>::value,
                              typename __base::__storage_type,
                              unsigned char>;

  using __exponent_iterator       = typename __storage_iterator_defs<__exponent_access_t>::iterator;
  using __const_exponent_iterator = typename __storage_iterator_defs<__exponent_access_t>::const_iterator;
  using __mantissa_iterator       = typename __storage_iterator_defs<__mantissa_access_t>::iterator;
  using __const_mantissa_iterator = typename __storage_iterator_defs<__mantissa_access_t>::const_iterator;

  template <class _Tp>
  _LIBCUDACXX_INLINE_VISIBILITY static __exponent_iterator __make_exp_iter(_Tp* __ptr, _CUDA_VSTD::size_t __pos)
  {
    return __exponent_iterator(reinterpret_cast<__exponent_access_t*>(__ptr), 0) + __pos;
  }

  template <class _Tp>
  _LIBCUDACXX_INLINE_VISIBILITY static __const_exponent_iterator
  __make_exp_iter(const _Tp* __ptr, _CUDA_VSTD::size_t __pos)
  {
    return __const_exponent_iterator(reinterpret_cast<const __exponent_access_t*>(__ptr), 0) + __pos;
  }

  template <class _Tp>
  _LIBCUDACXX_INLINE_VISIBILITY static __mantissa_iterator __make_mant_iter(_Tp* __ptr, _CUDA_VSTD::size_t __pos)
  {
    return __mantissa_iterator(reinterpret_cast<__mantissa_access_t*>(__ptr), 0) + __pos;
  }

  template <class _Tp>
  _LIBCUDACXX_INLINE_VISIBILITY static __const_mantissa_iterator
  __make_mant_iter(const _Tp* __ptr, _CUDA_VSTD::size_t __pos)
  {
    return __const_mantissa_iterator(reinterpret_cast<const __mantissa_access_t*>(__ptr), 0) + __pos;
  }

  __float_repr() = default;

  _LIBCUDACXX_INLINE_VISIBILITY constexpr __float_repr(
    __sign_val __sign, __int_least_t<_Exponent> __exponent, __uint_least_t<_Mantissa + 1> __mantissa)
  {
    *__sign_begin() = __sign == __sign_val::__positive ? 0 : 1;

    __uint_least_t<_Exponent> __exponent_biased =
      static_cast<__uint_least_t<_Exponent>>(__exponent)
      + ((static_cast<__uint_least_t<_Exponent>>(1) << (_Exponent - 1))) - 1;
    _CUDA_VSTD::copy_n(__make_exp_iter(&__exponent_biased, 0), _Exponent, __exponent_begin());
    _CUDA_VSTD::copy_n(__make_mant_iter(&__mantissa, 0), _Mantissa, __mantissa_begin());
  }

  _LIBCUDACXX_INLINE_VISIBILITY auto __mantissa_begin()
  {
    return __make_mant_iter(&this->__first_, 0);
  }

  _LIBCUDACXX_INLINE_VISIBILITY auto __mantissa_begin() const
  {
    return __make_mant_iter(&this->__first_, 0);
  }

  _LIBCUDACXX_INLINE_VISIBILITY auto __exponent_begin()
  {
    return __make_exp_iter(&this->__first_, _Mantissa);
  }

  _LIBCUDACXX_INLINE_VISIBILITY auto __exponent_begin() const
  {
    return __make_exp_iter(&this->__first_, _Mantissa);
  }

  _LIBCUDACXX_INLINE_VISIBILITY auto __sign_begin()
  {
    return __base::__make_iter(_Mantissa + _Exponent);
  }

  _LIBCUDACXX_INLINE_VISIBILITY auto __sign_begin() const
  {
    return __base::__make_iter(_Mantissa + _Exponent);
  }

  _LIBCUDACXX_INLINE_VISIBILITY __uint_least_t<_Exponent> __raw_exponent() const
  {
    __uint_least_t<_Exponent> __ret = 0;
    _CUDA_VSTD::copy_n(__exponent_begin(), _Exponent, __make_exp_iter(&__ret, 0));
    return __ret;
  }

  _LIBCUDACXX_INLINE_VISIBILITY __uint_least_t<_Mantissa> __raw_mantissa() const
  {
    __uint_least_t<_Mantissa> __ret = 0;
    _CUDA_VSTD::copy_n(__mantissa_begin(), _Mantissa, __make_mant_iter(&__ret, 0));
    return __ret;
  }
};

template <_CUDA_VSTD::size_t _Radix, _CUDA_VSTD::size_t _Exponent, _CUDA_VSTD::size_t _Mantissa>
struct __float_repr_traits<__float_repr<_Radix, _Exponent, _Mantissa>>
{
  static const constexpr _CUDA_VSTD::size_t __radix    = _Radix;
  static const constexpr _CUDA_VSTD::size_t __exponent = _Exponent;
  static const constexpr _CUDA_VSTD::size_t __mantissa = _Mantissa + 1; // accounts also for the implicit bit
};

template <_CUDA_VSTD::size_t _Exponent, _CUDA_VSTD::size_t _Mantissa>
struct __is_integral_representable<__float_repr<2, _Exponent, _Mantissa>>
    : _CUDA_VSTD::bool_constant<_CUDA_VSTD::is_integral<__uint_least_t<1 + _Exponent + _Mantissa>>::value>
{};

// Implementation of helper functions for __float_repr.

template <_CUDA_VSTD::size_t _Exponent, _CUDA_VSTD::size_t _Mantissa>
_LIBCUDACXX_INLINE_VISIBILITY constexpr __sign_val
__cuda_float_get_sign(const __float_repr<2, _Exponent, _Mantissa>& __repr)
{
  return *__repr.__sign_begin() == 0 ? __sign_val::__positive : __sign_val::__negative;
}

template <_CUDA_VSTD::size_t _Exponent, _CUDA_VSTD::size_t _Mantissa>
_LIBCUDACXX_INLINE_VISIBILITY constexpr __int_least_t<_Exponent>
__cuda_float_get_exponent(const __float_repr<2, _Exponent, _Mantissa>& __repr)
{
  return __cuda_float_unbias<_Exponent>(__repr.__raw_exponent());
}

template <_CUDA_VSTD::size_t _Exponent, _CUDA_VSTD::size_t _Mantissa>
_LIBCUDACXX_INLINE_VISIBILITY constexpr __uint_least_t<_Mantissa + 1>
__cuda_float_get_mantissa(const __float_repr<2, _Exponent, _Mantissa>& __repr)
{
  auto __raw = __repr.__raw_mantissa();
  return __repr.__raw_exponent() == 0 ? __raw : __cuda_float_mantissa_with_implicit<_Mantissa + 1>(__raw);
}

// Customizations of the generic functions.

template <_CUDA_VSTD::size_t _Exponent, _CUDA_VSTD::size_t _Mantissa>
_LIBCUDACXX_INLINE_VISIBILITY constexpr bool __cuda_float_isinf(const __float_repr<2, _Exponent, _Mantissa>& __repr)
{
  return __repr.__raw_exponent() == ((static_cast<__uint_least_t<_Exponent>>(1) << _Exponent) - 1)
      && __repr.__raw_mantissa() == 0;
}

template <_CUDA_VSTD::size_t _Exponent, _CUDA_VSTD::size_t _Mantissa>
_LIBCUDACXX_INLINE_VISIBILITY constexpr bool __cuda_float_isnan(const __float_repr<2, _Exponent, _Mantissa>& __repr)
{
  return __repr.__raw_exponent() == ((static_cast<__uint_least_t<_Exponent>>(1) << _Exponent) - 1)
      && __repr.__raw_mantissa() != 0;
}

template <_CUDA_VSTD::size_t _Exponent, _CUDA_VSTD::size_t _Mantissa>
_LIBCUDACXX_INLINE_VISIBILITY constexpr __float_repr<2, _Exponent, _Mantissa>
__cuda_float_infinity(_CUDA_VSTD::type_identity<__float_repr<2, _Exponent, _Mantissa>>, __sign_val __sign)
{
  __float_repr<2, _Exponent, _Mantissa> __ret;

  *__ret.__sign_begin() = __sign == __sign_val::__positive ? 0 : 1;
  _CUDA_VSTD::fill_n(__ret.__exponent_begin(), _Exponent, true);
  _CUDA_VSTD::fill_n(__ret.__mantissa_begin(), _Mantissa, false);

  return __ret;
}

template <_CUDA_VSTD::size_t _Exponent, _CUDA_VSTD::size_t _Mantissa>
_LIBCUDACXX_INLINE_VISIBILITY constexpr __float_repr<2, _Exponent, _Mantissa>
__cuda_float_nan(_CUDA_VSTD::type_identity<__float_repr<2, _Exponent, _Mantissa>>, __sign_val __sign)
{
  __float_repr<2, _Exponent, _Mantissa> __ret;

  *__ret.__sign_begin() = __sign == __sign_val::__positive ? 0 : 1;
  _CUDA_VSTD::fill_n(__ret.__exponent_begin(), _Exponent, true);
  *__ret.__mantissa_begin() = true;
  _CUDA_VSTD::fill_n(__ret.__mantissa_begin() + 1, _Mantissa - 1, false);

  return __ret;
}

template <_CUDA_VSTD::size_t _Exponent, _CUDA_VSTD::size_t _Mantissa>
_LIBCUDACXX_INLINE_VISIBILITY constexpr __float_repr<2, _Exponent, _Mantissa> __cuda_float_reconstruct(
  _CUDA_VSTD::type_identity<__float_repr<2, _Exponent, _Mantissa>>,
  __sign_val __sign,
  __int_least_t<_Exponent> __exponent,
  __uint_least_t<_Mantissa + 1> __mantissa)
{
  return {__sign, __exponent, __mantissa};
}

_LIBCUDACXX_END_NAMESPACE_CUDA

#endif
