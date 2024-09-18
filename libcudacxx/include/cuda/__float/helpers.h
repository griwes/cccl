#ifndef _LIBCUDACXX__FLOAT_DEFINITIONS
#define _LIBCUDACXX__FLOAT_DEFINITIONS

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/is_trivially_copy_constructible.h>
#include <cuda/std/__utility/declval.h>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

// Helper traits.

template <_CUDA_VSTD::size_t _Size>
using __int_least_t = _CUDA_VSTD::conditional_t<
  _Size <= 8,
  _CUDA_VSTD::int_least8_t,
  _CUDA_VSTD::conditional_t<
    _Size <= 16,
    _CUDA_VSTD::int_least16_t,
    _CUDA_VSTD::conditional_t<_Size <= 32,
                              _CUDA_VSTD::int_least32_t,
                              _CUDA_VSTD::conditional_t<_Size <= 64,
                                                        _CUDA_VSTD::int_least64_t,
                                                        _CUDA_VSTD::conditional_t<_Size <= 128, __int128, void>>>>>;

template <_CUDA_VSTD::size_t _Size>
using __uint_least_t = _CUDA_VSTD::conditional_t<
  _Size <= 8,
  _CUDA_VSTD::uint_least8_t,
  _CUDA_VSTD::conditional_t<
    _Size <= 16,
    _CUDA_VSTD::uint_least16_t,
    _CUDA_VSTD::conditional_t<
      _Size <= 32,
      _CUDA_VSTD::uint_least32_t,
      _CUDA_VSTD::conditional_t<_Size <= 64,
                                _CUDA_VSTD::uint_least64_t,
                                _CUDA_VSTD::conditional_t<_Size <= 128, unsigned __int128, void>>>>>;

// Enums.

enum class __sign_val
{
  __positive,
  __negative
};

// Useful representation helpers.

template <_CUDA_VSTD::size_t _Exponent, class _Repr>
_LIBCUDACXX_INLINE_VISIBILITY constexpr __int_least_t<_Exponent> __cuda_float_unbias(const _Repr& __repr)
{
  return static_cast<__int_least_t<_Exponent>>(__repr) - (static_cast<__uint_least_t<_Exponent>>(1) << (_Exponent - 1))
       + 1;
}

template <_CUDA_VSTD::size_t _Mantissa, class _Repr>
_LIBCUDACXX_INLINE_VISIBILITY constexpr __uint_least_t<_Mantissa>
__cuda_float_mantissa_with_implicit(const _Repr& __repr)
{
  return __repr | (static_cast<__uint_least_t<_Mantissa>>(1) << (_Mantissa - 1));
}

// Memcpy, but for single objects, with references, and making sure the compiler knows the alignments.

template <class _Tp, class _Up>
_LIBCUDACXX_INLINE_VISIBILITY void __aligned_memcpy(_Tp& __dest, const _Up& __src)
{
  static_assert(sizeof(_Tp) == sizeof(_Up), "");
  static_assert(alignof(_Tp) <= alignof(_Up), "");
  static_assert(_CUDA_VSTD::is_trivially_copy_constructible<_Tp>::value, "");
  static_assert(_CUDA_VSTD::is_trivially_copy_constructible<_Up>::value, "");

  auto __dest_ptr = __builtin_assume_aligned(reinterpret_cast<void*>(&__dest), alignof(_Tp));
  auto __src_ptr  = __builtin_assume_aligned(reinterpret_cast<const void*>(&__src), alignof(_Tp));
  memcpy(__dest_ptr, __src_ptr, sizeof(_Tp));
}

_LIBCUDACXX_END_NAMESPACE_CUDA

#endif
