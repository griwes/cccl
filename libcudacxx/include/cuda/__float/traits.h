#ifndef _LIBCUDACXX__FLOAT_TRAITS
#define _LIBCUDACXX__FLOAT_TRAITS

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_floating_point.h>
#include <cuda/std/bit>
#include <cuda/std/limits>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

// __float_repr_traits: traits alloing extraction of radix, exponent, and mantissa of a given float-like storage type.
// This includes custom-exponent-and-mantissa types defined later, but also standard floating point types, and CUDA
// runtime types such as __half.

template <class _Fp, class = void>
struct __float_repr_traits;

template <class _Fp>
struct __float_repr_traits<
  _Fp,
  _CUDA_VSTD::__enable_if_t<_CUDA_VSTD::is_floating_point<_Fp>::value && _CUDA_VSTD::numeric_limits<_Fp>::is_iec559
                            && _CUDA_VSTD::numeric_limits<_Fp>::radix == 2>>
{
  using __limits                                       = _CUDA_VSTD::numeric_limits<_Fp>;
  static const constexpr _CUDA_VSTD::size_t __radix    = 2;
  static const constexpr _CUDA_VSTD::size_t __exponent = _CUDA_VSTD::countr_zero(
    _CUDA_VSTD::bit_ceil(static_cast<_CUDA_VSTD::size_t>(__limits::max_exponent - __limits::min_exponent)));
  static const constexpr _CUDA_VSTD::size_t __mantissa = __limits::digits;
};

static_assert(__float_repr_traits<float>::__radix == 2);
static_assert(__float_repr_traits<float>::__exponent == 8);
static_assert(__float_repr_traits<float>::__mantissa == 23 + 1);

static_assert(__float_repr_traits<double>::__radix == 2);
static_assert(__float_repr_traits<double>::__exponent == 11);
static_assert(__float_repr_traits<double>::__mantissa == 52 + 1);

// Trait for finding a builtin type of the requested representation, if available.

template <_CUDA_VSTD::size_t _Radix, _CUDA_VSTD::size_t _Exponent, _CUDA_VSTD::size_t _Mantissa, class = void>
struct __float_of_representation
{
  static const constexpr bool __exists = false;
};

#define _FLOAT_OF_REPR(__type)                                                  \
  template <>                                                                   \
  struct __float_of_representation<__float_repr_traits<__type>::__radix,        \
                                   __float_repr_traits<__type>::__exponent,     \
                                   __float_repr_traits<__type>::__mantissa - 1> \
  {                                                                             \
    static const constexpr bool __exists = true;                                \
    using type                           = __type;                              \
  };

_FLOAT_OF_REPR(float)
_FLOAT_OF_REPR(double)

#undef _FLOAT_OF_REPR

_LIBCUDACXX_END_NAMESPACE_CUDA

#endif
