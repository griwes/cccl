#ifndef TEST_FLOAT_HELPERS
#define TEST_FLOAT_HELPERS

#include "testcases.h"

template <class F, std::size_t Exponent, std::size_t Mantissa>
struct test_combinator
{
  F f;

  template <class... Args>
  __host__ __device__ bool operator()(const Args&... args) const
  {
    auto&& cases = get_cases<Exponent, Mantissa>();
    for (auto&& case_ : cases)
    {
      auto res = f(args..., case_);
      if (!res)
      {
        return false;
      }
    }

    return true;
  }

  static const constexpr auto exponent = Exponent;
  static const constexpr auto mantissa = Mantissa;
};

template <class F, std::size_t Exponent = F::exponent, std::size_t Mantissa = F::mantissa>
__host__ __device__ test_combinator<F, Exponent, Mantissa> test_combine(F f)
{
  return test_combinator<F, Exponent, Mantissa>{f};
}

template <class UnderTest, class T = UnderTest, class ArgsTuple>
__host__ __device__ auto construct(const ArgsTuple& args_tuple)
{
  return cuda::std::apply(
    [](cuda::__sign_val sign, auto&&... args) {
      auto tag = cuda::std::type_identity<T>();
      if (sign == (cuda::__sign_val) 2)
      {
        return cuda::__cuda_float_infinity(tag, cuda::__sign_val::__positive);
      }
      if (sign == (cuda::__sign_val) 3)
      {
        return cuda::__cuda_float_infinity(tag, cuda::__sign_val::__negative);
      }
      if (sign == (cuda::__sign_val) 4)
      {
        return cuda::__cuda_float_nan(tag, cuda::__sign_val::__positive);
      }
      if (sign == (cuda::__sign_val) 5)
      {
        return cuda::__cuda_float_nan(tag, cuda::__sign_val::__negative);
      }
      return cuda::__cuda_float_convert_reconstruct<T, UnderTest>(sign, args...);
    },
    args_tuple);
}

template <class F, std::size_t Exponent, std::size_t Mantissa>
struct case_verifier
{
  F f;

  template <class... Args>
  __host__ __device__ bool operator()(const Args&... args) const
  {
    using under_test      = cuda::floating_point<2, Exponent, Mantissa>;
    using under_test_repr = cuda::__float_repr<2, Exponent, Mantissa>;

    auto result_repr = static_cast<double>(f(static_cast<under_test>(construct<under_test_repr>(args))...));
    // vvv TODO: that static cast to float is wrong
    auto result_double = static_cast<double>(static_cast<float>(f(construct<under_test_repr, double>(args)...)));
    NV_IF_TARGET(
      NV_IS_HOST,
      fprintf(stderr,
              "%+.20e, %+.20e, %+.20e, %+.20e, %lx, %lx\n",
              construct<under_test_repr, double>(args)...,
              result_repr,
              result_double,
              *(uint64_t*) &result_repr,
              *(uint64_t*) &result_double);)
    return result_repr == result_double
        || (cuda::std::isnan(result_repr) && cuda::std::isnan(result_double)
            && cuda::std::signbit(result_repr) == cuda::std::signbit(result_double));
  }

  static const constexpr auto exponent = Exponent;
  static const constexpr auto mantissa = Mantissa;
};

template <std::size_t Exponent, std::size_t Mantissa, class F>
__host__ __device__ case_verifier<F, Exponent, Mantissa> case_verify(F f)
{
  return case_verifier<F, Exponent, Mantissa>{f};
}

#endif // !TEST_FLOAT_HELPERS
