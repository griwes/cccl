#define _LIBCUDACXX_FLOAT_DISABLE_EXACT_REPRESENTATION_BUILTIN_OPS
#define _LIBCUDACXX_FLOAT_DISABLE_LARGER_REPRESENTATION_BUILTIN_OPS

#include <cuda/float>

#include <cassert>

#include "helpers.h"

int main(int, char**)
{
  assert(test_combine(test_combine(case_verify<8, 23>([](auto&& lhs, auto&& rhs) {
    return lhs + rhs;
  })))());

  return 0;
}
