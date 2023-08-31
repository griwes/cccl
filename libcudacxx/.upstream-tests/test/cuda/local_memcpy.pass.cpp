#include <cuda/barrier>
#include <cuda/pipeline>
#include <cooperative_groups.h>

#pragma nv_diag_suppress static_var_with_dynamic_init

using barrier = cuda::barrier<cuda::thread_scope_block>;

__global__ void kernel(float4 * x) {
    __shared__ float4 smem_x[1];
    __shared__ barrier bar;
    init(&bar, 1);

    //__shared__ cuda::pipeline_shared_state<cuda::thread_scope_block, 2> pss0;
    //auto group = cooperative_groups::this_thread_block();
    //cuda::pipeline<cuda::thread_scope_block> pipe =
        //cuda::make_pipeline(group, &pss0, 10);

    cuda::memcpy_async(smem_x, x, cuda::aligned_size_t<16>(16), bar);
    //cuda::memcpy_async(smem_x, x, cuda::aligned_size_t<16>(16), pipe);
}

int main(int, char **) {
    return 0;
}
