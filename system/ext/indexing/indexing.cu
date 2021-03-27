#include <torch/extension.h>
#include <thrust/device_vector.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

using CountAccessor = torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits>;
using IndexAccessor = torch::PackedTensorAccessor32<long, 1, torch::RestrictPtrTraits>;
using PackedIndAccessor = torch::PackedTensorAccessor32<long, 2, torch::RestrictPtrTraits>;
using ValueAccessor = torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits>;

static uint div_up(const uint a, const uint b) {
    return (a + b - 1) / b;
}

__global__ static void pack_batch_kernel(const IndexAccessor indices, PackedIndAccessor packed_inds, const uint n_all,
        const uint n_batch, const uint n_point, int* __restrict__ filled_count) {
    const uint i_data = blockIdx.x * blockDim.x + threadIdx.x;
    if (i_data >= n_all) {
        return;
    }

    long i_group = indices[i_data];
    if (i_group >= n_batch) {
        return;
    }

    // Get one valid id.
    int cur_count = atomicAdd(filled_count + i_group, 1);
    if (cur_count >= n_point) {
        return;
    }
    packed_inds[i_group][cur_count] = i_data;
}

__device__ static float atomicMax(float* __restrict__ address, float val) {
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
                          __float_as_int(::fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

__global__ static void groupby_max_kernel(const ValueAccessor values, const IndexAccessor indices, ValueAccessor reduced_values) {
    const uint i = blockIdx.x;
    const uint l = threadIdx.x;

    float value = values[i][l];
    long index = indices[i];

    float* rptr = reduced_values[index].data() + l;
    atomicMax(rptr, value);
}

__global__ static void groupby_sum_kernel(const ValueAccessor values, const IndexAccessor indices, ValueAccessor sum_values, CountAccessor sum_counts) {
    const uint i = blockIdx.x;
    const uint l = threadIdx.x;

    float value = values[i][l];
    long index = indices[i];

    float* rptr = sum_values[index].data() + l;
    int* iptr = &(sum_counts[index]);

    atomicAdd(rptr, value);
    atomicAdd(iptr, 1);
}

torch::Tensor pack_batch(torch::Tensor indices, uint n_batch, uint n_point) {
    CHECK_INPUT(indices);
    torch::Tensor packed_inds = torch::empty({n_batch, n_point}, torch::dtype(torch::kLong).device(torch::kCUDA));
    thrust::device_vector<int> filled_count(n_batch, 0);
    const uint n_all = indices.size(0);

    dim3 dimBlock = dim3(128);
    dim3 dimGrid = dim3(div_up(n_all, dimBlock.x));
    pack_batch_kernel<<<dimGrid, dimBlock>>>(
            indices.packed_accessor32<long, 1, torch::RestrictPtrTraits>(),
            packed_inds.packed_accessor32<long, 2, torch::RestrictPtrTraits>(),
            n_all, n_batch, n_point, filled_count.data().get());
    return packed_inds;
}


std::vector<torch::Tensor> groupby_sum(torch::Tensor values, torch::Tensor indices, uint C) {
    CHECK_INPUT(values);
    CHECK_INPUT(indices);

    const uint N = values.size(0);
    const uint L = values.size(1);

    torch::Tensor sum_values = torch::zeros({C, L}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    torch::Tensor sum_count = torch::zeros({C}, torch::dtype(torch::kInt32).device(torch::kCUDA));

    dim3 dimBlock = dim3(L);
    dim3 dimGrid = dim3(N);
    groupby_sum_kernel<<<dimGrid, dimBlock>>>(
        values.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        indices.packed_accessor32<long, 1, torch::RestrictPtrTraits>(),
        sum_values.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        sum_count.packed_accessor32<int, 1, torch::RestrictPtrTraits>()
    );

    return {sum_values, sum_count};
}
