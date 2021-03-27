#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

using DepthAccessor = torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits>;
using PCMapAccessor = torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits>;
using IntensityAccessor = torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits>;
using GradientAccessor = torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits>;
//using MaskAccessor = torch::PackedTensorAccessor32<bool, 2, torch::RestrictPtrTraits>;

struct matrix3
{
    float3 r1{0.0, 0.0, 0.0};
    float3 r2{0.0, 0.0, 0.0};
    float3 r3{0.0, 0.0, 0.0};

    explicit matrix3(const std::vector<float>& data) {
        r1.x = data[0]; r1.y = data[1]; r1.z = data[2];
        r2.x = data[3]; r2.y = data[4]; r2.z = data[5];
        r3.x = data[6]; r3.y = data[7]; r3.z = data[8];
    }

    __host__ __device__ float3 operator*(const float3& rv) const {
        return make_float3(
                r1.x * rv.x + r1.y * rv.y + r1.z * rv.z,
                r2.x * rv.x + r2.y * rv.y + r2.z * rv.z,
                r3.x * rv.x + r3.y * rv.y + r3.z * rv.z);
    }
};

static uint div_up(const uint a, const uint b) {
    return (a + b - 1) / b;
}

__device__ __forceinline__ static float3 get_vec3(const PCMapAccessor map, const uint i, const uint j) {
    return make_float3(map[i][j][0], map[i][j][1], map[i][j][2]);
}

inline __host__ __device__ float3 operator+(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline __host__ __device__ float3 operator-(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline __host__ __device__ void operator/=(float3 &a, float b) {
    a.x /= b; a.y /= b; a.z /= b;
}

inline __host__ __device__ void operator+=(float3 &a, const float3& b) {
    a.x += b.x; a.y += b.y; a.z += b.z;
}

inline __host__ __device__ void operator-=(float3 &a, const float3& b) {
    a.x -= b.x; a.y -= b.y; a.z -= b.z;
}

inline __host__ __device__ float3 cross(const float3& a, const float3& b) {
    return make_float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}

inline __host__ __device__ float dot(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline __host__ __device__ float length(const float3& v) {
    return sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

inline __host__ __device__ float squared_length(const float3& v) {
    return v.x * v.x + v.y * v.y + v.z * v.z;
}
