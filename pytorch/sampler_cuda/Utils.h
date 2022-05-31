// Copyright 2004-present Facebook. All Rights Reserved.
// The GPU version is drafted by heiwang1997@github.com

#include <vector>

#include <pangolin/geometry/geometry.h>
#include <pangolin/pangolin.h>
#include <Eigen/Core>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <curand_kernel.h>

std::vector<Eigen::Vector3f> EquiDistPointsOnSphere(const uint numSamples, const float radius);

unsigned int ValidPointsNormalCUDA(cudaArray_t normals, cudaArray_t verts, unsigned int img_width, unsigned int img_height,
                           thrust::device_vector<int>& tri_pos, thrust::device_vector<int>& tri_total,
                           float4* output_normals, float4* output_xyz);

__device__ int lower_bound(const float* __restrict__ A, float val, int n);

__global__ void TriangleAreaKernel(unsigned char* __restrict__ vertices, size_t vertices_pitch,
        unsigned char* __restrict__ triangles, size_t triangles_pitch, size_t num_tris, float* __restrict__ areas);

__global__ void RNGSetupKernel(curandState *state, size_t num_kernel);

void ComputeNormalizationParameters(
    pangolin::Geometry& geom,
    const float buffer, float2& ub_x, float2& ub_y, float2& ub_z);

void LinearizeObject(pangolin::Geometry& geom);

#ifndef cudaSafeCall
#define cudaSafeCall(err) __cudaSafeCall(err, __FILE__, __LINE__)

inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
    if( cudaSuccess != err) {
		printf("%s(%i) : cudaSafeCall() Runtime API error : %s.\n",
                file, line, cudaGetErrorString(err) );
        exit(-1);
    }
}

#endif

#ifndef cudaKernelCheck

// For normal operation
#define cudaKernelCheck

// For debugging purposes
//#define cudaKernelCheck { cudaDeviceSynchronize(); __cudaSafeCall(cudaPeekAtLastError(), __FILE__, __LINE__); }

#endif