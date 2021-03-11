// Copyright 2004-present Facebook. All Rights Reserved.

#include "Utils.h"

#include <random>

std::vector<Eigen::Vector3f> EquiDistPointsOnSphere(const uint numSamples, const float radius) {
    std::vector<Eigen::Vector3f> points(numSamples);
    const float offset = 2.f / numSamples;

    const float increment = static_cast<float>(M_PI) * (3.f - std::sqrt(5.f));

    for (uint i = 0; i < numSamples; i++) {
        const float y = ((i * offset) - 1) + (offset / 2);
        const float r = std::sqrt(1 - std::pow(y, 2.f));

        const float phi = (i + 1.f) * increment;

        const float x = cos(phi) * r;
        const float z = sin(phi) * r;

        points[i] = radius * Eigen::Vector3f(x, y, z);
    }

    return points;
}

__global__ static void ValidPointsNormalKernel(cudaTextureObject_t normals, cudaTextureObject_t verts,
        unsigned int img_width, unsigned int img_height, int* __restrict__ tri_pos, int* __restrict__ tri_total,
        float4* __restrict__ output_normals, float4* __restrict__ output_xyz, int* __restrict__ output_count) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= img_width || y >= img_height) return;

    float4 normal = tex2D<float4>(normals, (float) x, (float) y);
    float4 xyz = tex2D<float4>(verts, (float) x, (float) y);

    if (normal.w == 0.0f || xyz.w == 0.0f) {
        return;
    }
    unsigned int triInd = (unsigned int)(normal.w + 0.01f) - 1;

    // Compute a proxy for normal direction.
    // Nearby triangles tend to share similar normals so this branching is warp-efficient.
    int normal_dir = normal.x > 0 ? 1 : -1;
    if (fabs(normal.x) < 1e-6) {
        normal_dir = normal.y > 0 ? 1 : -1;
        if (fabs(normal.y) < 1e-6) {
            normal_dir = normal.z > 0 ? 1 : -1;
        }
    }
    atomicAdd(tri_total + triInd, 1);
    atomicAdd(tri_pos + triInd, normal_dir);

    // Gather all data. This step can be largely accelerated using a compaction algorithm.
    int idx = atomicAdd(output_count, 1);
    output_normals[idx] = normal;
    output_xyz[idx] = xyz;
}

unsigned int ValidPointsNormalCUDA(cudaArray_t normals, cudaArray_t verts, unsigned int img_width, unsigned int img_height,
        thrust::device_vector<int>& tri_pos, thrust::device_vector<int>& tri_total, float4* output_normals, float4* output_xyz) {

    cudaResourceDesc normal_res_desc{};     // Will init all fields to 0.
    cudaResourceDesc vert_res_desc{};
    normal_res_desc.resType = vert_res_desc.resType = cudaResourceTypeArray;
    normal_res_desc.res.array.array = normals;
    vert_res_desc.res.array.array = verts;

    // See https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#texture-object-api for default values.
    cudaTextureDesc normal_tex_desc{};
    cudaTextureDesc vert_tex_desc{};

    cudaTextureObject_t normal_tex_obj, vert_tex_obj;
    cudaCreateTextureObject(&normal_tex_obj, &normal_res_desc, &normal_tex_desc, nullptr);
    cudaCreateTextureObject(&vert_tex_obj, &vert_res_desc, &vert_tex_desc, nullptr);

    dim3 dimBlock = dim3(16, 16);
    unsigned int xBlocks = (img_width + dimBlock.x - 1) / dimBlock.x;
    unsigned int yBlocks = (img_height + dimBlock.y - 1) / dimBlock.y;
    dim3 dimGrid = dim3(xBlocks, yBlocks);

    thrust::device_vector<int> nOutput(1, 0);
    ValidPointsNormalKernel<<<dimGrid, dimBlock>>>(normal_tex_obj, vert_tex_obj, img_width, img_height, tri_pos.data().get(),
            tri_total.data().get(), output_normals, output_xyz, nOutput.data().get());
    cudaKernelCheck

    return nOutput[0];
}

void ComputeNormalizationParameters(
        pangolin::Geometry &geom,
        const float buffer, float2& ub_x, float2& ub_y, float2& ub_z) {

    float xMin = 1000000, xMax = -1000000, yMin = 1000000, yMax = -1000000, zMin = 1000000,
            zMax = -1000000;

    pangolin::Image<float> vertices =
            pangolin::get<pangolin::Image<float>>(geom.buffers["geometry"].attributes["vertex"]);

    const std::size_t numVertices = vertices.h;

    ///////// Only consider vertices that were used in some face
    std::vector<unsigned char> verticesUsed(numVertices, 0);
    // turn to true if the vertex is used
    for (const auto &object : geom.objects) {
        auto itVertIndices = object.second.attributes.find("vertex_indices");
        if (itVertIndices != object.second.attributes.end()) {
            pangolin::Image<uint32_t> ibo =
                    pangolin::get<pangolin::Image<uint32_t>>(itVertIndices->second);

            for (uint i = 0; i < ibo.h; ++i) {
                for (uint j = 0; j < 3; ++j) {
                    verticesUsed[ibo(j, i)] = 1;
                }
            }
        }
    }
    /////////

    // compute min max in each dimension
    for (size_t i = 0; i < numVertices; i++) {
        // pass when it's not used.
        if (verticesUsed[i] == 0)
            continue;
        xMin = fmin(xMin, vertices(0, i));
        yMin = fmin(yMin, vertices(1, i));
        zMin = fmin(zMin, vertices(2, i));
        xMax = fmax(xMax, vertices(0, i));
        yMax = fmax(yMax, vertices(1, i));
        zMax = fmax(zMax, vertices(2, i));
    }

    float center_x = (xMax + xMin) / 2.0f;
    float center_y = (yMax + yMin) / 2.0f;
    float center_z = (zMax + zMin) / 2.0f;

    float size_x = (xMax - xMin) + buffer;
    float size_y = (yMax - yMin) + buffer;
    float size_z = (zMax - zMin) + buffer;

    ub_x = make_float2(center_x - size_x / 2.0f, center_x + size_x / 2.0f);
    ub_y = make_float2(center_y - size_y / 2.0f, center_y + size_y / 2.0f);
    ub_z = make_float2(center_z - size_z / 2.0f, center_z + size_z / 2.0f);
}

__device__ int lower_bound(const float* __restrict__ A, float val, int n) {
    int l = 0;
    int h = n;
    while (l < h) {
        int mid = l + (h - l) / 2;
        if (val <= A[mid]) {
            h = mid;
        } else {
            l = mid + 1;
        }
    }
    return l;
}

void LinearizeObject(pangolin::Geometry& geom) {
    int total_num_faces = 0;
    for (const auto &object : geom.objects) {
        auto it_vert_indices = object.second.attributes.find("vertex_indices");
        if (it_vert_indices != object.second.attributes.end()) {
            pangolin::Image<uint32_t> ibo =
                    pangolin::get<pangolin::Image<uint32_t>>(it_vert_indices->second);

            total_num_faces += ibo.h;
        }
    }

    //      const int total_num_indices = total_num_faces * 3;
    pangolin::ManagedImage<uint8_t> new_buffer(3 * sizeof(uint32_t), total_num_faces);

    pangolin::Image<uint32_t> new_ibo =
            new_buffer.UnsafeReinterpret<uint32_t>().SubImage(0, 0, 3, total_num_faces);

    int index = 0;

    for (const auto &object : geom.objects) {
        auto it_vert_indices = object.second.attributes.find("vertex_indices");
        if (it_vert_indices != object.second.attributes.end()) {
            pangolin::Image<uint32_t> ibo =
                    pangolin::get<pangolin::Image<uint32_t>>(it_vert_indices->second);

            for (int i = 0; i < ibo.h; ++i) {
                new_ibo.Row(index).CopyFrom(ibo.Row(i));
                ++index;
            }
        }
    }

    geom.objects.clear();
    auto faces = geom.objects.emplace(std::string("mesh"), pangolin::Geometry::Element());

    faces->second.Reinitialise(3 * sizeof(uint32_t), total_num_faces);

    faces->second.CopyFrom(new_buffer);

    new_ibo = faces->second.UnsafeReinterpret<uint32_t>().SubImage(0, 0, 3, total_num_faces);
    faces->second.attributes["vertex_indices"] = new_ibo;
}

__global__ void TriangleAreaKernel(unsigned char* __restrict__ vertices, size_t vertices_pitch,
        unsigned char* __restrict__ triangles, size_t triangles_pitch, size_t num_tris, float* __restrict__ areas) {
    unsigned int tri_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (tri_id >= num_tris) {
        return;
    }
    auto* inds = (uint32_t*) (triangles + triangles_pitch * tri_id);

    auto* ap = (float*) (vertices + vertices_pitch * inds[0]);
    auto* bp = (float*) (vertices + vertices_pitch * inds[1]);
    auto* cp = (float*) (vertices + vertices_pitch * inds[2]);

    Eigen::Vector3f a(ap[0], ap[1], ap[2]);
    Eigen::Vector3f b(bp[0], bp[1], bp[2]);
    Eigen::Vector3f c(cp[0], cp[1], cp[2]);

    const Eigen::Vector3f ab = b - a;
    const Eigen::Vector3f ac = c - a;
    float abnorm = ab.norm(), acnorm = ac.norm();
    float costheta = ab.dot(ac) / (abnorm * acnorm);

    if (costheta < -1) // meaning theta is pi
        costheta = cos(static_cast<float>(M_PI) * 359.f / 360);
    else if (costheta > 1) // meaning theta is zero
        costheta = cos(static_cast<float>(M_PI) * 1.f / 360);

    const float sinTheta = sqrt(1 - costheta * costheta);

    float area = 0.5f * abnorm * acnorm * sinTheta;
    if (isnan(area)) {
        area = 0.0f;
    }

    areas[tri_id] = area;
}

__global__ void RNGSetupKernel(curandState *state, size_t num_kernel) {
    unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= num_kernel) {
        return;
    }
    /* Each thread gets same seed, a different sequence
       number, no offset */
    curand_init(1000, id, 0, &state[id]);
}
