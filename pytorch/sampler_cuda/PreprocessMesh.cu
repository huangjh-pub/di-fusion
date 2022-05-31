// Copyright 2004-present Facebook. All Rights Reserved.
// The GPU version is drafted by heiwang1997@github.com

#define FLANN_USE_CUDA
#include <flann/flann.hpp>

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <pangolin/geometry/geometry.h>
#include <pangolin/geometry/glgeometry.h>
#include <pangolin/gl/gl.h>
#include <pangolin/pangolin.h>
#include <pangolin/gl/glcuda.h>

#include <CLI/CLI.hpp>

#include "Utils.h"
#include <cuda_gl_interop.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <curand_kernel.h>

extern pangolin::GlSlProgram GetShaderProgram();
static const int METHOD2_SAMPLES_MULT = 10;

__global__ static void SampleUniformKernel(size_t num_uniform_sample, float2 ub_x, float2 ub_y, float2 ub_z,
                                           curandState *rng_state, float4* output) {
    unsigned int sample_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (sample_id >= num_uniform_sample) {
        return;
    }
    curandState local_rng = rng_state[sample_id];

    float4 pos;
    pos.x = curand_uniform(&local_rng) * (ub_x.y - ub_x.x) + ub_x.x;
    pos.y = curand_uniform(&local_rng) * (ub_y.y - ub_y.x) + ub_y.x;
    pos.z = curand_uniform(&local_rng) * (ub_z.y - ub_z.x) + ub_z.x;
    pos.w = 0.0f;
    output[sample_id] = pos;
}

__device__ __forceinline__ static float4 toFloat4(const Eigen::Vector3f& vec) {
    return make_float4(vec(0), vec(1), vec(2), 0.0f);
}

__global__ static void SampleSurfacePointKernel(size_t num_sample, float4* __restrict__ ref_xyz,
        float4* __restrict__ ref_normal, float total_area, curandState *rng_state,
        const float* __restrict__ tri_cdf, unsigned char* __restrict__ triangles, size_t triangles_pitch,
        size_t num_tris, unsigned char* __restrict__ vertices, size_t vertices_pitch, int n_samples) {
    unsigned int sample_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (sample_id >= num_sample) {
        return;
    }
    curandState local_rng = rng_state[sample_id];

    // Generate a uniform number, binary search that in cdf to get triangle id.
    const float u = curand_uniform(&local_rng) * total_area;
    int tri_id = lower_bound(tri_cdf, u, num_tris);
    auto* inds = (uint32_t*) (triangles + triangles_pitch * tri_id);

    // Randomly sample in that triangle.
    auto* ap = (float*) (vertices + vertices_pitch * inds[0]);
    auto* bp = (float*) (vertices + vertices_pitch * inds[1]);
    auto* cp = (float*) (vertices + vertices_pitch * inds[2]);

    Eigen::Vector3f va(ap[0], ap[1], ap[2]);
    Eigen::Vector3f vb(bp[0], bp[1], bp[2]);
    Eigen::Vector3f vc(cp[0], cp[1], cp[2]);
    float4 normal = toFloat4((vb - va).cross(vc - va).normalized());

//#pragma unroll
    for (int k = 0; k < n_samples; ++k) {
        size_t buf_id = sample_id * n_samples + k;
        float r1 = curand_uniform(&local_rng);
        float r2 = curand_uniform(&local_rng);
        float wa = 1 - sqrt(r1);
        float wb = (1 - wa) * (1 - r2);
        float wc = r2 * (1 - wa);
        ref_xyz[buf_id] = toFloat4(wa * va + wb * vb + wc * vc);
        ref_normal[buf_id] = normal;
    }

    rng_state[sample_id] = local_rng;
}

__global__ static void SamplePointKernel(size_t num_half_sample, float4* output,
        float total_area, curandState *rng_state, const float* __restrict__ tri_cdf,
        unsigned char* __restrict__ triangles, size_t triangles_pitch, size_t num_tris,
        unsigned char* __restrict__ vertices, size_t vertices_pitch, float small_std, float large_std) {
    unsigned int sample_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (sample_id >= num_half_sample) {
        return;
    }
    curandState local_rng = rng_state[sample_id];

    // Generate a uniform number, binary search that in cdf to get triangle id.
    const float u = curand_uniform(&local_rng) * total_area;
    int tri_id = lower_bound(tri_cdf, u, num_tris);
    auto* inds = (uint32_t*) (triangles + triangles_pitch * tri_id);

    // Randomly sample in that triangle.
    const float r1 = curand_uniform(&local_rng);
    const float r2 = curand_uniform(&local_rng);

    auto* ap = (float*) (vertices + vertices_pitch * inds[0]);
    auto* bp = (float*) (vertices + vertices_pitch * inds[1]);
    auto* cp = (float*) (vertices + vertices_pitch * inds[2]);

    const float wa = 1 - sqrt(r1);
    const float wb = (1 - wa) * (1 - r2);
    const float wc = r2 * (1 - wa);

    float4 pos1, pos2;
    pos1.x = wa * ap[0] + wb * bp[0] + wc * cp[0] + curand_normal(&local_rng) * small_std;
    pos1.y = wa * ap[1] + wb * bp[1] + wc * cp[1] + curand_normal(&local_rng) * small_std;
    pos1.z = wa * ap[2] + wb * bp[2] + wc * cp[2] + curand_normal(&local_rng) * small_std;
    pos1.w = 0.0f;

    pos2.x = wa * ap[0] + wb * bp[0] + wc * cp[0] + curand_normal(&local_rng) * large_std;
    pos2.y = wa * ap[1] + wb * bp[1] + wc * cp[1] + curand_normal(&local_rng) * large_std;
    pos2.z = wa * ap[2] + wb * bp[2] + wc * cp[2] + curand_normal(&local_rng) * large_std;
    pos2.w = 0.0f;

    output[sample_id] = pos1;
    output[sample_id + num_half_sample] = pos2;

    rng_state[sample_id] = local_rng;
}

__global__ static void ComputeSDFKernel(size_t num_samples, int num_votes, const float4* __restrict__ ref_xyz,
        const float4* __restrict__ ref_normals, const int* __restrict__ knn_index,
        float4* __restrict__ sample_xyz, float stdv, float max_ref_dist) {
    unsigned int sample_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (sample_id >= num_samples) {
        return;
    }

    float4 cur_xyz = sample_xyz[sample_id];
    Eigen::Vector3f sample_pos(cur_xyz.x, cur_xyz.y, cur_xyz.z);

    float sdf;
    int num_pos = 0;
    for (int vote_i = 0; vote_i < num_votes; ++vote_i) {
        int cur_ind = knn_index[sample_id * num_votes + vote_i];
        float4 nb_xyz = ref_xyz[cur_ind];
        float4 nb_normal = ref_normals[cur_ind];

        Eigen::Vector3f nb_pos(nb_xyz.x, nb_xyz.y, nb_xyz.z);
        Eigen::Vector3f nb_norm(nb_normal.x, nb_normal.y, nb_normal.z);
        Eigen::Vector3f ray_vec = sample_pos - nb_pos;
        float ray_vec_len = ray_vec.norm();

        // SDF value will take the first vote. (nearest neighbour)
        if (vote_i == 0) {
            if (ray_vec_len > max_ref_dist) {
                // Just invalidate this point.
                num_pos = 1;
                break;
            }
            if (ray_vec_len < stdv) {
                sdf = abs(nb_norm.dot(ray_vec));
            } else {
                sdf = ray_vec_len;
            }
        }
        float d = nb_norm.dot(ray_vec / ray_vec_len);
        if (d > 0) {
            num_pos += 1;
        }
    }

    if (num_pos == 0) {
        sample_xyz[sample_id].w = -sdf;
    } else if (num_pos == num_votes) {
        sample_xyz[sample_id].w = sdf;
    } else {
        sample_xyz[sample_id].w = NAN;
    }
}

struct Keep3Functor {
    __host__ __device__ float4 operator()(const float4 &x) const
    {return make_float4(x.x, x.y, x.z, 0.0);}
};

struct ValidWFunctor {
    __host__ __device__ bool operator()(const float4 &x) const
    {
        return !isnan(x.w);
    }
};

void GenerateSDFSamples(
        int sample_method,
        pangolin::GlGeometry &geom,
        int num_half_surface_sample,
        int num_uniform_sample,
        thrust::device_vector<float4>& ref_xyz,
        thrust::device_vector<float4>& ref_normal,
        float var_small, float var_large, float max_ref_dist,
        int num_votes,
        thrust::host_vector<float4>& valid_data,
        float2 ub_x, float2 ub_y, float2 ub_z) {

    //// Generate sampled points, by first sample from triangle and then perturb.
    const int num_surface_sample = num_half_surface_sample * 2;
    const int num_total_sample = num_surface_sample + num_uniform_sample;

    // Map OpenGL geometry into cuda.
    cudaGraphicsResource_t vbo_handle;
    cudaGraphicsResource_t ibo_handle;
    unsigned char* vbo_data; size_t vbo_nbytes;
    size_t vbo_stride = geom.buffers["geometry"].attributes["vertex"].stride_bytes;
    unsigned char* ibo_data; size_t ibo_nbytes; size_t ibo_stride = 0;
    size_t num_tris = 0;

    cudaSafeCall(cudaGraphicsGLRegisterBuffer(&vbo_handle, geom.buffers["geometry"].bo, cudaGraphicsMapFlagsReadOnly));
    for (const auto& object : geom.objects) {
        // assert: object.first == "mesh"
        auto it_vert_indices = object.second.attributes.find("vertex_indices");
        if (it_vert_indices != object.second.attributes.end()) {
            cudaSafeCall(cudaGraphicsGLRegisterBuffer(&ibo_handle, object.second.bo, cudaGraphicsMapFlagsReadOnly));
            ibo_stride = it_vert_indices->second.stride_bytes;
            num_tris = it_vert_indices->second.num_elements;
            break;
        }
    }
    cudaSafeCall(cudaGraphicsMapResources(1, &vbo_handle));
    cudaSafeCall(cudaGraphicsMapResources(1, &ibo_handle));
    cudaSafeCall(cudaGraphicsResourceGetMappedPointer((void**)&vbo_data, &vbo_nbytes, vbo_handle));
    cudaSafeCall(cudaGraphicsResourceGetMappedPointer((void**)&ibo_data, &ibo_nbytes, ibo_handle));

    // Compute triangle areas.
    thrust::device_vector<float> tri_area(num_tris);
    {
        dim3 dimBlock = dim3(128);
        dim3 dimGrid = dim3((num_tris + dimBlock.x - 1) / dimBlock.x);
        TriangleAreaKernel<<<dimGrid, dimBlock>>>(vbo_data, vbo_stride, ibo_data, ibo_stride, num_tris, tri_area.data().get());
        cudaKernelCheck
    }

    // Convert to CDF.
    thrust::inclusive_scan(tri_area.begin(), tri_area.end(), tri_area.begin());
    float total_area = tri_area.back();

    // Allocate space for RNG.
    curandState* p_rng_state;
    int num_rng = std::max(num_half_surface_sample, num_uniform_sample);
    cudaSafeCall(cudaMalloc((void**)&p_rng_state, num_rng * sizeof(curandState)));
    {
        dim3 dimBlock = dim3(256);
        dim3 dimGrid = dim3((num_rng + dimBlock.x - 1) / dimBlock.x);
        RNGSetupKernel<<<dimGrid, dimBlock>>>(p_rng_state, num_rng);
        cudaKernelCheck
    }

    // If use method2, just make samples and write to ref_xyz and ref_normal
    if (sample_method == 2) {
        ref_xyz.resize(num_rng * METHOD2_SAMPLES_MULT);
        ref_normal.resize(num_rng * METHOD2_SAMPLES_MULT);
        dim3 dimBlock(256);
        dim3 dimGrid((num_rng + dimBlock.x - 1) / dimBlock.x);
        SampleSurfacePointKernel<<<dimGrid, dimBlock>>>(num_rng, ref_xyz.data().get(), ref_normal.data().get(),
                total_area, p_rng_state, tri_area.data().get(), ibo_data, ibo_stride, num_tris, vbo_data, vbo_stride, METHOD2_SAMPLES_MULT);
        cudaKernelCheck
    }

    // Sample and perturb surface samples according to CDF.
    thrust::device_vector<float4> sampled_points(num_total_sample);
    {
        dim3 dimBlock = dim3(256);
        dim3 dimGrid = dim3((num_half_surface_sample + dimBlock.x - 1) / dimBlock.x);
        SamplePointKernel<<<dimGrid, dimBlock>>>(num_half_surface_sample, sampled_points.data().get(),
                total_area, p_rng_state, tri_area.data().get(), ibo_data, ibo_stride, num_tris, vbo_data, vbo_stride,
                std::sqrt(var_small), std::sqrt(var_large));
        cudaKernelCheck
    }

    cudaSafeCall(cudaGraphicsUnmapResources(1, &vbo_handle));
    cudaSafeCall(cudaGraphicsUnmapResources(1, &ibo_handle));
    cudaSafeCall(cudaGraphicsUnregisterResource(vbo_handle));
    cudaSafeCall(cudaGraphicsUnregisterResource(ibo_handle));

    // Also, add uniform samples.
    if (num_uniform_sample > 0) {
        dim3 dimBlock = dim3(256);
        dim3 dimGrid = dim3((num_uniform_sample + dimBlock.x - 1) / dimBlock.x);
        SampleUniformKernel<<<dimGrid, dimBlock>>>(num_uniform_sample, ub_x, ub_y, ub_z, p_rng_state,
                sampled_points.data().get() + num_surface_sample);
        cudaKernelCheck
    }
    cudaSafeCall(cudaFree(p_rng_state));

    //// Query all the generated samples to the view-sampled geometry, to get the sdf value.
    thrust::transform(ref_xyz.begin(), ref_xyz.end(), ref_xyz.begin(), Keep3Functor());
    // thrust::transform(sampled_points.begin(), sampled_points.end(), sampled_points.begin(), Keep3Functor());

    // Do kNN to retrieve nearest idx for all queries.
    thrust::device_vector<float> dist(num_total_sample * num_votes);
    thrust::device_vector<int> indices(num_total_sample * num_votes);

    std::cout << ref_xyz.size() << ", " << ref_normal.size() << std::endl;
    flann::Matrix<float> knn_ref((float*)ref_xyz.data().get(), ref_xyz.size(), 3, 4 * sizeof(float));
    flann::KDTreeCuda3dIndexParams knn_params;
    knn_params["input_is_gpu_float4"] = true;
    flann::KDTreeCuda3dIndex<flann::L2<float> > knn_index(knn_ref, knn_params);
    knn_index.buildIndex();
    flann::Matrix<float> knn_dist((float*)dist.data().get(), num_total_sample, num_votes);
    flann::Matrix<int> knn_indices((int*)indices.data().get(), num_total_sample, num_votes);
    flann::Matrix<float> knn_query((float*)sampled_points.data().get(), num_total_sample, 3, 4 * sizeof(float));
    flann::SearchParams params;
    params.matrices_in_gpu_ram = true;
    params.sorted = true;
    knn_index.knnSearch(knn_query, knn_indices, knn_dist, num_votes, params);

    // Compute SDF for the samples. Invalid samples' sdf will be marked NaN.
    {
        dim3 dimBlock = dim3(256);
        dim3 dimGrid = dim3((num_total_sample + dimBlock.x - 1) / dimBlock.x);
        ComputeSDFKernel<<<dimGrid, dimBlock>>>(num_total_sample, num_votes, ref_xyz.data().get(), ref_normal.data().get(),
                                                indices.data().get(), sampled_points.data().get(),
                                                std::sqrt(var_small), max_ref_dist);
        cudaKernelCheck
    }

    // Copy and delete all invalid sdfs.
    thrust::device_vector<float4> sampled_points_valid(num_total_sample);
    auto result_end = thrust::copy_if(sampled_points.begin(), sampled_points.end(),
                                      sampled_points_valid.begin(), ValidWFunctor());
    valid_data.resize(thrust::distance(sampled_points_valid.begin(), result_end));
    thrust::copy(sampled_points_valid.begin(), result_end, valid_data.begin());
}

struct TriTestFunctor {
    __host__ __device__ int operator()(const int& x, const int& y) const {
        return (abs(x) == y) ? y : -1;
    }
};

struct PlusPositiveFunctor {
    __host__ __device__ int operator()(const int &lhs, const int &rhs) const {
        if (lhs < 0) return rhs;
        if (rhs < 0) return lhs;
        return lhs + rhs;
    }
};

int main(int argc, char **argv) {
    std::string meshFileName;
    bool vis = false;

    std::string outputFileName;
    std::string cameraFileName;
    std::string surfaceFileName;        // For output of sampled surface.
    std::string referenceFileName;
    float variance = 0.005;
    int num_sample = 500000;
    float num_samp_near_surf_ratio = 47.0f / 50.0f;
    float uniform_sample_bbox_expand = 1.2f;
    int reference_method = 1;
    float max_ref_dist = 1.0e8f;

    CLI::App app{"PreprocessMesh"};
    app.add_option("-m", meshFileName, "Mesh File Name for Reading")->required();
    app.add_option("-o", outputFileName, "Output file path")->required();
    app.add_option("--surface", surfaceFileName, "Output surface file path")->required();
    app.add_option("-s", num_sample, "Number of attempted samples");
    app.add_option("-p", num_samp_near_surf_ratio, "Portion of near surface");
    app.add_option("-e", uniform_sample_bbox_expand, "Expansion of the bounding box for uniform sampling");
    app.add_option("--var", variance, "Set Variance");
    app.add_option("-r", reference_method, "Method 1 is camera. Method 2 is mesh normal. Method 3 is reference points.")->required();
    app.add_option("-c", cameraFileName, "Name of the camera definition (required for ref-method 1).");
    app.add_option("--ref", referenceFileName, "Name of the reference file.");
    app.add_option("--max_ref_dist", max_ref_dist, "Maximum reference dist to prune some invalid data.");
    app.add_flag("-v", vis, "enable visualization");

    CLI11_PARSE(app, argc, argv);

    if (reference_method == 1 && cameraFileName.empty()) {
        std::cout << "Camera not provided!" << std::endl;
        return -1;
    }

    float second_variance = variance / 5;
    std::cout << "variance: " << variance << " second: " << second_variance << std::endl;

    pangolin::Geometry geom = pangolin::LoadGeometry(meshFileName);
    std::cout << geom.objects.size() << " objects" << std::endl;

    // linearize the object indices
    LinearizeObject(geom);

    // remove textures
    geom.textures.clear();

    // Get bounding boxes.
    float2 ub_x, ub_y, ub_z;
    ComputeNormalizationParameters(geom, uniform_sample_bbox_expand, ub_x, ub_y, ub_z);

    // Get surface samples and normals
    thrust::device_vector<float4> point_normals;
    thrust::device_vector<float4> point_verts;

    if (vis)
        pangolin::CreateWindowAndBind("Main", 640, 480);
    else
        pangolin::CreateWindowAndBind("Main", 1, 1);

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
    glPixelStorei(GL_UNPACK_SKIP_PIXELS, 0);
    glPixelStorei(GL_UNPACK_SKIP_ROWS, 0);
    glEnable(GL_DEPTH_TEST);
    glDisable(GL_DITHER);
    glDisable(GL_POINT_SMOOTH);
    glDisable(GL_LINE_SMOOTH);
    glDisable(GL_POLYGON_SMOOTH);
    glHint(GL_POINT_SMOOTH, GL_DONT_CARE);
    glHint(GL_LINE_SMOOTH, GL_DONT_CARE);
    glHint(GL_POLYGON_SMOOTH_HINT, GL_DONT_CARE);
    glDisable(GL_MULTISAMPLE_ARB);
    glShadeModel(GL_FLAT);

    // Check if OpenGL direct resource mapping is usable. (interop with CUDA)
    // This may fail for software rendering (e.g. when using remote-glx).
    {
        unsigned int nGLDevCount;
        int cudaDevices[4];
        cudaSafeCall(cudaGLGetDevices(&nGLDevCount, cudaDevices, 4, cudaGLDeviceListAll));
        if (nGLDevCount == 0) {
            std::cerr << "No OpenGL hardware found." << std::endl;
            return -1;
        }
    }

    // Map geometry to gpu.
    pangolin::GlGeometry gl_geom = pangolin::ToGlGeometry(geom);

    if (reference_method == 1) {
        // Method1: Use camera to render the mesh. Take inverse camera ray as normal
        pangolin::Image<uint32_t> modelFaces = pangolin::get<pangolin::Image<uint32_t>>(
                geom.objects.begin()->second.attributes["vertex_indices"]);
        size_t num_tri = modelFaces.h;

        // Load in camera definition
        float max_dist;
        float z_extent[2];
        std::vector<pangolin::OpenGlMatrix> view_matrices;
        {
            std::ifstream fin(cameraFileName, std::ios::in | std::ios::binary);
            if (!fin) {
                std::cerr << "File " << cameraFileName << " not found!" << std::endl;
                return -1;
            }
            fin.read((char*)&max_dist, 4);
            fin.read((char*)z_extent, sizeof(float) * 2);
            float buffer[16];
            while (true) {
                fin.read((char*)buffer, sizeof(float) * 16);
                if (!fin) break;
                view_matrices.emplace_back();
                auto& m = view_matrices.back();
                for (int i = 0; i < 16; ++i) {
                    m.m[i] = buffer[i];
                }
            }
            std::cout << "Available cameras = " << view_matrices.size() << std::endl;
        }

        // Define Projection and initial ModelView matrix
        pangolin::OpenGlRenderState s_cam2(
                pangolin::ProjectionMatrixOrthographic(-max_dist, max_dist, max_dist, -max_dist, z_extent[0], z_extent[1]),
                pangolin::ModelViewLookAt(0, 0, -1, 0, 0, 0, pangolin::AxisY));

        // Create Interactive View in window
        pangolin::GlSlProgram prog = GetShaderProgram();

        if (vis) {
            pangolin::OpenGlRenderState s_cam(
                    pangolin::ProjectionMatrixOrthographic(-max_dist, max_dist, -max_dist, max_dist, z_extent[0], z_extent[1]),
                    pangolin::ModelViewLookAt(0, 0, -1, 0, 0, 0, pangolin::AxisY));
            s_cam.SetModelViewMatrix(view_matrices[0]);
            pangolin::Handler3D handler(s_cam);
            pangolin::View &d_cam = pangolin::CreateDisplay()
                    .SetBounds(0.0, 1.0, 0.0, 1.0, -640.0f / 480.0f)
                    .SetHandler(&handler);

            while (!pangolin::ShouldQuit()) {
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
                d_cam.Activate(s_cam);
                prog.Bind();
                prog.SetUniform("MVP", s_cam.GetProjectionModelViewMatrix());
                prog.SetUniform("V", s_cam.GetModelViewMatrix());
                pangolin::GlDraw(prog, gl_geom, nullptr);
                prog.Unbind();

                // Swap frames and Process Events
                pangolin::FinishFrame();
            }
        }

        // Create Framebuffer with attached textures
        size_t w = 400;
        size_t h = 400;
        size_t wh = w * h;
        pangolin::GlRenderBuffer zbuffer(w, h, GL_DEPTH_COMPONENT32);
        pangolin::GlTexture normals(w, h, GL_RGBA32F);
        pangolin::GlTexture vertices(w, h, GL_RGBA32F);
        pangolin::GlFramebuffer framebuffer(vertices, normals, zbuffer);

        // Register CUDA buffer.
        cudaGraphicsResource_t cudaResourceNormals;
        cudaGraphicsResource_t cudaResourceVertices;
        cudaSafeCall(cudaGraphicsGLRegisterImage(&cudaResourceNormals, normals.tid, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsReadOnly));
        cudaSafeCall(cudaGraphicsGLRegisterImage(&cudaResourceVertices, vertices.tid, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsReadOnly));

        // Thrust container for normals and vertices;
        unsigned int valid_point_num = 0;
        point_normals.resize(wh * 2);
        point_verts.resize(wh * 2);

        thrust::device_vector<int> tri_pos(num_tri, 0);
        thrust::device_vector<int> tri_all(num_tri, 0);

        for (unsigned int v = 0; v < view_matrices.size(); v++) {
            // change camera location
            s_cam2.SetModelViewMatrix(view_matrices[v]);
            // Draw the scene to the framebuffer
            framebuffer.Bind();
            glViewport(0, 0, w, h);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            prog.Bind();
            prog.SetUniform("MVP", s_cam2.GetProjectionModelViewMatrix());
            prog.SetUniform("V", s_cam2.GetModelViewMatrix());
            pangolin::GlDraw(prog, gl_geom, nullptr);
            prog.Unbind();

            framebuffer.Unbind();

            // Map Resource
            cudaArray_t normals_tex_arr, verts_tex_arr;
            cudaSafeCall(cudaGraphicsMapResources(1, &cudaResourceNormals));
            cudaSafeCall(cudaGraphicsMapResources(1, &cudaResourceVertices));
            cudaSafeCall(cudaGraphicsSubResourceGetMappedArray(&normals_tex_arr, cudaResourceNormals, 0, 0));
            cudaSafeCall(cudaGraphicsSubResourceGetMappedArray(&verts_tex_arr, cudaResourceVertices, 0, 0));

            if (point_normals.size() - valid_point_num < wh) {
                point_normals.resize(valid_point_num + wh);
                point_verts.resize(valid_point_num + wh);
            }
            auto view_points = ValidPointsNormalCUDA(normals_tex_arr, verts_tex_arr, normals.width, normals.height, tri_pos, tri_all,
                                                     point_normals.data().get() + valid_point_num, point_verts.data().get() + valid_point_num);
            valid_point_num += view_points;

            // Unmap Resource
            cudaSafeCall(cudaGraphicsUnmapResources(1, &cudaResourceNormals));
            cudaSafeCall(cudaGraphicsUnmapResources(1, &cudaResourceVertices));
        }

        // Unregister normal and vertex buffer.
        cudaSafeCall(cudaGraphicsUnregisterResource(cudaResourceNormals));
        cudaSafeCall(cudaGraphicsUnregisterResource(cudaResourceVertices));

        point_normals.resize(valid_point_num);
        point_verts.resize(valid_point_num);
    } else if (reference_method == 2) {
        // Method2: Believe in mesh normal.
        // Do nothing here.
    } else {
        // Method3: Load reference points captured outside.
        thrust::host_vector<float4> cpu_point_verts;
        thrust::host_vector<float4> cpu_point_normals;
        {
            std::ifstream fin(referenceFileName, std::ios::in | std::ios::binary);
            int point_count;
            fin.read((char*)&point_count, sizeof(int));
//            std::cout << "Point Count: " << point_count << std::endl;
            cpu_point_verts.resize(point_count);
            cpu_point_normals.resize(point_count);
            fin.read((char*)cpu_point_verts.data(), sizeof(float4) * point_count);
            fin.read((char*)cpu_point_normals.data(), sizeof(float4) * point_count);
        }
        point_verts = cpu_point_verts;
        point_normals = cpu_point_normals;
    }

    int num_samp_near_surf = num_sample * num_samp_near_surf_ratio;
    std::cout << "num_samp_near_surf: " << num_samp_near_surf << std::endl;

    thrust::host_vector<float4> sampled_data;

    auto start = std::chrono::high_resolution_clock::now();
    GenerateSDFSamples(reference_method, gl_geom, num_samp_near_surf / 2, num_sample - num_samp_near_surf,
            point_verts, point_normals, variance, second_variance, max_ref_dist, 11, sampled_data, ub_x, ub_y, ub_z);
    auto finish = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(finish - start).count();
    std::cout << elapsed << std::endl;

    std::cout << "num points sampled: " << sampled_data.size() << std::endl;

    // Write raw file.
    {
        std::ofstream fout(outputFileName, std::ios::out | std::ios::binary);
        fout.write((char*) sampled_data.data(), sizeof(float4) * sampled_data.size());
        fout.close();
    }

    // Write surface file.
    if (!surfaceFileName.empty()) {
        std::ofstream fout(surfaceFileName, std::ios::out | std::ios::binary);
        thrust::host_vector<float4> cpu_point_verts = point_verts;
        thrust::host_vector<float4> cpu_point_normals = point_normals;
        // NOTE: Here sampling method1 is also downsampled.
        int increment = reference_method > 2 ? 1 : METHOD2_SAMPLES_MULT;

        for (int i = 0; i < cpu_point_normals.size(); i += increment) {
            fout.write((char*) (cpu_point_verts.data() + i), sizeof(float3));
            fout.write((char*) (cpu_point_normals.data() + i), sizeof(float3));
        }
        fout.close();
    }

    return 0;
}
