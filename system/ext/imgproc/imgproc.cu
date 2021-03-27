#include "common.cuh"

// -----------------------------------------------------------------------------------------------------------

__global__ static void unproject_depth_kernel(const DepthAccessor depth, PCMapAccessor pc_map,
        const float fx, const float fy, const float cx, const float cy, const uint img_h, const uint img_w) {

    const uint img_v = blockIdx.x * blockDim.x + threadIdx.x;
    const uint img_u = blockIdx.y * blockDim.y + threadIdx.y;
    if (img_v >= img_h || img_u >= img_w) {
        return;
    }

    float d = depth[img_v][img_u];

    if (!isnan(d)) {
        pc_map[img_v][img_u][0] = (img_u - cx) / fx * d;
        pc_map[img_v][img_u][1] = (img_v - cy) / fy * d;
        pc_map[img_v][img_u][2] = d;
    } else {
        pc_map[img_v][img_u][0] = NAN;
    }

}

torch::Tensor unproject_depth(torch::Tensor depth,
                              const float fx, const float fy, const float cx, const float cy) {
    CHECK_INPUT(depth);

    const uint img_h = depth.size(0);
    const uint img_w = depth.size(1);

    torch::Tensor pc_map = torch::empty({img_h, img_w, 3}, torch::dtype(torch::kFloat32).device(torch::kCUDA));

    dim3 dimBlock = dim3(16, 16);
    dim3 dimGrid = dim3(div_up(img_h, dimBlock.x), div_up(img_w, dimBlock.y));
    unproject_depth_kernel<<<dimGrid, dimBlock>>>(
            depth.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            pc_map.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            fx, fy, cx, cy, img_h, img_w
    );

    return pc_map;
}

// -----------------------------------------------------------------------------------------------------------

#define MEAN_SIGMA_L 1.2232f
__global__ static void filter_depth_kernel(const DepthAccessor depth_in, DepthAccessor depth_out,
        const uint img_h, const uint img_w) {
    const uint img_v = blockIdx.x * blockDim.x + threadIdx.x;
    const uint img_u = blockIdx.y * blockDim.y + threadIdx.y;
    if (img_v < 2 || img_v >= img_h - 2 || img_u < 2 || img_u >= img_w - 2) {
        return;
    }

    float w_sum = 0.0f;
    float final_depth = 0.0f;
    float z = depth_in[img_v][img_u];
    if (z < 1e-6) {
        depth_out[img_v][img_u] = 0.0f; return;
    }
    float sigma_z = 1.0f / (0.0012f + 0.0019f*(z - 0.4f)*(z - 0.4f) + 0.0001f / sqrt(z) * 0.25f);

#pragma unroll
    for (int i = -2; i <= 2; ++i) {
        for (int j = -2; j <= 2; ++j) {
            float nn_z = depth_in[img_v + i][img_u + j];
            if (nn_z < 1e-6) continue;
            float dz = (nn_z - z) * (nn_z - z);
            float w = exp(-0.5f * ((abs(i) + abs(j)) * MEAN_SIGMA_L*MEAN_SIGMA_L + dz * sigma_z * sigma_z));
            w_sum += w;
            final_depth += w * nn_z;
        }
    }

    final_depth /= w_sum;
    depth_out[img_v][img_u] = final_depth;
}

void filter_depth(torch::Tensor depth_in, torch::Tensor depth_out) {
    CHECK_INPUT(depth_in);
    CHECK_INPUT(depth_out);
    const uint img_h = depth_in.size(0);
    const uint img_w = depth_in.size(1);

    dim3 dimBlock = dim3(16, 16);
    dim3 dimGrid = dim3(div_up(img_h, dimBlock.x), div_up(img_w, dimBlock.y));
    filter_depth_kernel<<<dimGrid, dimBlock>>>(
            depth_in.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            depth_out.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            img_h, img_w
    );
}

// -----------------------------------------------------------------------------------------------------------

__global__ static void compute_normal_weight_kernel(const PCMapAccessor pc_map, PCMapAccessor normal_weight,
                                                    const uint img_h, const uint img_w) {
    const uint img_v = blockIdx.x * blockDim.x + threadIdx.x;
    const uint img_u = blockIdx.y * blockDim.y + threadIdx.y;
    if (img_v < 1 || img_v > img_h - 2 || img_u < 1 || img_u > img_w - 2) {
        normal_weight[img_v][img_u][3] = -1.0f;
        return;
    }

    float3 cur_pos = get_vec3(pc_map, img_v, img_u);
    if (cur_pos.z <= 1e-6) {
        normal_weight[img_v][img_u][3] = -1.0f;
        return;
    }

    // Compute Normal
    float3 xp1_y = get_vec3(pc_map, img_v, img_u + 1);
    float3 xm1_y = get_vec3(pc_map, img_v, img_u - 1);
    float3 x_yp1 = get_vec3(pc_map, img_v + 1, img_u);
    float3 x_ym1 = get_vec3(pc_map, img_v - 1, img_u);
    if (xp1_y.z < 1e-6 || xm1_y.z < 1e-6 || x_yp1.z < 1e-6 || x_ym1.z < 1e-6) {
        normal_weight[img_v][img_u][3] = -1.0f;
        return;
    }

    float3 diff_x = xp1_y - xm1_y;
    float3 diff_y = x_yp1 - x_ym1;
    float3 normal = cross(diff_y, diff_x);
    float normal_length = length(normal);
    if (normal_length < 1e-6) {
        normal_weight[img_v][img_u][3] = -1.0f;
        return;
    }
    normal /= normal_length;
    float theta = acos(normal.z);
    float theta_diff = theta / (0.5f * 3.14159f - theta);
    float weight = (0.0012f + 0.0019f * (cur_pos.z - 0.4f) * (cur_pos.z - 0.4f) +
                    0.0001f / sqrt(cur_pos.z) * theta_diff * theta_diff);

    normal_weight[img_v][img_u][0] = normal.x;
    normal_weight[img_v][img_u][1] = normal.y;
    normal_weight[img_v][img_u][2] = normal.z;
    normal_weight[img_v][img_u][3] = 1.0f / weight;
}

torch::Tensor compute_normal_weight(torch::Tensor pc_map) {
    CHECK_INPUT(pc_map);

    const uint img_h = pc_map.size(0);
    const uint img_w = pc_map.size(1);

    torch::Tensor normal_weight = torch::empty({img_h, img_w, 4}, torch::dtype(torch::kFloat32).device(torch::kCUDA));

    dim3 dimBlock = dim3(16, 16);
    dim3 dimGrid = dim3(div_up(img_h, dimBlock.x), div_up(img_w, dimBlock.y));
    compute_normal_weight_kernel<<<dimGrid, dimBlock>>>(
            pc_map.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            normal_weight.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            img_h, img_w
    );

    return normal_weight;
}

// -----------------------------------------------------------------------------------------------------------

__device__ float4 sym3eig(float3 x1, float3 x2, float3 x3) {
    float4 ret_val;

    const float p1 = x1.y * x1.y + x1.z * x1.z + x2.z * x2.z;
    const float q = (x1.x + x2.y + x3.z) / 3.0f;
    const float p2 = (x1.x - q) * (x1.x - q) + (x2.y - q) * (x2.y - q) + (x3.z - q) * (x3.z - q) + 2 * p1;
    const float p = sqrt(p2 / 6.0f);
    const float b11 = (1.0f / p) * (x1.x - q);
    const float b12 = (1.0f / p) * x1.y;
    const float b13 = (1.0f / p) * x1.z;
    const float b21 = (1.0f / p) * x2.x;
    const float b22 = (1.0f / p) * (x2.y - q);
    const float b23 = (1.0f / p) * x2.z;
    const float b31 = (1.0f / p) * x3.x;
    const float b32 = (1.0f / p) * x3.y;
    const float b33 = (1.0f / p) * (x3.z - q);

    float r = b11 * b22 * b33 + b12 * b23 * b31 + b13 * b21 * b32 -
              b13 * b22 * b31 - b12 * b21 * b33 - b11 * b23 * b32;
    r = r / 2.0f;

    float phi;
    if (r <= -1) {
        phi = M_PI / 3.0f;
    } else if (r >= 1) {
        phi = 0;
    } else {
        phi = acos(r) / 3.0f;
    }

//    float e0 = q + 2 * p * cos(phi);
    ret_val.w = q + 2 * p * cos(phi + (2 * M_PI / 3));
//    ret_val.w = 3 * q - e0 - e1;

    x1.x -= ret_val.w;
    x2.y -= ret_val.w;
    x3.z -= ret_val.w;

    const float r12_1 = x1.y * x2.z - x1.z * x2.y;
    const float r12_2 = x1.z * x2.x - x1.x * x2.z;
    const float r12_3 = x1.x * x2.y - x1.y * x2.x;
    const float r13_1 = x1.y * x3.z - x1.z * x3.y;
    const float r13_2 = x1.z * x3.x - x1.x * x3.z;
    const float r13_3 = x1.x * x3.y - x1.y * x3.x;
    const float r23_1 = x2.y * x3.z - x2.z * x3.y;
    const float r23_2 = x2.z * x3.x - x2.x * x3.z;
    const float r23_3 = x2.x * x3.y - x2.y * x3.x;

    const float d1 = r12_1 * r12_1 + r12_2 * r12_2 + r12_3 * r12_3;
    const float d2 = r13_1 * r13_1 + r13_2 * r13_2 + r13_3 * r13_3;
    const float d3 = r23_1 * r23_1 + r23_2 * r23_2 + r23_3 * r23_3;

    float d_max = d1;
    int i_max = 0;

    if (d2 > d_max) {
        d_max = d2;
        i_max = 1;
    }

    if (d3 > d_max) {
        i_max = 2;
    }

    if (i_max == 0) {
        ret_val.x = r12_1 / sqrt(d1);
        ret_val.y = r12_2 / sqrt(d1);
        ret_val.z = r12_3 / sqrt(d1);
    } else if (i_max == 1) {
        ret_val.x = r13_1 / sqrt(d2);
        ret_val.y = r13_2 / sqrt(d2);
        ret_val.z = r13_3 / sqrt(d2);
    } else {
        ret_val.x = r23_1 / sqrt(d3);
        ret_val.y = r23_2 / sqrt(d3);
        ret_val.z = r23_3 / sqrt(d3);
    }

    return ret_val;
}

#define ROBUST_RADIUS 3
__global__ static void compute_normal_weight_robust_kernel(const PCMapAccessor pc_map, PCMapAccessor normal_weight,
                                                           const uint img_h, const uint img_w) {
    const uint img_v = blockIdx.x * blockDim.x + threadIdx.x;
    const uint img_u = blockIdx.y * blockDim.y + threadIdx.y;
    if (img_v < ROBUST_RADIUS || img_v > img_h - 1 - ROBUST_RADIUS ||
        img_u < ROBUST_RADIUS || img_u > img_w - 1 - ROBUST_RADIUS) {
        normal_weight[img_v][img_u][3] = -1.0f;
        return;
    }

    float3 cur_pos = get_vec3(pc_map, img_v, img_u);
    if (cur_pos.z <= 1e-6) {
        normal_weight[img_v][img_u][3] = -1.0f;
        return;
    }

    // Compute mean pc position.
    float3 pc_mean{0.0f, 0.0f, 0.0f};
    float valid_count = 0.0f;
    for (int du = -ROBUST_RADIUS; du <= ROBUST_RADIUS; ++du) {
        for (int dv = -ROBUST_RADIUS; dv <= ROBUST_RADIUS; ++dv) {
            float3 pos = get_vec3(pc_map, img_v + dv, img_u + du);
            float pdist = squared_length(pos - cur_pos);
            if (pos.z > 1e-6 && pdist < 0.01) {
                pc_mean += pos;
                valid_count += 1.0f;
            }
        }
    }
    if (valid_count < 9.0f) {
        normal_weight[img_v][img_u][3] = -1.0f;
        return;
    }
    pc_mean /= valid_count;

    // Compute the covariance matrix.
    float3 cov_x1{0.0f, 0.0f, 0.0f};
    float3 cov_x2{0.0f, 0.0f, 0.0f};
    float3 cov_x3{0.0f, 0.0f, 0.0f};
    for (int du = -ROBUST_RADIUS; du <= ROBUST_RADIUS; ++du) {
        for (int dv = -ROBUST_RADIUS; dv <= ROBUST_RADIUS; ++dv) {
            float3 pos = get_vec3(pc_map, img_v + dv, img_u + du);
            float pdist = squared_length(pos - cur_pos);
            if (pos.z > 1e-6 && pdist < 0.01) {
                pos -= pc_mean;
                cov_x1.x += pos.x * pos.x; cov_x1.y += pos.x * pos.y; cov_x1.z += pos.x * pos.z;
                cov_x2.x += pos.y * pos.x; cov_x2.y += pos.y * pos.y; cov_x2.z += pos.y * pos.z;
                cov_x3.x += pos.z * pos.x; cov_x3.y += pos.z * pos.y; cov_x3.z += pos.z * pos.z;
            }
        }
    }
    float4 eigv = sym3eig(cov_x1, cov_x2, cov_x3);
    float3 normal = make_float3(eigv.x, eigv.y, eigv.z);
    if (dot(normal, cur_pos) > 0.0f) {
        normal = make_float3(-normal.x, -normal.y, -normal.z);
    }

    float theta = acos(normal.z);
    float theta_diff = theta / (0.5f * 3.14159f - theta);
    float weight = (0.0012f + 0.0019f * (cur_pos.z - 0.4f) * (cur_pos.z - 0.4f) +
                    0.0001f / sqrt(cur_pos.z) * theta_diff * theta_diff);

    normal_weight[img_v][img_u][0] = normal.x;
    normal_weight[img_v][img_u][1] = normal.y;
    normal_weight[img_v][img_u][2] = normal.z;
    normal_weight[img_v][img_u][3] = 1.0f / weight;
}

torch::Tensor compute_normal_weight_robust(torch::Tensor pc_map) {
    CHECK_INPUT(pc_map);

    const uint img_h = pc_map.size(0);
    const uint img_w = pc_map.size(1);

    torch::Tensor normal_weight = torch::empty({img_h, img_w, 4}, torch::dtype(torch::kFloat32).device(torch::kCUDA));

    dim3 dimBlock = dim3(16, 16);
    dim3 dimGrid = dim3(div_up(img_h, dimBlock.x), div_up(img_w, dimBlock.y));
    compute_normal_weight_robust_kernel<<<dimGrid, dimBlock>>>(
            pc_map.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            normal_weight.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            img_h, img_w
    );

    return normal_weight;
}
