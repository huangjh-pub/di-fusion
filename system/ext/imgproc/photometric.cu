#include "common.cuh"

__global__ static void gradient_xy_kernel(const IntensityAccessor intensity, GradientAccessor out_grad) {
    const uint v = blockIdx.x * blockDim.x + threadIdx.x;
    const uint u = blockIdx.y * blockDim.y + threadIdx.y;
    if (v > intensity.size(0) - 1 || u > intensity.size(1) - 1) { return; }
    if (v < 1 || v > intensity.size(0) - 2 || u < 1 || u > intensity.size(1) - 2) {
        out_grad[v][u][0] = out_grad[v][u][1] = NAN;
        return;
    }

    // Sobel morph.
    float u_d1 = intensity[v - 1][u + 1] - intensity[v - 1][u - 1];
    float u_d2 = intensity[v][u + 1] - intensity[v][u - 1];
    float u_d3 = intensity[v + 1][u + 1] - intensity[v + 1][u - 1];
    out_grad[v][u][0] = (u_d1 + 2 * u_d2 + u_d3) / 8.0f;

    float v_d1 = intensity[v + 1][u - 1] - intensity[v - 1][u - 1];
    float v_d2 = intensity[v + 1][u] - intensity[v - 1][u];
    float v_d3 = intensity[v + 1][u + 1] - intensity[v - 1][u + 1];
    out_grad[v][u][1] = (v_d1 + 2 * v_d2 + v_d3) / 8.0f;
}

__global__ static void evaluate_fJ(const IntensityAccessor prev_img, const DepthAccessor prev_depth,
        const IntensityAccessor cur_img, const DepthAccessor cur_depth,
        const GradientAccessor cur_dIdxy, const float min_grad_scale, const float max_depth_delta,
        matrix3 krkinv, float3 kt, float4 calib,
        IntensityAccessor f_val, GradientAccessor J_val, bool compute_J) {

    const uint v = blockIdx.x * blockDim.x + threadIdx.x;
    const uint u = blockIdx.y * blockDim.y + threadIdx.y;
    const uint img_h = prev_img.size(0);
    const uint img_w = prev_img.size(1);

    // The boundary will not be valid anyway.
    if (v > img_h - 1 || u > img_w - 1) { return; }

    f_val[v][u] = NAN;

    // Also prune if gradient is too small (which is useless for pose estimation)
    float dI_dx = cur_dIdxy[v][u][0], dI_dy = cur_dIdxy[v][u][1];
    float mTwo = (dI_dx * dI_dx) + (dI_dy * dI_dy);
    if (mTwo < min_grad_scale || isnan(mTwo)) {
        return;
    }

    float d1 = cur_depth[v][u];
    if (isnan(d1)) {
        return;
    }

    float warpped_d1 = d1 * (krkinv.r3.x * u + krkinv.r3.y * v + krkinv.r3.z) + kt.z;
    int u0 = __float2int_rn((d1 * (krkinv.r1.x * u + krkinv.r1.y * v + krkinv.r1.z) + kt.x) / warpped_d1);
    int v0 = __float2int_rn((d1 * (krkinv.r2.x * u + krkinv.r2.y * v + krkinv.r2.z) + kt.y) / warpped_d1);

    if (u0 >= 0 && u0 < img_w && v0 >= 0 && v0 < img_h) {
        float d0 = prev_depth[v0][u0];
        // Make sure this pair of obs is not outlier and is really observed.
        if (!isnan(d0) && abs(warpped_d1 - d0) <= max_depth_delta && d0 > 0.0) {
            // Compute function value.
            f_val[v][u] = cur_img[v][u] - prev_img[v0][u0];
            // Compute gradient w.r.t. xi.
            if (compute_J) {
                float3 G = make_float3(d0 * (u0 - calib.z) / calib.x, d0 * (v0 - calib.w) / calib.y, d0);
                float p0 = dI_dx * calib.x / G.z;
                float p1 = dI_dy * calib.y / G.z;
                float p2 = -(p0 * G.x + p1 * G.y) / G.z;
                J_val[v][u][0] = p0;
                J_val[v][u][1] = p1;
                J_val[v][u][2] = p2;
                J_val[v][u][3] = -G.z * p1 + G.y * p2;
                J_val[v][u][4] =  G.z * p0 - G.x * p2;
                J_val[v][u][5] = -G.y * p0 + G.x * p1;
            }
        }
    }
}

torch::Tensor gradient_xy(torch::Tensor cur_intensity) {
    CHECK_INPUT(cur_intensity);
    const uint img_h = cur_intensity.size(0);
    const uint img_w = cur_intensity.size(1);

    dim3 dimBlock = dim3(16, 16);
    dim3 dimGrid = dim3(div_up(img_h, dimBlock.x), div_up(img_w, dimBlock.y));

    torch::Tensor cur_dIdxy = torch::empty({img_h, img_w, 2}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    gradient_xy_kernel<<<dimGrid, dimBlock>>>(
            cur_intensity.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            cur_dIdxy.packed_accessor32<float, 3, torch::RestrictPtrTraits>()
    );
    return cur_dIdxy;
}

std::vector<torch::Tensor> rgb_odometry(
        torch::Tensor prev_intensity, torch::Tensor prev_depth,
        torch::Tensor cur_intensity, torch::Tensor cur_depth, torch::Tensor cur_dIdxy,
        const std::vector<float>& intr,
        const std::vector<float>& krkinv_data,
        const std::vector<float>& kt_data,
        float min_grad_scale, float max_depth_delta, bool compute_J) {

    CHECK_INPUT(prev_intensity); CHECK_INPUT(prev_depth);
    CHECK_INPUT(cur_intensity); CHECK_INPUT(cur_depth); CHECK_INPUT(cur_dIdxy);

    const uint img_h = cur_intensity.size(0);
    const uint img_w = cur_intensity.size(1);

    dim3 dimBlock = dim3(16, 16);
    dim3 dimGrid = dim3(div_up(img_h, dimBlock.x), div_up(img_w, dimBlock.y));

    matrix3 krkinv(krkinv_data);
    float3 kt{kt_data[0], kt_data[1], kt_data[2]};
    float4 calib{intr[0], intr[1], intr[2], intr[3]};

    torch::Tensor f_img = torch::empty({img_h, img_w}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    torch::Tensor J_img = torch::empty({0, 0, 6}, torch::dtype(torch::kFloat32).device(torch::kCUDA));;
    if (compute_J) {
        J_img = torch::empty({img_h, img_w, 6}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    }

    evaluate_fJ<<<dimGrid, dimBlock>>>(
            prev_intensity.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            prev_depth.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            cur_intensity.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            cur_depth.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            cur_dIdxy.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            min_grad_scale, max_depth_delta,
            krkinv, kt, calib,
            f_img.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            J_img.packed_accessor32<float, 3, torch::RestrictPtrTraits>(), compute_J);

    if (compute_J) {
        return {f_img, J_img};
    } else {
        return {f_img};
    }
}
