#include <torch/extension.h>

torch::Tensor unproject_depth(torch::Tensor depth, float fx, float fy, float cx, float cy);
//torch::Tensor unproject_depth(torch::Tensor depth, float fx, float fy, float cx, float cy);

// We might do this several times, this interface enables re-use memory.
void filter_depth(torch::Tensor depth_in, torch::Tensor depth_out);
torch::Tensor compute_normal_weight(torch::Tensor pc_map);
torch::Tensor compute_normal_weight_robust(torch::Tensor pc_map);

// Compute rgb-image based odometry. Will return per-correspondence residual (M, ) and jacobian (M, 6) w.r.t. lie algebra.
// ... based on current given estimate ( T(xi) * prev_xyz = cur_xyz ).
// prev_intensity (H, W), float32, raning from 0 to 1.
std::vector<torch::Tensor> rgb_odometry(
        torch::Tensor prev_intensity, torch::Tensor prev_depth,
        torch::Tensor cur_intensity, torch::Tensor cur_depth, torch::Tensor cur_dIdxy,
        const std::vector<float>& intr,
        const std::vector<float>& krkinv_data,
        const std::vector<float>& kt_data,
        float min_grad_scale, float max_depth_delta, bool compute_J);
torch::Tensor gradient_xy(torch::Tensor cur_intensity);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("unproject_depth", &unproject_depth, "Unproject Depth (CUDA)");
    m.def("filter_depth", &filter_depth, "Filter Depth (CUDA)");
    m.def("compute_normal_weight", &compute_normal_weight, "Compute normal and weight (CUDA)");
    m.def("compute_normal_weight_robust", &compute_normal_weight_robust, "Compute normal and weight (CUDA)");
    m.def("rgb_odometry", &rgb_odometry, "Compute the function value and gradient for RGB Odometry (CUDA)");
    m.def("gradient_xy", &gradient_xy, "Compute Gradient of an image for jacobian computation. (CUDA)");
}
