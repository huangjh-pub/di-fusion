#include <torch/extension.h>

torch::Tensor remove_radius_outlier(
        torch::Tensor input_pc,
        int nb_points,
        float radius
);

torch::Tensor estimate_normals(
        torch::Tensor input_pc,
        int max_nn,
        float radius,
        const std::vector<float>& cam_xyz
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("remove_radius_outlier", &remove_radius_outlier, "Remove point outliers by radius (CUDA)");
    m.def("estimate_normals", &estimate_normals, "Estimate point cloud normals (CUDA)");
}
