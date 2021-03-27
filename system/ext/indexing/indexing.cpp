#include <torch/extension.h>

torch::Tensor pack_batch(torch::Tensor indices, uint n_batch, uint n_point);
std::vector<torch::Tensor> groupby_sum(torch::Tensor values, torch::Tensor indices, uint C);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("pack_batch", &pack_batch, "Pack Batch (CUDA)");
    m.def("groupby_sum", &groupby_sum, "GroupBy Sum (CUDA)");
}
