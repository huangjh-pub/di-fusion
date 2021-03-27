from pathlib import Path
from torch.utils.cpp_extension import load


def p(rel_path):
    abs_path = Path(__file__).parent / rel_path
    return str(abs_path)


__COMPILE_VERBOSE = False

# Load in Marching cubes.
_marching_cubes_module = load(name='marching_cubes',
                              sources=[p('marching_cubes/mc.cpp'),
                                       p('marching_cubes/mc_interp_kernel.cu')],
                              verbose=__COMPILE_VERBOSE)
marching_cubes_interp = _marching_cubes_module.marching_cubes_sparse_interp

# Load in Image processing modules.
_imgproc_module = load(name='imgproc',
                       sources=[p('imgproc/imgproc.cu'), p('imgproc/imgproc.cpp'), p('imgproc/photometric.cu')],
                       verbose=__COMPILE_VERBOSE)
unproject_depth = _imgproc_module.unproject_depth
compute_normal_weight = _imgproc_module.compute_normal_weight
compute_normal_weight_robust = _imgproc_module.compute_normal_weight_robust
filter_depth = _imgproc_module.filter_depth
rgb_odometry = _imgproc_module.rgb_odometry
gradient_xy = _imgproc_module.gradient_xy

# Load in Indexing modules. (which deal with complicated indexing scheme)
_indexing_module = load(name='indexing',
                        sources=[p('indexing/indexing.cpp'), p('indexing/indexing.cu')],
                        verbose=__COMPILE_VERBOSE)
pack_batch = _indexing_module.pack_batch
groupby_sum = _indexing_module.groupby_sum

# We need point cloud processing module.
_pcproc_module = load(name='pcproc',
                      sources=[p('pcproc/pcproc.cpp'), p('pcproc/pcproc.cu'), p('pcproc/cuda_kdtree.cu')],
                      verbose=__COMPILE_VERBOSE)
remove_radius_outlier = _pcproc_module.remove_radius_outlier
estimate_normals = _pcproc_module.estimate_normals
