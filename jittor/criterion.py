import math
import jittor as jt


def normal_log_prob(loc, scale, value):
    var = (scale ** 2)
    log_scale = scale.log()
    return -((value - loc) ** 2) / (2 * var) - log_scale - math.log(math.sqrt(2 * math.pi))


def neg_log_likelihood(args, info: dict, pd_sdf, pd_sdf_std, gt_sdf, **kwargs):
    if args.enforce_minmax:
        gt_sdf = jt.clamp(gt_sdf, -args.clamping_distance, args.clamping_distance)
        pd_sdf = jt.clamp(pd_sdf, -args.clamping_distance, args.clamping_distance)

    pd_dist_loc = pd_sdf.squeeze(1)
    pd_dist_std = pd_sdf_std.squeeze(1)

    sdf_loss = -normal_log_prob(pd_dist_loc, pd_dist_std, gt_sdf.squeeze(1)).sum() / info["num_sdf_samples"]
    return {
        'll': sdf_loss
    }


def reg_loss(args, info: dict, latent_vecs, **kwargs):
    l2_size_loss = jt.sum(jt.norm(latent_vecs, k=2, dim=1))
    reg_loss = min(1, info["epoch"] / 100) * l2_size_loss / info["num_sdf_samples"]
    return {
        'reg': reg_loss * args.code_reg_lambda
    }
