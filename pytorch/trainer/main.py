import importlib
import json
import logging
from pathlib import Path

import lr_schedule
import torch
import tqdm
import yaml
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from dataset.training import lif_dataset as ldata
from network import criterion
from utils import exp_util


class TensorboardViz(object):

    def __init__(self, logdir):
        self.logdir = logdir
        self.writter = SummaryWriter(self.logdir)

    def text(self, _text):
        # Enhance line break and convert to code blocks
        _text = _text.replace('\n', '  \n\t')
        self.writter.add_text('Info', _text)

    def update(self, mode, it, eval_dict):
        self.writter.add_scalars(mode, eval_dict, global_step=it)

    def flush(self):
        self.writter.flush()


parser = exp_util.ArgumentParserX(add_hyper_arg=True)
parser.add_argument('-v', '--visualize', action='store_true', help='Visualize')


def main():
    logging.basicConfig(level=logging.INFO)
    args = parser.parse_args()
    logging.info(args)

    checkpoints = list(range(args.snapshot_frequency, args.num_epochs + 1, args.snapshot_frequency))
    for checkpoint in args.additional_snapshots:
        checkpoints.append(checkpoint)
    checkpoints.sort()

    lr_schedules = lr_schedule.get_learning_rate_schedules(args)

    net_module = importlib.import_module("network." + args.network_name)
    model = net_module.Model(args.code_length, **args.network_specs).cuda()
    model = torch.nn.DataParallel(model)

    encoder_module = importlib.import_module("network." + args.encoder_name)
    args.encoder_specs.update({"latent_size": args.code_length})
    encoder = encoder_module.Model(**args.encoder_specs, mode='train').cuda()
    encoder = torch.nn.DataParallel(encoder)

    lif_dataset = ldata.LifCombinedDataset(*[
        ldata.LifDataset(**t, num_sample=args.samples_per_lif) for t in args.train_set
    ])
    lif_loader = DataLoader(
        lif_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        drop_last=True,
    )

    loss_func_args = exp_util.dict_to_args(args.training_loss)
    loss_funcs = [
        getattr(criterion, t) for t in loss_func_args.types
    ]

    optimizer_all = torch.optim.Adam([
        { "params": model.parameters(), "lr": lr_schedules[0].get_learning_rate(0) },
        { "params": encoder.parameters(), "lr": lr_schedules[1].get_learning_rate(0) },
    ])

    save_base_dir = Path("../di-checkpoints/%s" % args.run_name)
    assert not save_base_dir.exists()
    save_base_dir.mkdir(parents=True, exist_ok=True)

    viz = TensorboardViz(logdir=str(save_base_dir / 'tensorboard'))
    viz.text(yaml.dump(vars(args)))
    with (save_base_dir / "hyper.json").open("w") as f:
        json.dump(vars(args), f, indent=2)

    start_epoch = 1

    logging.info("starting from epoch {}".format(start_epoch))
    logging.info(
        "Number of decoder parameters: {}".format(sum(p.data.nelement() for p in model.parameters()))
    )
    logging.info(
        "Number of encoder parameters: {}".format(sum(p.data.nelement() for p in encoder.parameters()))
    )

    epoch_bar = tqdm.trange(start_epoch, args.num_epochs + 1, desc='epochs')
    all_lat_vecs = torch.zeros((len(lif_dataset), args.code_length)).cuda()

    it = 0
    for epoch in epoch_bar:
        model.train()
        lr_schedule.adjust_learning_rate(lr_schedules, optimizer_all, epoch)
        train_meter = exp_util.AverageMeter()
        train_running_meter = exp_util.RunningAverageMeter(alpha=0.3)
        batch_bar = tqdm.tqdm(total=len(lif_loader), leave=False, desc='train')

        for sdf_data, surface_data, idx in lif_loader:

            # if args.visualize:
            #     print(sdf_data.size(), surface_data.size())
            #     # Visualize training data.
            #     from pycg import vis
            #     vis_xyz = sdf_data[0, :, :3].numpy()
            #     vis_sdf = sdf_data[0, :, 3:].numpy()
            #     if vis_sdf.shape[1] == 1:
            #         input_pcd = vis.pointcloud(vis_xyz, cfloat=vis_sdf[:, 0])
            #     else:
            #         input_pcd = vis.pointcloud(vis_xyz, normal=vis_sdf)
            #     vis.show_3d([input_pcd,
            #                   vis.pointcloud(surface_data[0, :, :3].numpy(), normal=surface_data[0, :, 3:6].numpy(), is_sphere=True),
            #                   vis.pointcloud(surface_data[0, :, :3].numpy(), normal=surface_data[0, :, 3:6].numpy()),
            #                   vis.wireframe_bbox([-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]),
            #                   vis.wireframe_bbox([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0])])

            # Process the input data
            sdf_data = sdf_data.reshape(-1, sdf_data.size(-1)).cuda()
            surface_data = surface_data.cuda()      # (B, N, 6)
            idx = idx.cuda()                        # (B, )

            num_sdf_samples = sdf_data.shape[0]
            xyz = sdf_data[:, 0:3]
            sdf_gt = sdf_data[:, 3:]

            xyz = torch.chunk(xyz, args.batch_split)
            sdf_gt = torch.chunk(sdf_gt, args.batch_split)

            batch_loss = exp_util.CombinedChunkLoss()
            optimizer_all.zero_grad()

            lat_vecs = encoder(surface_data)    # (B, L)
            all_lat_vecs[idx] = lat_vecs        # Just for recording, no other usages.
            lat_vecs = lat_vecs.unsqueeze(1).repeat(1, args.samples_per_lif, 1).view(-1, lat_vecs.size(-1))   # (BxS, L)
            lat_vecs = torch.chunk(lat_vecs, args.batch_split)

            for i in range(args.batch_split):
                xyz[i].requires_grad_(True)
                net_input = torch.cat([lat_vecs[i], xyz[i]], dim=1)
                pred_sdf, pred_sdf_std = model(net_input)

                for loss_func in loss_funcs:
                    batch_loss.update_loss_dict(loss_func(
                        args=loss_func_args, pd_sdf=pred_sdf, pd_sdf_std=pred_sdf_std, gt_sdf=sdf_gt[i],
                        latent_vecs=lat_vecs[i], coords=xyz[i],
                        info={"num_sdf_samples": num_sdf_samples, "epoch": epoch}
                    ))

                xyz[i].requires_grad_(False)
                batch_loss.get_total_loss().backward()

            # if args.gradient_clip_norm is not None:
            #     torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip_norm)
            optimizer_all.step()

            loss_res = batch_loss.get_accumulated_loss_dict()
            del batch_loss

            # Let's directly forward surface xyz to test whether output is close to 0.
            # This is of course not fair, but can give us a rough impression of which is good/bad.
            with torch.no_grad():
                surf_lat_vecs = all_lat_vecs[idx].unsqueeze(1).repeat(
                    1, surface_data.size(1), 1).view(-1, all_lat_vecs[idx].size(-1))
                surf_xyz = surface_data[..., :3].reshape(-1, 3)
                net_input = torch.cat([surf_lat_vecs, surf_xyz], dim=1)
                surf_sdf, _ = model(net_input)
                surf_sdf_error = surf_sdf.abs().mean().item()
                loss_res["validation"] = surf_sdf_error

            batch_bar.update()
            train_running_meter.append_loss(loss_res)
            batch_bar.set_postfix(train_running_meter.get_loss_dict())
            epoch_bar.refresh()
            it += 1

            if it % 10 == 0:
                for loss_name, loss_val in loss_res.items():
                    viz.update('train/' + loss_name, it, {'scalar': loss_val})
            train_meter.append_loss(loss_res)

        batch_bar.close()

        # At the end of each epoch.
        train_avg = train_meter.get_mean_loss_dict()
        for meter_key, meter_val in train_avg.items():
            viz.update("epoch_sum/" + meter_key, epoch, {'train': meter_val})
        for sid, schedule in enumerate(lr_schedules):
            viz.update(f"train_stat/lr_{sid}", epoch, {'scalar': schedule.get_learning_rate(epoch)})

        mean_latent_norm = torch.mean(torch.norm(all_lat_vecs.detach(), dim=1))
        viz.update("train_stat/latent_norm", epoch, {'scalar': mean_latent_norm.item()})

        if epoch in checkpoints:
            torch.save({
                "epoch": epoch,
                "model_state": model.module.state_dict(),
            }, save_base_dir / f"model_{epoch}.pth.tar")
            torch.save({
                "epoch": epoch,
                "optimizer_state": optimizer_all.state_dict(),
                "latent_vec": all_lat_vecs
            }, save_base_dir / f"training_{epoch}.pth.tar")
            torch.save({
                "epoch": epoch,
                "model_state": encoder.module.state_dict(),
            }, save_base_dir / f"encoder_{epoch}.pth.tar")


if __name__ == '__main__':
    main()
