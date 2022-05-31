import json
import jittor as jt
import logging
import shutil
from pathlib import Path

import lr_schedule
import tqdm
import yaml
from tensorboardX import SummaryWriter

import lif_dataset as ldata
import criterion
from utils import exp_util
from network import DIDecoder, DIEncoder


if jt.has_cuda:
    jt.flags.use_cuda = 1


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


def main():
    logging.basicConfig(level=logging.INFO)
    args = parser.parse_args()
    logging.info(args)

    checkpoints = list(range(args.snapshot_frequency, args.num_epochs + 1, args.snapshot_frequency))
    for checkpoint in args.additional_snapshots:
        checkpoints.append(checkpoint)
    checkpoints.sort()

    lr_schedules = lr_schedule.get_learning_rate_schedules(args)

    model = DIDecoder()
    encoder = DIEncoder()

    lif_loader = ldata.LifDataset(**args.train_set[0], num_sample=args.samples_per_lif, batch_size=args.batch_size)

    loss_func_args = exp_util.dict_to_args(args.training_loss)
    loss_funcs = [
        getattr(criterion, t) for t in loss_func_args.types
    ]

    optimizer_all = jt.nn.Adam(model.parameters() + encoder.parameters(), lr=lr_schedules[0].get_learning_rate(0))

    save_base_dir = Path("../di-checkpoints/%s" % args.run_name)
    shutil.rmtree(save_base_dir, ignore_errors=True)
    save_base_dir.mkdir(parents=True, exist_ok=True)

    viz = TensorboardViz(logdir=str(save_base_dir / 'tensorboard'))
    viz.text(yaml.dump(vars(args)))
    with (save_base_dir / "hyper.json").open("w") as f:
        json.dump(vars(args), f, indent=2)

    start_epoch = 1
    epoch_bar = tqdm.trange(start_epoch, args.num_epochs + 1, desc='epochs')

    it = 0
    for epoch in epoch_bar:
        model.train()
        encoder.train()
        lr_schedule.adjust_learning_rate(lr_schedules, optimizer_all, epoch)
        train_meter = exp_util.AverageMeter()
        train_running_meter = exp_util.RunningAverageMeter(alpha=0.3)
        batch_bar = tqdm.tqdm(total=len(lif_loader), leave=False, desc='train')

        for sdf_data, surface_data, idx in lif_loader:
            # Process the input data
            sdf_data = sdf_data.reshape(-1, sdf_data.size(-1))
            surface_data = surface_data      # (B, N, 6)

            num_sdf_samples = sdf_data.shape[0]
            xyz = sdf_data[:, 0:3]
            sdf_gt = sdf_data[:, 3:]

            optimizer_all.zero_grad()

            lat_vecs = encoder(surface_data)    # (B, L)
            lat_vecs = lat_vecs.unsqueeze(1).repeat(1, args.samples_per_lif, 1).view(-1, lat_vecs.size(-1))   # (BxS, L)

            net_input = jt.concat([lat_vecs, xyz], dim=1)
            pred_sdf, pred_sdf_std = model(net_input)

            loss_dict = {}
            for loss_func in loss_funcs:
                loss_dict.update(loss_func(
                    args=loss_func_args, pd_sdf=pred_sdf, pd_sdf_std=pred_sdf_std, gt_sdf=sdf_gt,
                    latent_vecs=lat_vecs, coords=xyz,
                    info={"num_sdf_samples": num_sdf_samples, "epoch": epoch}
                ))
            loss_sum = sum(loss_dict.values())
            loss_res = {"value": loss_sum.item()}
            optimizer_all.step(loss_sum)

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

        train_avg = train_meter.get_mean_loss_dict()
        for meter_key, meter_val in train_avg.items():
            viz.update("epoch_sum/" + meter_key, epoch, {'train': meter_val})
        for sid, schedule in enumerate(lr_schedules):
            viz.update(f"train_stat/lr_{sid}", epoch, {'scalar': schedule.get_learning_rate(epoch)})

        if epoch in checkpoints:
            model.save(str(save_base_dir / f"model_{epoch}.jt.tar"))
            encoder.save(str(save_base_dir / f"encoder_{epoch}.jt.tar"))


if __name__ == '__main__':
    main()
