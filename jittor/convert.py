import torch
import pickle
from pathlib import Path

ENC_MAPPING = {
    "layer0.conv": 0,
    "layer0.normlayer.bn": 1,
    "layer1.conv": 3,
    "layer1.normlayer.bn": 4,
    "layer2.conv": 6,
    "layer2.normlayer.bn": 7,
    "layer3.conv": 9
}

enc_jittor_path = Path("../di-checkpoints/default/encoder_300.jt.tar")
dec_jittor_path = Path("../di-checkpoints/default/model_300.jt.tar")

with enc_jittor_path.open("rb") as f:
    enc_jt_weight = pickle.load(f)
pth_dict = {}
for wkey in list(enc_jt_weight.keys()):
    wnew_key = None
    for mkey in ENC_MAPPING.keys():
        if str(ENC_MAPPING[mkey]) in wkey:
            wnew_key = wkey.replace(str(ENC_MAPPING[mkey]), mkey)
    pth_dict[wnew_key] = torch.from_numpy(enc_jt_weight[wkey]).cuda()
torch.save({"epoch": 300, "model_state": pth_dict}, "./encoder_300.pth.tar")

with dec_jittor_path.open("rb") as f:
    dec_jt_weight = pickle.load(f)
for wkey in list(dec_jt_weight.keys()):
    dec_jt_weight[wkey] = torch.from_numpy(dec_jt_weight[wkey]).cuda()
torch.save({"epoch": 300, "model_state": dec_jt_weight}, "./model_300.pth.tar")
