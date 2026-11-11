"""离线一致性检查：拿训练集里的样本做 teacher-forcing，比较预测动作与专家动作。"""
import json
import os

import numpy as np
import torch

from mini_vla import MiniVLA, Normalizer, encode_text

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, "data")
CK = os.path.join(HERE, "checkpoints", "mini_vla_v2.pt")

ck = torch.load(CK, map_location="cuda", weights_only=False)
HIST = ck["hist"]
model = MiniVLA(vocab_size=len(ck["vocab"]), chunk=ck["chunk"], act_dim=8,
                state_dim=HIST * 8 + 1, img=ck["img"]).to("cuda").eval()
model.load_state_dict(ck["model"])
s_norm = Normalizer(ck["state_mean"], ck["state_std"])
a_norm = Normalizer(ck["act_mean"], ck["act_std"])
mean_len = ck["mean_len"]

for task in ("bin", "plate", "marker"):
    imgs = np.load(os.path.join(DATA, f"{task}_images.npy"), mmap_mode="r")
    st = np.load(os.path.join(DATA, f"{task}_states.npy"))
    acts = np.load(os.path.join(DATA, f"{task}_actions_abs.npy"))
    with open(os.path.join(DATA, f"{task}_meta.json"), encoding="utf-8") as f:
        instr = json.load(f)["instructions"][0]

    errs = []
    for t in (30, 80, 120, 160):
        vec = np.concatenate([st[max(t - (HIST - 1 - h), 0)] for h in range(HIST)]
                             + [[min(t / mean_len, 1.3)]]).astype(np.float32)
        s = torch.from_numpy(s_norm.norm(vec)[None]).cuda()
        im = np.transpose(imgs[t].astype(np.float32) / 255.0, (0, 3, 1, 2))
        im = torch.from_numpy((im - 0.5) / 0.5)[None].cuda()
        txt = torch.from_numpy(encode_text([instr], ck["vocab"], ck["max_len"])).cuda()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            mem = model.encode(im, s, txt)
        pred = a_norm.denorm(model.sample(mem, steps=8)[0].float().cpu().numpy())
        gt = acts[t:t + ck["chunk"]]
        err = np.abs(pred[:len(gt)] - gt).max(0)[:7] * 1000
        errs.append((t, float(np.abs(pred[0, :7] - gt[0, :7]).max()), float(np.mean(err))))
        print(f"{task} t={t:3d} 预测-真值 最大关节偏差={errs[-1][1]*1000:.1f}mrad "
              f"chunk内平均={errs[-1][2]:.1f}mrad  gripper_pred={pred[0,7]:.2f} gt={gt[0,7]:.2f}",
              flush=True)
