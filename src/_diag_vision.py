"""检验策略是否真的用了视觉：同一时刻 t、不同 episode（方块初始位置不同），
看预测动作是否随 episode 变化，并与各自真值比对。"""
import json
import os

import numpy as np
import torch

from mini_vla import MiniVLA, Normalizer, encode_text

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, "data")

ck = torch.load(os.path.join(HERE, "checkpoints", "mini_vla_v2.pt"), map_location="cuda",
                weights_only=False)
HIST = ck["hist"]
model = MiniVLA(vocab_size=len(ck["vocab"]), chunk=ck["chunk"], act_dim=8,
                state_dim=HIST * 8 + 1, img=ck["img"]).to("cuda").eval()
model.load_state_dict(ck["model"])
s_norm = Normalizer(ck["state_mean"], ck["state_std"])
a_norm = Normalizer(ck["act_mean"], ck["act_std"])
mean_len = ck["mean_len"]

task = "bin"
imgs = np.load(os.path.join(DATA, f"{task}_images.npy"), mmap_mode="r")
st = np.load(os.path.join(DATA, f"{task}_states.npy"))
acts = np.load(os.path.join(DATA, f"{task}_actions_abs.npy"))
ep_len = np.load(os.path.join(DATA, f"{task}_ep_len.npy"))
with open(os.path.join(DATA, f"{task}_meta.json"), encoding="utf-8") as f:
    instr = json.load(f)["instructions"][0]

T = 34   # 下降阶段：动作强依赖方块位置
offs, gts, preds = [], [], []
off = 0
for ep in range(10):
    idx = off + T
    vec = np.concatenate([st[max(idx - (HIST - 1 - h), 0)] for h in range(HIST)]
                         + [[min(T / mean_len, 1.3)]]).astype(np.float32)
    s = torch.from_numpy(s_norm.norm(vec)[None]).cuda()
    im = np.transpose(imgs[idx].astype(np.float32) / 255.0, (0, 3, 1, 2))
    im = torch.from_numpy((im - 0.5) / 0.5)[None].cuda()
    txt = torch.from_numpy(encode_text([instr], ck["vocab"], ck["max_len"])).cuda()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        mem = model.encode(im, s, txt)
    p = a_norm.denorm(model.sample(mem, steps=8)[0].float().cpu().numpy())[0, :7]
    preds.append(p)
    gts.append(acts[idx, :7])
    offs.append(st[idx, :2].copy())
    off += int(ep_len[ep])

preds = np.array(preds)
gts = np.array(gts)
print("各 episode 起点关节(bin 前两个关节):", np.round(np.array(offs), 3), flush=True)
print("预测 关节2 (qpos[1]) 跨 episode:", np.round(preds[:, 1], 3), flush=True)
print("真值 关节2 (qpos[1]) 跨 episode:", np.round(gts[:, 1], 3), flush=True)
print("预测跨episode标准差:", np.round(preds.std(0), 4), flush=True)
print("|预测-真值| 每episode最大值(mrad):", np.round(np.abs(preds - gts).max(1) * 1000, 1), flush=True)
