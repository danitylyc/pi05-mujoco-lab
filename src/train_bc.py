"""在 MuJoCo 示教数据上训练 mini-VLA（条件流匹配行为克隆）。

观测设计（关键）：状态不是单帧，而是
    [过去 4 帧本体状态(4x8) , 归一化任务进度 t_norm]  -> 33 维
动作语义：绝对关节位置目标 + 夹爪指令（7+1），与 pi0.5 输出口径一致。
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from mini_vla import MiniVLA, Normalizer, build_vocab, encode_text

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, "data")
CKPT = os.path.join(HERE, "checkpoints")

HIST = 4


class DemoDataset(Dataset):
    def __init__(self, tasks, chunk=8):
        self.chunk = chunk
        self.tasks = tasks
        self.img = {t: np.load(os.path.join(DATA, f"{t}_images.npy"), mmap_mode="r") for t in tasks}
        self.st = {t: np.load(os.path.join(DATA, f"{t}_states.npy")) for t in tasks}
        self.ac = {t: np.load(os.path.join(DATA, f"{t}_actions_abs.npy")) for t in tasks}

        self.eps = []          # (task, start, length, instr)
        vocab_src = []
        for t in tasks:
            with open(os.path.join(DATA, f"{t}_meta.json"), "r", encoding="utf-8") as f:
                meta = json.load(f)
            instrs = meta["instructions"]
            vocab_src += instrs
            ep_len = np.load(os.path.join(DATA, f"{t}_ep_len.npy"))
            off = 0
            for k, L in enumerate(ep_len):
                self.eps.append((t, int(off), int(L), instrs[k % len(instrs)]))
                off += int(L)

        self.mean_len = float(np.mean([e[2] for e in self.eps]))
        self.vocab = build_vocab(vocab_src)
        self.max_len = max(len(s.split()) for s in vocab_src)
        self.samples = [(e, t) for e in range(len(self.eps)) for t in range(self.eps[e][2])]

        all_state = np.concatenate([self.st[t] for t in tasks], 0)
        all_act = np.concatenate([self.ac[t] for t in tasks], 0)
        s_mean, s_std = all_state.mean(0), all_state.std(0) + 1e-6
        self.state_norm = Normalizer(
            np.concatenate([np.tile(s_mean, HIST), [0.5]]),
            np.concatenate([np.tile(s_std, HIST), [0.29]]),
        )
        self.act_norm = Normalizer(all_act.mean(0), all_act.std(0) + 1e-6)
        print(f"mean_len={self.mean_len:.1f} act_std={np.round(all_act.std(0), 3)}", flush=True)

    def __len__(self):
        return len(self.samples)

    def state_vec(self, task, idx, t, L):
        st = self.st[task]
        hist = [st[max(idx - (HIST - 1 - h), 0)] for h in range(HIST)]
        t_norm = min(t / self.mean_len, 1.3)
        return self.state_norm.norm(np.concatenate(hist + [[t_norm]]).astype(np.float32))

    def __getitem__(self, i):
        e, t = self.samples[i]
        task, start, L, instr = self.eps[e]
        idx = start + t

        imgs = np.asarray(self.img[task][idx], dtype=np.float32) / 255.0   # (2,H,W,3)
        imgs = np.transpose(imgs, (0, 3, 1, 2))                            # (2,3,H,W)
        imgs = (imgs - 0.5) / 0.5
        state = self.state_vec(task, idx, t, L)

        a = self.ac[task][idx:idx + self.chunk]
        if len(a) < self.chunk:
            a = np.concatenate([a, np.repeat(a[-1:], self.chunk - len(a), 0)], 0)
        a = self.act_norm.norm(a).astype(np.float32)

        txt = encode_text([instr], self.vocab, self.max_len)[0]
        return (torch.from_numpy(imgs), torch.from_numpy(state.astype(np.float32)),
                torch.from_numpy(txt), torch.from_numpy(a))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--img", type=int, default=112)
    ap.add_argument("--tag", type=str, default="mini_vla")
    ap.add_argument("--max-steps", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(CKPT, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, torch.cuda.get_device_name(0) if device == "cuda" else "", flush=True)

    ds = DemoDataset(("bin", "plate", "marker"), chunk=args.chunk)
    print(f"episodes={len(ds.eps)} samples={len(ds)} vocab={len(ds.vocab)} "
          f"max_len={ds.max_len} img={ds.img['bin'].shape[-2:]}", flush=True)

    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=0, drop_last=True)

    model = MiniVLA(vocab_size=len(ds.vocab), chunk=args.chunk, act_dim=8,
                    state_dim=HIST * 8 + 1, img=args.img).to(device)
    print(f"params={sum(p.numel() for p in model.parameters())/1e6:.2f}M", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total = args.epochs * len(dl)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")

    t0, gstep = time.time(), 0
    for ep in range(args.epochs):
        run, n = 0.0, 0
        for imgs, state, txt, act in dl:
            imgs, state, txt, act = imgs.to(device), state.to(device), txt.to(device), act.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                memory = model.encode(imgs, state, txt)
                loss = model.flow_loss(memory, act)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            run += loss.item()
            n += 1
            gstep += 1
            if gstep % 200 == 0:
                print(f"  step {gstep} loss={run/n:.4f} {time.time()-t0:.0f}s "
                      f"gpu_peak={torch.cuda.max_memory_allocated()/2**30:.2f}GB", flush=True)
            if args.max_steps and gstep >= args.max_steps:
                break
        print(f"epoch {ep+1:2d}/{args.epochs} loss={run/max(n,1):.4f} "
              f"lr={sched.get_last_lr()[0]:.2e} elapsed={time.time()-t0:.0f}s", flush=True)
        if args.max_steps and gstep >= args.max_steps:
            break

    path = os.path.join(CKPT, f"{args.tag}.pt")
    torch.save({
        "model": model.state_dict(),
        "vocab": ds.vocab, "max_len": ds.max_len,
        "chunk": args.chunk, "img": args.img, "hist": HIST,
        "mean_len": ds.mean_len,
        "state_mean": ds.state_norm.mean, "state_std": ds.state_norm.std,
        "act_mean": ds.act_norm.mean, "act_std": ds.act_norm.std,
    }, path)
    print("saved", path, flush=True)


if __name__ == "__main__":
    main()
