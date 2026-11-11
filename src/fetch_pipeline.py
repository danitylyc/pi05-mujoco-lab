"""主流 benchmark 全链路：FetchPickAndPlace-v4 上采示教 -> 训练 mini-VLA -> 闭环评测。

和自建 LIBERO 风格环境的差别只有三处，正好检验模型骨架的通用性：
  · 动作 4 维（TCP 增量 dx,dy,dz + 夹爪），不是 7 关节绝对位置
  · 观测 25 维（含物体位置，benchmark 原生给出）
  · 单相机 112px（480 渲染后降采样）
"""
import os

os.environ.setdefault("MUJOCO_GL", "glfw")

import json
import time

import gymnasium as gym
import gymnasium_robotics  # noqa: F401
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from mini_vla import MiniVLA, Normalizer, encode_text

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(HERE, "checkpoints")
OUT = os.path.join(HERE, "outputs")
HIST = 4
IMG = 112
CHUNK = 8
ACT = 4
INSTR = "pick up the cube and move it to the target"
MAX_STEPS = 100


def state8(obs):
    o = np.asarray(obs["observation"], dtype=np.float32)
    tcp = o[:3]
    grip = o[-1] if o.shape[0] >= 25 else 0.0
    rel = np.asarray(obs["achieved_goal"], dtype=np.float32) - tcp
    return np.concatenate([tcp, [grip], rel, [0.0]]).astype(np.float32)


def img112(env):
    f = np.asarray(env.render())
    idx = np.linspace(0, f.shape[0] - 1, IMG).astype(int)
    return f[idx][:, idx, :3].astype(np.uint8)


def phase_targets(cube, goal):
    above = np.array([cube[0], cube[1], cube[2] + 0.10])
    at = np.array([cube[0], cube[1], cube[2] + 0.005])
    lift = np.array([cube[0], cube[1], cube[2] + 0.12])
    over = np.array([goal[0], goal[1], goal[2] + 0.12])
    down = np.array([goal[0], goal[1], goal[2] + 0.01])
    return [(above, 12, +1.0), (at, 10, +1.0), (at, 8, -1.0), (lift, 10, -1.0),
            (over, 16, -1.0), (down, 10, -1.0), (down, 6, +1.0), (lift, 6, +1.0)]


def expert_episode(env, seed, record=False):
    obs, info = env.reset(seed=seed)
    cube = np.asarray(obs["achieved_goal"], dtype=np.float64)
    goal = np.asarray(obs["desired_goal"], dtype=np.float64)
    imgs, states, acts = [], [], []
    for target, steps, grip in phase_targets(cube, goal):
        for _ in range(steps):
            tcp = np.asarray(obs["observation"][:3], dtype=np.float64)
            d = np.clip(6.0 * (target - tcp), -1.0, 1.0)
            a = np.concatenate([d, [grip]]).astype(np.float32)
            if record:
                imgs.append(img112(env))
                states.append(state8(obs))
                acts.append(a)
            obs, r, done, trunc, info = env.step(a)
            if info.get("is_success", False) or done or trunc:
                return True, imgs, states, acts
    return bool(info.get("is_success", False)), imgs, states, acts


class FetchDS(Dataset):
    def __init__(self, imgs, states, acts, ep_start, ep_len, vocab, mean_len):
        self.imgs, self.states, self.acts = imgs, states, acts
        self.ep_start, self.ep_len = ep_start, ep_len
        self.vocab, self.mean_len = vocab, mean_len
        self.idx = []
        for s, L in zip(ep_start, ep_len):
            for t in range(L - CHUNK):
                self.idx.append((s + t, s, t, L))
        self.state_norm = Normalizer(states.mean(0), states.std(0) + 1e-6)
        self.act_norm = Normalizer(acts.mean(0), acts.std(0) + 1e-6)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j, s, t, L = self.idx[i]
        hist = np.concatenate([self.states[max(j - (HIST - 1 - h), s)] for h in range(HIST)]
                              + [[min(t / self.mean_len, 1.3)]]).astype(np.float32)
        st = self.state_norm.norm(hist)
        img = np.transpose(self.imgs[j].astype(np.float32) / 255.0, (2, 0, 1))
        img = (img - 0.5) / 0.5
        act = self.act_norm.norm(self.acts[j:j + CHUNK])
        txt = encode_text([INSTR], self.vocab, 8)[0]
        return (torch.from_numpy(img), torch.from_numpy(st), torch.from_numpy(txt),
                torch.from_numpy(act.astype(np.float32)))


def main(n_demos=200, epochs=10, batch=32):
    vocab = {"<pad>": 0}
    for w in sorted(set(INSTR.lower().split())):
        vocab[w] = len(vocab)

    env = gym.make("FetchPickAndPlace-v4", render_mode="rgb_array", max_episode_steps=140)
    t0 = time.time()
    imgs, states, acts, ep_start, ep_len = [], [], [], [], []
    ok = 0
    for s in range(n_demos):
        succ, im, st, ac = expert_episode(env, 2000 + s, record=True)
        ok += int(succ)
        ep_start.append(sum(ep_len))
        ep_len.append(len(im))
        imgs.append(np.stack(im))
        states.append(np.stack(st))
        acts.append(np.stack(ac))
    env.close()
    imgs = np.concatenate(imgs)
    states = np.concatenate(states)
    acts = np.concatenate(acts)
    print(f"采示教 {n_demos} 条，专家成功率 {ok}/{n_demos}，共 {len(imgs)} 帧，"
          f"{time.time() - t0:.0f}s", flush=True)

    ds = FetchDS(imgs, states, acts, np.array(ep_start), np.array(ep_len), vocab,
                 float(np.mean(ep_len)))
    dl = DataLoader(ds, batch_size=batch, shuffle=True, drop_last=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MiniVLA(vocab_size=len(vocab), n_cam=1, act_dim=ACT, chunk=CHUNK,
                    state_dim=HIST * 8 + 1, img=IMG, d=160, n_head=8, n_layer=3,
                    n_act_layer=2).to(device)
    print(f"samples={len(ds)} params={sum(p.numel() for p in model.parameters())/1e6:.2f}M", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(dl))
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    gstep, t1 = 0, time.time()
    for ep in range(epochs):
        run, n = 0.0, 0
        for im, st, tx, ac in dl:
            im, st, tx, ac = im.to(device), st.to(device), tx.to(device), ac.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                mem = model.encode(im[:, None], st, tx)
                loss = model.flow_loss(mem, ac)
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
        print(f"epoch {ep+1:2d}/{epochs} loss={run/max(n,1):.4f} {time.time() - t1:.0f}s", flush=True)

    path = os.path.join(CKPT, "mini_vla_fetch.pt")
    torch.save({"model": model.state_dict(), "vocab": vocab, "max_len": 8, "chunk": CHUNK,
                "img": IMG, "hist": HIST, "n_cam": 1, "act_dim": ACT,
                "mean_len": float(np.mean(ep_len)),
                "state_mean": ds.state_norm.mean, "state_std": ds.state_norm.std,
                "act_mean": ds.act_norm.mean, "act_std": ds.act_norm.std}, path)
    print("saved", path, flush=True)
    return model, ds


if __name__ == "__main__":
    main()
