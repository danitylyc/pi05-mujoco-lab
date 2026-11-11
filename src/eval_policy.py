"""闭环评测：加载训练好的 mini-VLA，在 MuJoCo 环境里滚动作，统计成功率并录像。"""
from __future__ import annotations

import argparse
import json
import os
from collections import deque

import cv2
import numpy as np
import torch

from mini_vla import MiniVLA, Normalizer, encode_text
from mujoco_env import PandaLab, TASKS

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(HERE, "checkpoints")
OUT = os.path.join(HERE, "outputs")
MAX_STEPS = 260
CENTERS = {"bin": np.array([0.50, 0.28]), "plate": np.array([0.50, -0.26]),
           "marker": np.array([0.68, 0.0])}


class Policy:
    def __init__(self, ckpt_path, device="cuda", exec_steps=6, denoise=8):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        self.hist = ck.get("hist", 1)
        self.model = MiniVLA(vocab_size=len(ck["vocab"]), chunk=ck["chunk"], act_dim=8,
                             state_dim=self.hist * 8 + 1, img=ck["img"]).to(device).eval()
        self.model.load_state_dict(ck["model"])
        self.vocab, self.max_len = ck["vocab"], ck["max_len"]
        self.chunk, self.img = ck["chunk"], ck["img"]
        self.mean_len = ck.get("mean_len", 190.0)
        self.state_norm = Normalizer(ck["state_mean"], ck["state_std"])
        self.act_norm = Normalizer(ck["act_mean"], ck["act_std"])
        self.device, self.exec_steps, self.denoise = device, exec_steps, denoise

    def reset(self):
        self.buf = deque(maxlen=self.hist)
        self.t = 0

    def observe(self, obs):
        self.buf.append(obs["state"].astype(np.float32))
        while len(self.buf) < self.hist:
            self.buf.appendleft(obs["state"].astype(np.float32))
        vec = np.concatenate(list(self.buf) + [[min(self.t / self.mean_len, 1.3)]]).astype(np.float32)
        return self.state_norm.norm(vec)

    @torch.no_grad()
    def chunk_actions(self, obs, instr):
        imgs = obs["images"].astype(np.float32) / 255.0
        imgs = np.transpose(imgs, (0, 3, 1, 2))
        imgs = torch.from_numpy((imgs - 0.5) / 0.5)[None].to(self.device)
        state = torch.from_numpy(self.observe(obs)[None]).to(self.device)
        txt = torch.from_numpy(encode_text([instr], self.vocab, self.max_len)).to(self.device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device == "cuda"):
            memory = self.model.encode(imgs, state, txt)
        a = self.model.sample(memory, steps=self.denoise)[0].float().cpu().numpy()
        return self.act_norm.denorm(a)


def run_episode(env, policy, task, seed, record=False, cam=0):
    obs = env.reset(task, seed)
    policy.reset()
    instr = TASKS[task][seed % len(TASKS[task])]
    writer = None
    if record:
        env.init_video(w=480, h=360, which=cam)
        writer = cv2.VideoWriter(os.path.join(OUT, f"eval_{task}_seed{seed}.mp4"),
                                 cv2.VideoWriter_fourcc(*"mp4v"), 20, (480, 360))
    ok, t = False, 0
    while t < MAX_STEPS and not env.success():
        if record:
            writer.write(cv2.cvtColor(env.render_frame(), cv2.COLOR_RGB2BGR))
        chunk = policy.chunk_actions(obs, instr)
        for a in chunk[:policy.exec_steps]:
            ctrl = np.concatenate([a[:7], [np.clip(a[7], 0.0, 1.0) * 255.0]])
            env.step_ctrl(ctrl)
            obs = env.get_obs()
            policy.t += 1
            t += 1
            if record:
                writer.write(cv2.cvtColor(env.render_frame(), cv2.COLOR_RGB2BGR))
            if env.success() or t >= MAX_STEPS:
                break
    ok = bool(env.success())
    if record:
        for _ in range(10):
            writer.write(cv2.cvtColor(env.render_frame(), cv2.COLOR_RGB2BGR))
        writer.release()
    return ok, t, float(np.linalg.norm(env.cube_pos()[:2] - CENTERS[task]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(CKPT, "mini_vla.pt"))
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--exec-steps", type=int, default=6)
    ap.add_argument("--denoise", type=int, default=8)
    ap.add_argument("--video", type=int, default=2)
    ap.add_argument("--seed0", type=int, default=5000)
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = Policy(args.ckpt, device, exec_steps=args.exec_steps, denoise=args.denoise)
    env = PandaLab(img_size=policy.img)

    report = {}
    for task in ("bin", "plate", "marker"):
        oks, dists = [], []
        for k in range(args.episodes):
            ok, t, dist = run_episode(env, policy, task, args.seed0 + k, record=k < args.video)
            oks.append(ok)
            dists.append(dist)
            print(f"  {task} seed={args.seed0+k} success={ok} steps={t} dist={dist*100:.1f}cm", flush=True)
        report[task] = {"success_rate": float(np.mean(oks)), "n": args.episodes,
                        "mean_dist_cm": float(np.mean(dists) * 100)}
        print(f"[{task}] success={np.mean(oks)*100:.1f}% ({sum(oks)}/{args.episodes})", flush=True)

    report["overall"] = float(np.mean([report[t]["success_rate"] for t in ("bin", "plate", "marker")]))
    with open(os.path.join(OUT, "eval_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("overall", round(report["overall"], 3), flush=True)


if __name__ == "__main__":
    main()
