"""在主流 benchmark（FetchPickAndPlace-v4）上闭环评测训练好的 mini-VLA。"""
import os

os.environ.setdefault("MUJOCO_GL", "glfw")

import collections
import json

import gymnasium as gym
import gymnasium_robotics  # noqa: F401
import numpy as np
import torch

from mini_vla import MiniVLA, Normalizer, encode_text
from fetch_pipeline import state8, img112, INSTR

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CK = os.path.join(HERE, "checkpoints", "mini_vla_fetch.pt")
EXEC = 6


class Policy:
    def __init__(self, path):
        ck = torch.load(path, map_location="cuda", weights_only=False)
        self.ck = ck
        self.model = MiniVLA(vocab_size=len(ck["vocab"]), n_cam=1, act_dim=ck["act_dim"],
                             chunk=ck["chunk"], state_dim=ck["hist"] * 8 + 1,
                             img=ck["img"], d=160, n_head=8, n_layer=3,
                             n_act_layer=2).cuda().eval()
        self.model.load_state_dict(ck["model"])
        self.s_norm = Normalizer(ck["state_mean"], ck["state_std"])
        self.a_norm = Normalizer(ck["act_mean"], ck["act_std"])
        self.tok = torch.from_numpy(encode_text([INSTR], ck["vocab"], 8)).cuda()

    def reset(self):
        self.buf = collections.deque(maxlen=self.ck["hist"])
        self.t = 0

    def act(self, obs):
        st = state8(obs)
        self.buf.append(st)
        while len(self.buf) < self.ck["hist"]:
            self.buf.appendleft(st.copy())
        vec = np.concatenate(list(self.buf) + [[min(self.t / self.ck["mean_len"], 1.3)]])
        s = torch.from_numpy(self.s_norm.norm(vec)[None].astype(np.float32)).cuda()
        return s


def main(episodes=20):
    pol = Policy(CK)
    env = gym.make("FetchPickAndPlace-v4", render_mode="rgb_array", max_episode_steps=140)

    ok, d0, d1 = 0, [], []
    for k in range(episodes):
        obs, info = env.reset(seed=3000 + k)
        pol.reset()
        d0.append(float(np.linalg.norm(np.asarray(obs["achieved_goal"]) - obs["desired_goal"])))
        done = False
        while not done:
            img = img112(env)
            im = np.transpose(img.astype(np.float32) / 255.0, (2, 0, 1))
            im = torch.from_numpy((im - 0.5) / 0.5)[None, None].cuda()
            s = pol.act(obs)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                mem = pol.model.encode(im, s, pol.tok)
                chunk = pol.a_norm.denorm(pol.model.sample(mem, steps=8)[0].float().cpu().numpy())
            for a in chunk[:EXEC]:
                obs, r, term, trunc, info = env.step(np.clip(a, -1, 1).astype(np.float32))
                pol.t += 1
                if term or trunc:
                    done = True
                    break
        succ = bool(info.get("is_success", False))
        ok += int(succ)
        d1.append(float(np.linalg.norm(np.asarray(obs["achieved_goal"]) - obs["desired_goal"])))
        print(f"  seed={3000 + k} success={succ} dist={d1[-1] * 100:.1f}cm", flush=True)

    print(f"FetchPickAndPlace-v4 闭环成功率 {ok}/{episodes} = {ok / episodes * 100:.1f}%", flush=True)
    out = os.path.join(HERE, "outputs", "eval_fetch.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"task": "FetchPickAndPlace-v4", "episodes": episodes,
                   "success_rate": ok / episodes,
                   "mean_init_dist_cm": float(np.mean(d0) * 100),
                   "mean_final_dist_cm": float(np.mean(d1) * 100)}, f, indent=2)
    env.close()


if __name__ == "__main__":
    main()
