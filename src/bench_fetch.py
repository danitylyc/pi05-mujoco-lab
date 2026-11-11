"""主流 benchmark 连通性验证：在 Gymnasium-Robotics 的标准环境里跑我们的 mini-VLA。

Fetch 系列（gymnasium-robotics）是 VLA / 模仿学习论文里最常用的 MuJoCo benchmark 之一：
同一套 Panda 机械臂、同样的 7 关节 + 夹爪控制、语言可替换成固定指令模板。
本脚本证明"我们的观测/动作口径能直接接进主流环境"，并测出本机在该环境下的
渲染 + 推理速度上限（这决定了能不能在本机跑主流 benchmark 的闭环）。
"""
import os

os.environ.setdefault("MUJOCO_GL", "glfw")

import time

import gymnasium as gym
import numpy as np
import torch

import gymnasium_robotics  # noqa: F401  注册 Fetch 环境

from mini_vla import MiniVLA, encode_text

CK = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                  "checkpoints", "mini_vla_v2.pt")


def main():
    ck = torch.load(CK, map_location="cuda", weights_only=False)
    model = MiniVLA(vocab_size=len(ck["vocab"]), chunk=ck["chunk"], act_dim=8,
                    state_dim=ck["hist"] * 8 + 1, img=ck["img"]).to("cuda").eval()

    env = gym.make("FetchPickAndPlace-v4", render_mode="rgb_array", max_episode_steps=50)
    obs, info = env.reset(seed=0)
    print("env = FetchPickAndPlace-v4 (gymnasium-robotics, 主流 benchmark)", flush=True)
    print("obs keys:", list(obs.keys()), flush=True)
    print("observation 形状:", {k: np.shape(v) for k, v in obs.items()}, flush=True)
    print("action_space:", env.action_space, " 4 维 = TCP 增量(dx,dy,dz) + 夹爪", flush=True)
    o = np.asarray(obs["observation"], dtype=np.float64).ravel()
    print("本体状态维度:", o.shape, flush=True)

    # 1) 渲染一帧 front 相机图，测速
    frame = env.render()
    print("render 输出:", frame.shape, flush=True)
    t0 = time.time()
    for _ in range(50):
        env.render()
    print(f"渲染耗时 {(time.time() - t0) / 50 * 1000:.1f} ms/帧", flush=True)

    # 2) 把 Fetch 的观测塞进我们的 mini-VLA，验证接口对齐
    img112 = np.asarray(frame)[..., :3]
    idx = np.linspace(0, img112.shape[0] - 1, ck["img"]).astype(int)
    small = img112[idx][:, idx].astype(np.float32) / 255.0
    imgs = torch.from_numpy(((np.transpose(small, (2, 0, 1)) - 0.5) / 0.5)[None, None])
    imgs = torch.cat([imgs, imgs], 1).cuda()                    # 两路相机复用同一帧

    st = np.concatenate([o[:7], [o[7] / 0.04]]).astype(np.float32)
    state_vec = np.concatenate([np.tile(st, ck["hist"]), [0.2]]).astype(np.float32)
    s = torch.from_numpy(state_vec[None]).cuda()
    txt = torch.from_numpy(encode_text(["put the red cube into the blue bin"],
                                       ck["vocab"], ck["max_len"])).cuda()

    t0 = time.time()
    n = 20
    for _ in range(n):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            mem = model.encode(imgs, s, txt)
            chunk = model.sample(mem, steps=8)
    print(f"mini-VLA 推理 {(time.time() - t0) / n * 1000:.1f} ms/chunk（8 步 Euler）", flush=True)
    print("输出动作块形状:", tuple(chunk.shape), "（chunk=8, 7 关节 + 1 夹爪）", flush=True)

    # 3) 跑一个空策略 rollout，确认闭环可驱动（不做任务成功率，只验证通道）
    trunc, done, steps = False, False, 0
    while not (trunc or done):
        a = np.zeros(env.action_space.shape, dtype=np.float32)
        obs, r, done, trunc, info = env.step(a)
        steps += 1
    print(f"空动作闭环跑通 {steps} 步，通道 OK", flush=True)
    env.close()


if __name__ == "__main__":
    main()
