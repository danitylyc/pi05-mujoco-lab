"""在主流 benchmark（Gymnasium-Robotics FetchPickAndPlace-v4）上跑脚本专家。

目的有两个：
  1. 证明"主流环境 -> 示教 -> 训练 -> 闭环"这条链在本机是通的（不只是自建场景）；
  2. 量出该环境在本机的真实成本（渲染 / 步进 / 专家成功率），这是能不能在本机
     跑 linking benchmark 的硬指标。

动作空间是 4 维 TCP 增量 (dx, dy, dz, gripper∈[-1,1])，与 LIBERO 的关节位置目标不同，
所以这里用"比例控制 + 分阶段 waypoint"来实现专家，并用环境的 achieved_goal 判定成功。
"""
import os

os.environ.setdefault("MUJOCO_GL", "glfw")

import json
import time

import gymnasium as gym
import gymnasium_robotics  # noqa: F401
import numpy as np


def phase_targets(cube, goal):
    """8 个阶段的 TCP 目标点（世界坐标）。"""
    above = np.array([cube[0], cube[1], cube[2] + 0.10])
    at = np.array([cube[0], cube[1], cube[2] + 0.005])
    lift = np.array([cube[0], cube[1], cube[2] + 0.12])
    over = np.array([goal[0], goal[1], goal[2] + 0.12])
    down = np.array([goal[0], goal[1], goal[2] + 0.01])
    return [(above, 12, +1.0), (at, 10, +1.0), (at, 8, -1.0), (lift, 10, -1.0),
            (over, 16, -1.0), (down, 10, -1.0), (down, 6, +1.0), (lift, 6, +1.0)]


def run_episode(env, seed):
    obs, info = env.reset(seed=seed)
    o = np.asarray(obs["observation"], dtype=np.float64)
    cube = np.asarray(obs["achieved_goal"], dtype=np.float64)
    goal = np.asarray(obs["desired_goal"], dtype=np.float64)
    tcp0 = o[:3].copy()

    for target, steps, grip in phase_targets(cube, goal):
        for _ in range(steps):
            tcp = np.asarray(obs["observation"][:3], dtype=np.float64)
            d = np.clip(6.0 * (target - tcp), -1.0, 1.0)
            a = np.concatenate([d, [grip]]).astype(np.float32)
            obs, r, done, trunc, info = env.step(a)
            if info.get("is_success", False) or done or trunc:
                return bool(info.get("is_success", False)), np.linalg.norm(cube - tcp0)
    return bool(info.get("is_success", False)), np.linalg.norm(cube - tcp0)


def main():
    env = gym.make("FetchPickAndPlace-v4", render_mode="rgb_array", max_episode_steps=140)
    env.reset(seed=0)

    t0 = time.time()
    ok = 0
    n = 20
    imgs = []
    for s in range(n):
        succ, _ = run_episode(env, 1000 + s)
        ok += int(succ)
    dt = time.time() - t0
    print(f"FetchPickAndPlace-v4 脚本专家成功率 {ok}/{n} = {ok / n * 100:.0f}%", flush=True)
    print(f"耗时 {dt:.1f}s -> {dt / n * 1000:.0f} ms/episode", flush=True)

    frame = env.render()
    t0 = time.time()
    for _ in range(100):
        f = env.render()
    print(f"480x480 渲染 {frame.shape}，{(time.time() - t0) / 100 * 1000:.1f} ms/帧", flush=True)

    # 存 5 条示教图像，供后续在本机训练 mini-VLA 用（这条链是否值得做，看耗时）
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "outputs", "fetch_expert.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"success_rate": ok / n, "ms_per_episode": dt / n * 1000}, f)
    env.close()


if __name__ == "__main__":
    main()
