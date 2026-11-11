"""用脚本化专家（DLS 逆运动学伺服）在 MuJoCo 里采集示教数据。

产出 data/<task>.npz：图像、本体状态、8 维动作（7 关节增量 + 夹爪归一化）、episode 边界与成功率。
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
from tqdm import tqdm

from mujoco_env import (PandaLab, TASKS, BIN_CENTER, PLATE_CENTER, MARKER_CENTER,
                        GRIP_OPEN, GRIP_CLOSED)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, "data")

TARGETS = {"bin": BIN_CENTER, "plate": PLATE_CENTER, "marker": MARKER_CENTER}
LOWZ = {"bin": 0.105, "plate": 0.085, "marker": 0.045}


def rollout_expert(env, task, seed, record=True):
    """返回 (images, states, actions, success)。动作 = 7 关节增量 + 夹爪归一化。"""
    obs = env.reset(task, seed)
    rec = []

    def act(target, grip, **kw):
        nonlocal obs
        target = np.asarray(target, dtype=float)
        ctrl = env.compute_servo(target, grip, **kw)
        if record:
            dq = ctrl[:7] - env.d.qpos[:7]
            a = np.concatenate([dq, [ctrl[7] / 255.0]]).astype(np.float32)
            rec.append((obs["images"], obs["state"], a))
        env.step_ctrl(ctrl)
        obs = env.get_obs()

    cube = env.cube_pos().copy()
    tgt_xy = TARGETS[task].astype(float)
    zlow = LOWZ[task]

    for _ in range(34):                                   # 1 悬停在方块上方
        act(cube + [0, 0, 0.12], GRIP_OPEN, max_dq=0.35)
    for _ in range(26):                                   # 2 缓慢下降到方块中心
        act([cube[0], cube[1], 0.024], GRIP_OPEN, max_dq=0.14)
    for _ in range(20):                                   # 3 闭合夹爪
        act([cube[0], cube[1], 0.024], GRIP_CLOSED, max_dq=0.14)
    for h in np.linspace(0.030, 0.205, 26):               # 4 慢速竖直抬起
        act([cube[0], cube[1], h], GRIP_CLOSED, max_dq=0.06)
    for _ in range(44):                                   # 5 平移到目标上方
        act([tgt_xy[0], tgt_xy[1], 0.205], GRIP_CLOSED, max_dq=0.16)
    for _ in range(18):                                   # 6 下降到落点
        act([tgt_xy[0], tgt_xy[1], zlow], GRIP_CLOSED, max_dq=0.09)
    for _ in range(14):                                   # 7 松开
        act([tgt_xy[0], tgt_xy[1], zlow], GRIP_OPEN, max_dq=0.09)
    for _ in range(9):                                    # 8 撤离
        act([tgt_xy[0], tgt_xy[1], zlow + 0.09], GRIP_OPEN, max_dq=0.25)

    if not record:
        return None, None, None, env.success()

    imgs = np.stack([r[0] for r in rec], 0)
    states = np.stack([r[1] for r in rec], 0)
    acts = np.stack([r[2] for r in rec], 0)
    return imgs, states, acts, env.success()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--img", type=int, default=84)
    args = ap.parse_args()

    os.makedirs(DATA, exist_ok=True)
    env = PandaLab(img_size=args.img)
    t0 = time.time()

    for task in ("bin", "plate", "marker"):
        all_img, all_state, all_act, ep_idx, ep_len, ep_ok = [], [], [], [], [], []
        for ep in tqdm(range(args.episodes), desc=task):
            imgs, states, acts, ok = rollout_expert(env, task, seed=1000 + ep)
            all_img.append(imgs)
            all_state.append(states)
            all_act.append(acts)
            ep_idx.append(np.full(len(acts), ep, dtype=np.int32))
            ep_len.append(len(acts))
            ep_ok.append(ok)
        np.savez_compressed(
            os.path.join(DATA, f"{task}.npz"),
            images=np.concatenate(all_img, 0),
            states=np.concatenate(all_state, 0),
            actions=np.concatenate(all_act, 0),
            ep_idx=np.concatenate(ep_idx, 0),
            ep_len=np.array(ep_len, dtype=np.int32),
            ep_success=np.array(ep_ok),
            instructions=np.array(TASKS[task]),
        )
        print(f"[{task}] episodes={len(ep_len)} steps={sum(ep_len)} "
              f"expert success={np.mean(ep_ok) * 100:.1f}%")

    print(f"done in {time.time() - t0:.1f}s -> {DATA}")


if __name__ == "__main__":
    main()
