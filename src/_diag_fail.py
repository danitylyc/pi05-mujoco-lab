"""量化闭环失败模式：对每个 seed 记录 tcp 与方块最近距离、夹爪闭合时的状态。"""
import os
import sys

import numpy as np
import torch

from eval_policy import Policy, CENTERS
from mujoco_env import PandaLab, TASKS

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ck = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "checkpoints", "mini_vla_v3.pt")

policy = Policy(ck, "cuda", exec_steps=6)
env = PandaLab(img_size=policy.img)
task = "bin"

for seed in (5000, 5006, 5011, 5012, 5014):
    obs = env.reset(task, seed)
    policy.reset()
    instr = TASKS[task][0]
    min_d, close_z, close_d, t_close, cube0 = 9.9, None, None, None, env.cube_pos().copy()
    for t in range(260):
        chunk = policy.chunk_actions(obs, instr)
        for a in chunk[:policy.exec_steps]:
            grip_cmd = float(np.clip(a[7], 0, 1))
            if close_z is None and grip_cmd < 0.3:
                close_z = env.tcp_pos()[2]
                close_d = float(np.linalg.norm(env.tcp_pos()[:2] - env.cube_pos()[:2]))
                t_close = t
            env.step_ctrl(np.concatenate([a[:7], [grip_cmd * 255.0]]))
            obs = env.get_obs()
            policy.t += 1
            d = float(np.linalg.norm(env.tcp_pos()[:2] - env.cube_pos()[:2]))
            min_d = min(min_d, d)
        if env.success():
            break
    print(f"seed={seed} cube0={np.round(cube0[:2],3)} 最近水平距离={min_d*100:.1f}cm "
          f"闭合时刻t={t_close} 闭合时z={None if close_z is None else round(close_z,3)} "
          f"闭合时距离={None if close_d is None else round(close_d*100,1)}cm "
          f"success={env.success()}", flush=True)
