"""闭环单条 episode 逐段追踪：看策略在哪一步开始偏离。"""
import os

import numpy as np
import torch

from eval_policy import Policy, CENTERS
from mujoco_env import PandaLab, TASKS

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ck = os.path.join(HERE, "checkpoints", "mini_vla_v2.pt")

policy = Policy(ck, "cuda", exec_steps=6)
env = PandaLab(img_size=policy.img)

task, seed = "bin", 5000
obs = env.reset(task, seed)
policy.reset()
instr = TASKS[task][0]
print("cube0", env.cube_pos().round(3), "tcp0", env.tcp_pos().round(3), flush=True)

t = 0
while t < 200 and not env.success():
    chunk = policy.chunk_actions(obs, instr)
    print(f"t={t:3d} tcp={np.round(env.tcp_pos(),3)} cube={np.round(env.cube_pos(),3)} "
          f"grip={env.d.qpos[7]:.3f} 首动作={np.round(chunk[0][:3],3)} 末动作={np.round(chunk[-1][:3],3)}",
          flush=True)
    for a in chunk[:policy.exec_steps]:
        env.step_ctrl(np.concatenate([a[:7], [np.clip(a[7], 0, 1) * 255.0]]))
        obs = env.get_obs()
        policy.t += 1
        t += 1
print("final cube", np.round(env.cube_pos(), 3), "success", env.success(),
      "dist_cm", round(float(np.linalg.norm(env.cube_pos()[:2] - CENTERS[task])) * 100, 1), flush=True)
