"""Debug: why does the scripted expert fail?"""
import numpy as np

import gen_demos
from mujoco_env import PandaLab, BIN_CENTER, PLATE_CENTER, PUSH_CENTER, TASKS

env = PandaLab(img_size=84)
for task in ("bin", "plate", "push"):
    for seed in (1000, 1001):
        imgs, states, acts, ok = gen_demos.rollout_expert(env, task, seed)
        c = env.cube_pos()
        tgt = {"bin": BIN_CENTER, "plate": PLATE_CENTER, "push": PUSH_CENTER}[task]
        print(f"{task} seed={seed} steps={len(acts)} ok={ok} "
              f"cube={np.round(c,3)} dist_to_tgt={np.linalg.norm(c[:2]-tgt)*100:.1f}cm "
              f"lifted={env.lifted} max_lift={env.max_lift:.3f} grip={env.d.qpos[7]:.3f}")
