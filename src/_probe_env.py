"""Probe the env: scripted pick-and-place via DLS servo, then dump observation images."""
import time
import numpy as np
from PIL import Image

from mujoco_env import PandaLab, GRIP_OPEN, GRIP_CLOSED, BIN_CENTER

env = PandaLab(img_size=96)
obs = env.reset("bin", seed=0)
print("cube", env.cube_pos().round(3), "tcp", env.tcp_pos().round(3), "hand_z", env.hand_z().round(3))

t0 = time.time()


def go(target, gripper, ticks, label):
    for _ in range(ticks):
        env.servo(target, gripper)
    err = np.linalg.norm(np.asarray(target) - env.tcp_pos())
    print(f"{label:14s} target={np.round(target,3)} err={err*100:5.2f} cm z={env.tcp_pos()[2]:.3f}")
    return err


cube = env.cube_pos()
go(cube + [0, 0, 0.10], GRIP_OPEN, 45, "approach")
go(cube, GRIP_OPEN, 30, "descend")
for _ in range(12):
    env.step_ctrl(np.concatenate([env.d.qpos[:7], [GRIP_CLOSED]]))
print("after close: cube", env.cube_pos().round(3), "finger", round(float(env.d.qpos[7]), 3))
go(cube + [0, 0, 0.16], GRIP_CLOSED, 35, "lift")
print("lifted?", env.lifted, "max_lift", round(env.max_lift, 3), "cube", env.cube_pos().round(3))

tgt = np.array([BIN_CENTER[0], BIN_CENTER[1], 0.17])
go(tgt, GRIP_CLOSED, 45, "to bin")
go([BIN_CENTER[0], BIN_CENTER[1], 0.05], GRIP_CLOSED, 20, "lower")
for _ in range(10):
    env.step_ctrl(np.concatenate([env.d.qpos[:7], [GRIP_OPEN]]))
for _ in range(10):
    env.step_ctrl(np.concatenate([env.d.qpos[:7], [GRIP_OPEN]]))
go([BIN_CENTER[0], BIN_CENTER[1], 0.16], GRIP_OPEN, 18, "retreat")

print("final cube", env.cube_pos().round(3), "success:",
      env.success())
print(f"total {time.time()-t0:.2f}s")

obs = env.get_obs()
for k, v in obs["images"].items():
    Image.fromarray(v).save(f"../outputs/probe_{k}.png")
print("state", obs["state"].round(3), "img", obs["images"]["front"].shape)
