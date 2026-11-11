"""Diagnose the grasp: log cube z, finger joint, tcp each control tick."""
import numpy as np

from mujoco_env import PandaLab, GRIP_OPEN, GRIP_CLOSED

env = PandaLab(img_size=96)
obs = env.reset("bin", seed=0)
cube0 = env.cube_pos().copy()
print("cube", cube0.round(3))

for _ in range(45):
    env.servo(cube0 + [0, 0, 0.10], GRIP_OPEN)
for _ in range(35):
    env.servo(cube0, GRIP_OPEN)
print("descended tcp", env.tcp_pos().round(3), "cube", env.cube_pos().round(3))

for k in range(40):
    env.step_ctrl(np.concatenate([env.d.qpos[:7], [GRIP_CLOSED]]))
    if k % 4 == 0:
        print(f"close {k:2d}: finger_joint={env.d.qpos[7]:.4f} cube_z={env.cube_pos()[2]:.4f} "
              f"cube_xy={env.cube_pos()[:2].round(3)}")

print("--- lift in 5 vertical waypoints, 8 ticks each ---")
start = env.cube_pos().copy()
for h in (0.05, 0.09, 0.13, 0.17, 0.20):
    tgt = np.array([start[0], start[1], h])
    for _ in range(8):
        env.servo(tgt, GRIP_CLOSED, max_dq=0.15)
    print(f"lift h={h:.2f}: cube={env.cube_pos().round(3)} finger={env.d.qpos[7]:.3f} "
          f"slip={np.linalg.norm(env.cube_pos()[:2]-start[:2])*100:.2f}cm")
