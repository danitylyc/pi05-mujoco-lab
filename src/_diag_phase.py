"""Phase-by-phase trace of the pick-and-place expert."""
import numpy as np

from mujoco_env import PandaLab, BIN_CENTER, GRIP_OPEN, GRIP_CLOSED

env = PandaLab(img_size=84)
obs = env.reset("bin", seed=1000)
cube = env.cube_pos().copy()
print("cube0", cube.round(4), "tcp0", env.tcp_pos().round(4))


def phase(label, target, grip, n, **kw):
    for _ in range(n):
        env.servo(target, grip, **kw)
    print(f"{label:10s} target={np.round(target,3)} tcp={env.tcp_pos().round(3)} "
          f"err={np.linalg.norm(env.tcp_pos()-np.asarray(target))*100:5.2f}cm "
          f"cube={env.cube_pos().round(3)} finger={env.d.qpos[7]:.3f}")


phase("approach", cube + [0, 0, 0.12], GRIP_OPEN, 34, max_dq=0.40)
phase("descend", [cube[0], cube[1], 0.024], GRIP_OPEN, 24, max_dq=0.20)
phase("close", [cube[0], cube[1], 0.024], GRIP_CLOSED, 22, max_dq=0.20)
phase("close2", [cube[0], cube[1], 0.024], GRIP_CLOSED, 20, max_dq=0.20)
for h in (0.09, 0.15, 0.20):
    phase(f"lift{h}", [cube[0], cube[1], h], GRIP_CLOSED, 9, max_dq=0.18)
print("lifted", env.lifted, "max_lift", round(env.max_lift, 3))
phase("to bin", [BIN_CENTER[0], BIN_CENTER[1], 0.20], GRIP_CLOSED, 40, max_dq=0.45)
print("cube after transfer", env.cube_pos().round(3))
phase("lower", [BIN_CENTER[0], BIN_CENTER[1], 0.105], GRIP_CLOSED, 20, max_dq=0.22)
phase("open", [BIN_CENTER[0], BIN_CENTER[1], 0.105], GRIP_OPEN, 16, max_dq=0.22)
phase("retreat", [BIN_CENTER[0], BIN_CENTER[1], 0.195], GRIP_OPEN, 10, max_dq=0.35)
print("final cube", env.cube_pos().round(3), "success", env.success())
