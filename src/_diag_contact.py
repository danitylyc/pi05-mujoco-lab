"""Per-tick trace: cube z, finger joint, tcp z, and contact force between pads and cube."""
import numpy as np
import mujoco

from mujoco_env import PandaLab, GRIP_OPEN, GRIP_CLOSED

env = PandaLab(img_size=84)
env.reset("bin", seed=1000)
cube = env.cube_pos().copy()

m, d = env.m, env.d
cg = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")


def contact_force():
    tot = 0.0
    for i in range(d.ncon):
        c = d.contact[i]
        if c.geom1 == cg or c.geom2 == cg:
            f = np.zeros(6)
            mujoco.mj_contactForce(m, d, i, f)
            tot += abs(f[0])
    return tot


def run(label, target, grip, n, **kw):
    for k in range(n):
        env.servo(target, grip, **kw)
        if k % 3 == 0 or True:
            print(f"{label:9s} k={k:2d} tcp_z={env.tcp_pos()[2]:.3f} cube_z={env.cube_pos()[2]:.4f} "
                  f"finger={env.d.qpos[7]:.4f} fN={contact_force():6.2f}")


run("approach", cube + [0, 0, 0.12], GRIP_OPEN, 34, max_dq=0.35)
run("descend", [cube[0], cube[1], 0.024], GRIP_OPEN, 26, max_dq=0.14)
run("close", [cube[0], cube[1], 0.024], GRIP_CLOSED, 20, max_dq=0.14)
for h in np.linspace(0.030, 0.205, 26)[:12]:
    run(f"lift{h:.3f}", [cube[0], cube[1], h], GRIP_CLOSED, 1, max_dq=0.06)
