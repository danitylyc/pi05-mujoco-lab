"""FK / gripper / IK probe for the local MuJoCo Panda scene."""
import os
import numpy as np
import mujoco

os.environ.setdefault("MUJOCO_GL", "glfw")

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENE = os.path.join(HERE, "assets", "panda", "scene_pi05.xml")

m = mujoco.MjModel.from_xml_path(SCENE)
d = mujoco.MjData(m)

print("nq", m.nq, "nv", m.nv, "nu", m.nu, "nkey", m.nkey)
print("joint names:", [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(m.njnt)])
print("actuator names:", [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)])

hand = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "hand")
fing1 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
fing2 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_finger")
cube = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "cube")

mujoco.mj_resetDataKeyframe(m, d, 0)
mujoco.mj_forward(m, d)
print("home hand pos", d.xpos[hand].round(4))

# gripper sweep: what does ctrl 0 vs 255 mean?
for g in (0.0, 127.5, 255.0):
    mujoco.mj_resetDataKeyframe(m, d, 0)
    for _ in range(60):
        d.ctrl[:7] = d.qpos[:7]
        d.ctrl[7] = g
        mujoco.mj_step(m, d)
    gap = np.linalg.norm(d.xpos[fing1] - d.xpos[fing2])
    print(f"gripper ctrl={g:6.1f} finger joint={d.qpos[7]:.4f} gap={gap*100:.3f} cm")

# joint limits
print("jnt ranges:")
for i in range(7):
    print("  joint", i + 1, m.jnt_range[i].round(3))
print("finger range", m.jnt_range[7].round(4))
