"""Probe: can MuJoCo do offscreen rendering on this Windows box?"""
import os
import time

os.environ.setdefault("MUJOCO_GL", "glfw")

import mujoco
import numpy as np

XML = """
<mujoco>
  <visual><global offwidth="320" offheight="240"/></visual>
  <worldbody>
    <light pos="0 0 2"/>
    <geom type="plane" size="1 1 .1" rgba=".6 .6 .6 1"/>
    <body name="box" pos="0 0 .2">
      <freejoint/>
      <geom type="box" size=".05 .05 .05" rgba=".9 .2 .2 1"/>
    </body>
    <camera name="cam" pos="0.6 -0.6 0.5" xyaxes="1 1 0 -0.4 0.4 1"/>
  </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(XML)
data = mujoco.MjData(model)

renderer = mujoco.Renderer(model, 240, 320)
data.qpos[:3] = [0, 0, 0.2]
for _ in range(10):
    mujoco.mj_step(model, data)
renderer.update_scene(data, camera="cam")
img = renderer.render()

out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "render_probe.png")
if img.ndim == 3 and img.shape[-1] == 3:
    try:
        import cv2
        cv2.imwrite(out, img[:, :, ::-1])
    except Exception:
        import PIL.Image
        PIL.Image.fromarray(img).save(out)

print("shape", img.shape, "dtype", img.dtype, "mean", float(img.mean()))
print("saved", out)

t0 = time.time()
for _ in range(20):
    renderer.update_scene(data, camera="cam")
    img = renderer.render()
dt = (time.time() - t0) / 20
print(f"render latency {dt*1000:.1f} ms/frame")
