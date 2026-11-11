"""录一条专家示教视频，用于和学出来的策略做对照。"""
import os

import cv2

from gen_demos import TARGETS, LOWZ
from mujoco_env import PandaLab, GRIP_OPEN, GRIP_CLOSED
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "outputs")


def main():
    env = PandaLab(img_size=112)
    for task in ("bin", "plate", "marker"):
        obs = env.reset(task, seed=1000)
        env.init_video(w=480, h=360, which=0)
        w = cv2.VideoWriter(os.path.join(OUT, f"expert_{task}.mp4"),
                            cv2.VideoWriter_fourcc(*"mp4v"), 20, (480, 360))
        cube = env.cube_pos().copy()
        tgt = TARGETS[task].astype(float)
        zlow = LOWZ[task]

        def act(target, grip, **kw):
            w.write(cv2.cvtColor(env.render_frame(), cv2.COLOR_RGB2BGR))
            env.servo(target, grip, **kw)

        for _ in range(34):
            act(cube + [0, 0, 0.12], GRIP_OPEN, max_dq=0.35)
        for _ in range(26):
            act([cube[0], cube[1], 0.024], GRIP_OPEN, max_dq=0.14)
        for _ in range(20):
            act([cube[0], cube[1], 0.024], GRIP_CLOSED, max_dq=0.14)
        for h in np.linspace(0.030, 0.205, 26):
            act([cube[0], cube[1], h], GRIP_CLOSED, max_dq=0.06)
        for _ in range(44):
            act([tgt[0], tgt[1], 0.205], GRIP_CLOSED, max_dq=0.16)
        for _ in range(18):
            act([tgt[0], tgt[1], zlow], GRIP_CLOSED, max_dq=0.09)
        for _ in range(14):
            act([tgt[0], tgt[1], zlow], GRIP_OPEN, max_dq=0.09)
        for _ in range(12):
            act([tgt[0], tgt[1], zlow + 0.09], GRIP_OPEN, max_dq=0.25)
        w.release()
        print(task, "success=", env.success(), flush=True)


if __name__ == "__main__":
    main()
