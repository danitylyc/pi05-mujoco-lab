"""MuJoCo 本地 VLA 实验环境：Franka Panda + 桌面抓放场景（LIBERO 风格）。

控制口径与 pi0.5 一致：8 维关节位置目标（7 关节 + 1 夹爪），20Hz 决策频率。
"""
from __future__ import annotations

import os

import numpy as np
import mujoco

os.environ.setdefault("MUJOCO_GL", "glfw")

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENE = os.path.join(HERE, "assets", "panda", "scene_pi05.xml")

TASKS = {
    "bin": ["put the red cube into the blue bin", "place the cube in the blue bin"],
    "plate": ["put the red cube on the yellow plate", "place the cube on the yellow plate"],
    "marker": ["put the red cube on the green target", "place the cube on the green marker"],
}

BIN_CENTER = np.array([0.50, 0.28])
BIN_HALF = 0.075
PLATE_CENTER = np.array([0.50, -0.26])
MARKER_CENTER = np.array([0.68, 0.0])

CUBE_HALF = 0.022
TCP_OFFSET = np.array([0.0, 0.0, 0.108])
GRIP_OPEN = 255.0
GRIP_CLOSED = 0.0
MAX_DELTA = 0.30          # 每个控制周期的关节增量上限（rad）


def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


class PandaLab:
    def __init__(self, img_size: int = 96, frame_skip: int = 10):
        self.m = mujoco.MjModel.from_xml_path(SCENE)
        self.d = mujoco.MjData(self.m)
        self.img_size = img_size
        self.frame_skip = frame_skip

        self.hand = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "hand")
        self.cube = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "cube")

        self.home_q = self.m.key_qpos[0][:7].copy()
        self.home_ctrl = np.concatenate([self.home_q, [GRIP_OPEN]])

        self.renderer = mujoco.Renderer(self.m, img_size, img_size)
        self._cams = self._build_cameras()
        self.cam_names = [c[0] for c in self._cams]

        self.task = "bin"
        self.lifted = False
        self.max_lift = 0.0
        self._i_err = np.zeros(3)
        self._last_target = None

    # ---------------------------------------------------------------- cameras
    def _build_cameras(self):
        def cam(lookat, dist, azim, elev):
            c = mujoco.MjvCamera()
            c.type = mujoco.mjtCamera.mjCAMERA_FREE
            c.lookat[:] = lookat
            c.distance = dist
            c.azimuth = azim
            c.elevation = elev
            return c

        return [
            ("front", cam([0.56, 0.0, 0.08], 0.85, 180.0, -30.0)),
            ("wrist", cam([0.5, 0.0, 0.05], 0.26, 180.0, -72.0)),
        ]

    # ---------------------------------------------------------------- state
    def tcp_pos(self):
        return self.d.xpos[self.hand] + self.d.xmat[self.hand].reshape(3, 3) @ TCP_OFFSET

    def hand_z(self):
        return self.d.xmat[self.hand].reshape(3, 3)[:, 2].copy()

    def cube_pos(self):
        return self.d.xpos[self.cube].copy()

    def joint_state(self):
        return np.concatenate([self.d.qpos[:7], [self.d.qpos[7] / 0.04]]).astype(np.float32)

    def _jacobian(self):
        jp = np.zeros((3, self.m.nv))
        jr = np.zeros((3, self.m.nv))
        mujoco.mj_jacBody(self.m, self.d, jp, jr, self.hand)
        r = self.tcp_pos() - self.d.xpos[self.hand]
        return jp - skew(r) @ jr, jr

    # ---------------------------------------------------------------- control
    def step_ctrl(self, ctrl):
        ctrl = np.asarray(ctrl, dtype=float).copy()
        ctrl = np.clip(ctrl, self.m.actuator_ctrlrange[:, 0], self.m.actuator_ctrlrange[:, 1])
        for _ in range(self.frame_skip):
            self.d.ctrl[:] = ctrl
            mujoco.mj_step(self.m, self.d)
        self._track_lift()
        return ctrl

    def compute_servo(self, target_pos, gripper, approach=(0.0, 0.0, -1.0),
                      pos_gain=1.35, rot_gain=0.45, max_dq=0.25, lam=0.06, ki=0.05):
        """只计算一步关节目标（不推进仿真），便于先记录观测再执行。"""
        tgt = np.asarray(target_pos, dtype=float)
        if self._last_target is None or np.linalg.norm(tgt - self._last_target) > 0.02:
            self._i_err[:] = 0.0          # 目标切换时清积分，避免 windup
        self._last_target = tgt.copy()

        err_p = tgt - self.tcp_pos()
        self._i_err = np.clip(self._i_err + err_p, -0.02, 0.02)
        err_p = err_p + ki * self._i_err

        want_z = np.asarray(approach, dtype=float)
        want_z /= np.linalg.norm(want_z) + 1e-9
        err_r = np.cross(self.hand_z(), want_z)

        jp, jr = self._jacobian()
        J = np.vstack([jp, rot_gain * jr])[:, :7]
        e = np.concatenate([pos_gain * err_p, rot_gain * err_r])
        dq = J.T @ np.linalg.solve(J @ J.T + lam ** 2 * np.eye(6), e)

        n = np.linalg.norm(dq)
        if n > max_dq:
            dq *= max_dq / n
        return np.concatenate([self.d.qpos[:7] + dq, [gripper]])

    def servo(self, target_pos, gripper, **kw):
        ctrl = self.compute_servo(target_pos, gripper, **kw)
        self.step_ctrl(ctrl)
        return ctrl

    def hold(self, gripper):
        ctrl = np.concatenate([self.d.qpos[:7], [gripper]])
        self.step_ctrl(ctrl)
        return ctrl

    # ---------------------------------------------------------------- episode
    def reset(self, task="bin", seed=0):
        rng = np.random.default_rng(seed)
        self.task = task
        self._i_err[:] = 0.0
        self._last_target = None
        mujoco.mj_resetDataKeyframe(self.m, self.d, 0)

        while True:
            cx, cy = rng.uniform(0.40, 0.62), rng.uniform(-0.13, 0.13)
            p = np.array([cx, cy])
            if (np.linalg.norm(p - BIN_CENTER) > 0.13
                    and np.linalg.norm(p - PLATE_CENTER) > 0.13
                    and np.linalg.norm(p - MARKER_CENTER) > 0.13):
                break

        self.d.qpos[9:12] = [p[0], p[1], 0.03]
        self.d.qpos[12:16] = [1, 0, 0, 0]
        self.d.ctrl[:] = self.home_ctrl
        mujoco.mj_forward(self.m, self.d)
        for _ in range(40):
            self.d.ctrl[:] = self.home_ctrl
            mujoco.mj_step(self.m, self.d)
        self.lifted = False
        self.max_lift = 0.0
        self._i_err[:] = 0.0
        return self.get_obs()

    def _track_lift(self):
        if self.cube_pos()[2] > 0.055:
            self.lifted = True
        self.max_lift = max(self.max_lift, float(self.cube_pos()[2]))

    def get_obs(self):
        imgs = []
        for _, cam in self._cams:
            if cam is self._cams[1][1]:
                cam.lookat[:] = self.tcp_pos()
            self.renderer.update_scene(self.d, camera=cam)
            imgs.append(self.renderer.render().copy())
        return {"images": np.stack(imgs, 0), "state": self.joint_state()}

    # ------------------------------------------------------------- success
    def success(self):
        c = self.cube_pos()
        xy = c[:2]
        grip_open = self.d.qpos[7] > 0.030
        if self.task == "marker":
            return bool(np.linalg.norm(xy - MARKER_CENTER) < 0.055 and c[2] < 0.06
                        and self.lifted and grip_open)
        if self.task == "plate":
            return bool(np.linalg.norm(xy - PLATE_CENTER) < 0.05 and c[2] < 0.06
                        and self.lifted and grip_open)
        if self.task == "bin":
            inside = abs(xy[0] - BIN_CENTER[0]) < BIN_HALF and abs(xy[1] - BIN_CENTER[1]) < BIN_HALF
            return bool(inside and c[2] < 0.07 and self.lifted and grip_open)
        raise ValueError(self.task)

    # ------------------------------------------------------------- video
    def init_video(self, w=400, h=300, which=0):
        self.video_renderer = mujoco.Renderer(self.m, h, w)
        self._video_cam = self._cams[which][1]
        return self.video_renderer

    def render_frame(self):
        if self._video_cam is self._cams[1][1]:
            self._video_cam.lookat[:] = self.tcp_pos()
        self.video_renderer.update_scene(self.d, camera=self._video_cam)
        return self.video_renderer.render().copy()
