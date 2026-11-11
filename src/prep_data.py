"""把 data/*.npz 转成 memmap 友好的 .npy，并额外生成"绝对关节目标"动作。

动作语义（与 pi0.5 一致）= 绝对关节位置目标 + 夹爪指令：
    actions_abs[:, :7] = states[:, :7] + dq
    actions_abs[:, 7]  = ctrl_gripper / 255
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, "data")


def main():
    for task in ("bin", "plate", "marker"):
        z = np.load(os.path.join(DATA, f"{task}.npz"), allow_pickle=False)
        images = z["images"].astype(np.uint8)
        states = z["states"].astype(np.float32)
        acts = z["actions"].astype(np.float32)

        abs_acts = acts.copy()
        abs_acts[:, :7] = states[:, :7] + acts[:, :7]
        abs_acts[:, 7] = acts[:, 7]

        np.save(os.path.join(DATA, f"{task}_images.npy"), images)
        np.save(os.path.join(DATA, f"{task}_states.npy"), states)
        np.save(os.path.join(DATA, f"{task}_actions.npy"), acts)
        np.save(os.path.join(DATA, f"{task}_actions_abs.npy"), abs_acts)
        np.save(os.path.join(DATA, f"{task}_ep_idx.npy"), z["ep_idx"].astype(np.int32))
        np.save(os.path.join(DATA, f"{task}_ep_len.npy"), z["ep_len"].astype(np.int32))
        with open(os.path.join(DATA, f"{task}_meta.json"), "w", encoding="utf-8") as f:
            json.dump({"instructions": [str(x) for x in z["instructions"]],
                       "success": [bool(x) for x in z["ep_success"]]}, f)
        print(task, images.shape, "ok", flush=True)


if __name__ == "__main__":
    main()
