# pi05_mujoco_lab —— 本机 MuJoCo 上的 mini-VLA 落地实验

在本机（RTX 3050 Ti Laptop / 4GB 显存）上跑不通原版 π0.5，于是按 π0.5 的结构骨架缩比，
自建 LIBERO 风格 MuJoCo 环境 + 脚本专家示教 + 从零训练一个小 VLA，跑通闭环。

完整结论与数据见 **`pi05_本机落地报告.md`**。

## 目录结构

```
pi05_mujoco_lab/
├── assets/panda/            # Franka Panda (mujoco_menagerie) + 自建场景
│   ├── panda.xml            # keyframe qpos 已补成 16 元（含方块 freejoint）
│   └── scene_pi05.xml       # 实际使用：Panda + 蓝箱 / 黄盘 / 绿标 + 4.8cm 方块
├── src/
│   ├── mujoco_env.py        # PandaLab：20Hz、双相机、8 维动作、DLS IK 伺服、成功判定
│   ├── gen_demos.py         # 脚本化专家，采集示教 -> data/*.npz
│   ├── prep_data.py         # npz -> memmap .npy，并生成"绝对关节目标"动作
│   ├── mini_vla.py          # mini-VLA 模型（CNN patch 编码 + Transformer + Flow Matching 动作专家）
│   ├── train_bc.py          # 条件流匹配行为克隆训练
│   ├── eval_policy.py       # 闭环评测 + 录像
│   └── _diag_*.py           # 调试期间用的探针/诊断脚本（保留作证据）
├── data/                    # 示教数据（npz + npy）
├── checkpoints/             # 模型权重
└── outputs/                 # 日志、视频、评测报告
```

## 三个任务

| task | 指令 | 目标 |
|---|---|---|
| bin | "put the red cube into the blue bin" | 放进蓝箱 |
| plate | "put the red cube on the yellow plate" | 放到黄盘上 |
| marker | "put the red cube on the green target" | 放到绿标上 |

三任务共用同一场景与动作空间，**只有指令和目标不同**。

## 跑法

```powershell
$py = "C:\Users\12698\AppData\Local\Programs\Python\Python311\python.exe"
cd src

& $py gen_demos.py --episodes 100 --img 112   # 1) 采数据（本机约 320s）
& $py prep_data.py                            # 2) 转 npy
& $py train_bc.py --epochs 25 --batch 32 --img 112 --tag mini_vla_v2   # 3) 训练
& $py eval_policy.py --episodes 20 --video 2  # 4) 闭环评测
```

## 关键观测/动作口径

- 观测：`state` = 过去 4 帧关节状态（4×8=32 维）+ 归一化任务进度 `t_norm` → 33 维；图像 2 路（front / wrist）112×112
- 动作：8 维 = 7 关节**绝对**位置目标 + 夹爪指令（0=闭合，255=全开），20Hz
- 模型：3.42M 参数，chunk=8，推理 8 步 Euler 积分，每次只执行前 6 步
- 训练显存峰值 0.30GB

## 实测结果

| 版本 | 视觉主干 | 动作 | 离线误差 | 闭环成功率 |
|---|---|---|---|---|
| v1 | 4 层 stride-16，84px | 相对增量 | 中等 | **28.3%**（bin 25 / plate 45 / marker 15） |
| v2 | 4 层 stride-16，112px | 绝对关节 + 4 帧历史 | **2–5 mrad** | 0%（抓空，见下） |
| v3 | stride-4，112px（28×28 patch） | 绝对关节 + 4 帧历史 | 训练中 | 评测中 |

**v2 的关键教训**：离线 teacher-forcing 误差只有 2–5 mrad，但闭环 0%——因为视觉主干总 stride=16，
特征格子约 15cm，而方块在 112px 图上只有 ~5px，位置信息在下采样里被抹掉了。
策略于是退化成"复现平均轨迹"：下降点离方块偏 4cm、夹爪夹了个空，方块全程没动。
v3 把主干总 stride 降到 4（单格 3.7cm），让模型真正能看见方块在哪。

详细复盘见 `pi05_本机落地报告.md`。

