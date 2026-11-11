# mini-VLA on MuJoCo · 按 π0.5 骨架缩比，从零训一个能真跑闭环的小型视觉-语言-动作模型

> **在一台装不下 π0.5 的笔记本上，把「环境 → 示教 → 训练 → 闭环评测」整条 VLA 链路自己搭了出来，并用公开 benchmark 验证了同一套骨架的可迁移性。**

π0.5 官方权重约 12 GB，本机显存 4 GB —— 下载不了、量化也塞不进。
而 π0.5 的能力来自 PaliGemma 主干的预训练，**把主干换小等于随机初始化**，
所以"一个小号的 π0.5"并不存在，只能按它的结构骨架自己训一个。
这个仓库就是这件事的全部代码、数据口径、失败复盘与可复现脚本。

```
MuJoCo 自建场景  →  脚本专家采示教  →  训练 mini-VLA  →  闭环评测 + 录像
Panda + 3 目标      8 阶段 / DLS IK      条件流匹配 BC       20 seeds / 任务
```

**当前状态**

| 轨道 | 状态 | 结果 |
|---|---|---|
| 自建环境 · v2 | ✅ 完成 | 闭环 **46.7%**（bin 15 / plate 60 / marker 65） |
| 自建环境 · v3 | 🔄 训练中 | stride-4 + 空间 soft-argmax 视觉，10 epoch |
| 主流 benchmark · Fetch | 🟡 已训待评 | 专家 100%（20/20），mini-VLA 10 epoch / loss 0.145 |

---

## 为什么不是直接跑 π0.5

| 事实 | 数值 |
|---|---|
| π0.5 官方 checkpoint | ≈ 12 GB（bf16，单分片最大 2.3 GB） |
| 本机显存 | 4 GB（RTX 3050 Ti Laptop） |
| int8 量化后 | ≈ 6 GB，仍装不下；int4 勉强装权重，激活 + KV cache 立刻溢出 |
| 本机 C 盘剩余 | 6.6 GB，连落盘都不够 |

硬件这条线断掉之后，能做的选择只有两个：放弃，或者**照骨架缩比、从零训**。选了后者。

---

## 结果

| 版本 | 视觉主干 | 动作表示 | 离线 teacher-forcing | 闭环成功率（20 seeds/任务） |
|---|---|---|---|---|
| v1 | 4 层 stride-16 @84px | 关节相对增量 | — | **28.3%**（bin 25 / plate 45 / marker 15） |
| v2 | 4 层 stride-16 @112px | 绝对关节 + 4 帧历史 | **2–5 mrad（≈1 mm）** | **46.7%**（bin 15 / plate 60 / marker 65） |
| v3 | stride-4 + soft-argmax @112px | 绝对关节 + 4 帧历史 | 训练中（loss 0.081） | 待评 |

v2 平均末端距离：bin 26.79 cm / plate 6.92 cm / marker 7.82 cm。
bin 最难——它是唯一需要**抬起越过箱沿再投放**的任务，最吃绝对定位精度。

模型规模：**3.32M 参数（v2）/ 3.35M（v3）**，训练 25 epoch / 34.5 分钟，**显存峰值 0.30 GB**。

---

## 模型：π0.5 骨架的缩比映射

| π0.5 组件 | 本实现 |
|---|---|
| PaliGemma 视觉语言主干（3B） | CNN patch encoder（每路相机 5×5 patch；v3 另加空间 soft-argmax 关键点）+ 16 词表文本 embedding + **4 层 Transformer** |
| 本体状态投影 | 过去 **4 帧**关节状态（4×8=32）+ 归一化任务进度 `t_norm` → **33 维** |
| Flow Matching 动作专家 | **2 层 cross-attention**（attend 主干 memory），flow timestep 用 sinusoidal + MLP 做 adaLN 式注入 |
| 条件流匹配损失 | `a_t = (1−t)·z0 + t·a_gt`，回归 `a_gt − z0` |
| 推理 | **8 步 Euler 积分** |
| 动作块 / RTC | chunk = 8，**每次只执行前 6 步**（receding horizon） |

动作口径与 π0.5 的 LIBERO 设定对齐：**8 维 = 7 个关节绝对位置目标 + 夹爪指令（0=闭合，255=全开）**，20 Hz。

---

## 一个关键发现：stride 决定 VLA 能不能"看见"

v2 出现过最反直觉的一幕：**离线拟合精确到 2–5 mrad，闭环却大面积失败**。
逐段追踪之后发现臂在下降时停在 `(0.44, 0.00)`，而方块在 `(0.412, −0.038)` ——
**偏 4 cm，夹爪夹了个空**，方块全程纹丝不动。

根因不是过拟合，是**信息在下采样里被物理性地删掉了**：视觉主干总 stride = 16，
特征格子 ≈ 15 cm，而方块在 112 px 图上只有 **5 px**。策略于是退化成
"复现平均轨迹 + 用 `t_norm` 对齐相位"。

验证：固定 t=34 取 10 个不同 episode，看预测是否随视觉变化 —— 跨 episode 标准差 0.02–0.17，
说明视觉**确实在用**，但精度不够。

修法不是硬堆注意力（stride=4 的 1568 token 一个 epoch 要 11 分钟，跑不动），而是
**在 stride-4 的 28×28 特征图上做空间 soft-argmax**：对每个通道沿空间 softmax 求期望 (x, y)，
压成 1 个关键点 token。**定位精度来自 28×28 网格（单格 3.7 cm），注意力成本几乎不变**（2.6 ms/样本）。

---

## 快速开始

```powershell
$py = "C:\Users\12698\AppData\Local\Programs\Python\Python311\python.exe"
cd src

& $py gen_demos.py --episodes 100 --img 112   # 1) 采示教（约 320 s）
& $py prep_data.py                            # 2) npz → memmap npy + 绝对关节动作
& $py train_bc.py --epochs 25 --batch 32 --img 112 --tag mini_vla_v2   # 3) 训练
& $py eval_policy.py --episodes 20 --video 2  # 4) 闭环评测 + 录像
```

依赖：`mujoco`、`torch`（CUDA）、`numpy`、`opencv-python-headless`、`tqdm`；
跑主流 benchmark 另需 `gymnasium[mujoco]==1.2.1` + `gymnasium-robotics`。
渲染需 `MUJOCO_GL=glfw`。数据与权重不入库（体积），按上面四步重新生成即可。

```powershell
# 主流 benchmark（FetchPickAndPlace-v4，注意 v3 已弃用）
& $py bench_fetch.py                 # 接口 / 渲染 / 推理耗时验证
& $py fetch_demos.py                 # 脚本专家：20/20 = 100%，65 ms/episode
& $py fetch_pipeline.py              # 采示教 → 训练 mini-VLA（2.08M 参数，10 epoch / loss 0.145）
& $py fetch_eval.py                  # 闭环评测（已写好，待跑）
```

---

## 目录

```
pi05_mujoco_lab/
├── README.md                   # 本文件
├── 项目描述.md                  # 工作量 + 算法 + 结果（对外介绍用）
├── 项目简介.md                  # 更短的一页版介绍
├── pi05_本机落地报告.md         # 完整技术复盘（11 节）
├── assets/panda/
│   ├── panda.xml               # menagerie 资产（keyframe qpos 补成 16 元，含方块 freejoint）
│   └── scene_pi05.xml          # 实际场景：Panda + 蓝箱 / 黄盘 / 绿标 + 4.8cm 方块
├── src/
│   ├── mujoco_env.py           # PandaLab：20Hz、双相机、8 维动作、DLS IK 伺服、成功判定
│   ├── gen_demos.py            # 8 阶段脚本专家，采集示教
│   ├── prep_data.py            # npz → memmap npy + actions_abs
│   ├── mini_vla.py             # CNN patch 编码 + Transformer + Flow Matching 动作专家
│   ├── train_bc.py             # 条件流匹配行为克隆
│   ├── eval_policy.py          # 闭环评测 + 录像
│   ├── bench_fetch.py          # Fetch 接口 / 耗时验证
│   ├── fetch_demos.py          # Fetch 脚本专家
│   ├── fetch_pipeline.py       # Fetch：采示教 → 训练
│   ├── fetch_eval.py           # Fetch：闭环评测
│   └── _diag_*.py, _probe_*.py # 调试期诊断脚本（全部保留作证据）
├── data/                       # 示教数据（.gitignore）
├── checkpoints/                # 权重（.gitignore）
└── outputs/                    # 日志 / 评测报告 / 录像
```

三个任务共用同一场景与动作空间，**只有指令和目标不同**：

| task | 指令 | 目标 |
|---|---|---|
| bin | "put the red cube into the blue bin" | 放进蓝箱 |
| plate | "put the red cube on the yellow plate" | 放到黄盘上 |
| marker | "put the red cube on the green target" | 放到绿标上 |

---

## 边界（别过度解读）

- **它不是 π0.5**：3.3M vs 3B+，300 条示教 vs 10k+ 小时，能力不可比。它证明的是"结构可以照搬、闭环能真跑通"。
- 语言是玩具级（16 词表、模板句），验证的是"同一观测下指令决定目标"，不是开放词汇泛化。
- `t_norm` 是时间索引，换任务时长会退化（所以同时叠了 4 帧状态历史）。
- 没做 RTC（实时 chunk 延迟补偿）；本机推理快到不需要，模型放大后必须加。
- Fetch 链路里观测含物体位置（benchmark 原生给出），属于该 benchmark 的常规设定，不是"纯像素"设定。

---

## 延伸阅读

- **`项目描述.md`** —— 工作量与算法拆解（含 5 次定位到根因的返工）
- **`项目简介.md`** —— 一页纸速览
- **`pi05_本机落地报告.md`** —— 11 节完整复盘：可行性论证、环境设计、示教、训练、评测、失败分析
- 上游参考：π0.5（openpi，Apache-2.0）、π0.6 RECAP、π0.7 五大上下文与制导涌现
