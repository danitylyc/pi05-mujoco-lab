"""mini-VLA：按 pi0.5 的结构骨架缩比实现的本地视觉-语言-动作模型。

结构对应关系：
  pi0.5                       本实现
  SigLIP/Gemma 视觉主干   ->  小型 CNN patch encoder（每路相机独立编码）
  Gemma 语言主干          ->  word-level 文本 embedding + 共享 Transformer
  本体状态投影            ->  MLP token
  Flow Matching 动作专家  ->  cross-attention 动作专家 + 条件流匹配损失
"""
from __future__ import annotations

import json
import math

import numpy as np
import torch
import torch.nn as nn


# --------------------------------------------------------------------- text
def build_vocab(instructions):
    words = set()
    for s in instructions:
        words.update(s.lower().split())
    vocab = {"<pad>": 0}
    for w in sorted(words):
        vocab[w] = len(vocab)
    return vocab


def encode_text(texts, vocab, max_len=None):
    ids = []
    for t in texts:
        toks = [vocab.get(w, 0) for w in t.lower().split()]
        ids.append(toks)
    L = max_len or max(len(x) for x in ids)
    out = np.zeros((len(ids), L), dtype=np.int64)
    for i, x in enumerate(ids):
        out[i, :len(x)] = x[:L]
    return out


# -------------------------------------------------------------------- model
def sinusoid_embed(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t[:, None] * freqs[None, :]
    return torch.cat([torch.sin(args), torch.cos(args)], -1)


class Block(nn.Module):
    def __init__(self, d, n_head, dropout=0.0):
        super().__init__()
        self.n1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_head, dropout=dropout, batch_first=True)
        self.n2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        h = self.n1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.ff(self.n2(x))


class CrossBlock(nn.Module):
    def __init__(self, d, n_head, dropout=0.0):
        super().__init__()
        self.n1 = nn.LayerNorm(d)
        self.self_attn = nn.MultiheadAttention(d, n_head, dropout=dropout, batch_first=True)
        self.n2 = nn.LayerNorm(d)
        self.cross_attn = nn.MultiheadAttention(d, n_head, dropout=dropout, batch_first=True)
        self.n3 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, memory):
        h = self.n1(x)
        x = x + self.self_attn(h, h, h, need_weights=False)[0]
        x = x + self.cross_attn(self.n2(x), memory, memory, need_weights=False)[0]
        return x + self.ff(self.n3(x))


class MiniVLA(nn.Module):
    def __init__(self, vocab_size, n_cam=2, state_dim=8, act_dim=8,
                 chunk=8, d=192, n_head=6, n_layer=4, n_act_layer=2, img=84):
        super().__init__()
        self.chunk, self.act_dim = chunk, act_dim

        # 关键：总 stride 只做到 4。方块在 112px 图上只有 ~5px，stride 16 会把位置信息抹平，
        # 策略就退化成"复现平均轨迹"、抓空。stride 4 -> 28x28 网格，单格约 3.7cm。
        self.vision = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.GroupNorm(8, 32), nn.SiLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.GroupNorm(8, 64), nn.SiLU(),
            nn.Conv2d(64, 96, 3, 1, 1), nn.GroupNorm(8, 96), nn.SiLU(),
            nn.Conv2d(96, 96, 3, 1, 1), nn.GroupNorm(8, 96), nn.SiLU(),
        )
        patches = self._patch_grid(img)
        self.n_patch = patches * patches
        self.patch_proj = nn.Conv2d(96, d, 1)
        self.patch_pos = nn.Parameter(torch.zeros(self.n_patch, d))
        self.cam_emb = nn.Parameter(torch.zeros(n_cam, d))

        self.txt_emb = nn.Embedding(vocab_size, d, padding_idx=0)
        self.state_mlp = nn.Sequential(nn.Linear(state_dim, d), nn.SiLU(), nn.Linear(d, d))

        self.blocks = nn.ModuleList([Block(d, n_head) for _ in range(n_layer)])

        self.time_mlp = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        self.act_in = nn.Linear(act_dim, d)
        self.pos_chunk = nn.Parameter(torch.zeros(chunk, d))
        self.act_blocks = nn.ModuleList([CrossBlock(d, n_head) for _ in range(n_act_layer)])
        self.act_out = nn.Linear(d, act_dim)

        nn.init.trunc_normal_(self.patch_pos, std=0.02)
        nn.init.trunc_normal_(self.pos_chunk, std=0.02)
        nn.init.trunc_normal_(self.cam_emb, std=0.02)

    @staticmethod
    def _patch_grid(img):
        """与上面 2 层 stride-2 卷积对应的特征网格边长（112 -> 28）。"""
        return img // 4

    def encode(self, images, state, txt_ids):
        B, N, C, H, W = images.shape
        f = self.vision(images.reshape(B * N, C, H, W))                 # (B*N,96,h,w)
        f = self.patch_proj(f).flatten(2).transpose(1, 2)                # (B*N,P,d)
        f = f + self.patch_pos[None] + self.cam_emb.repeat_interleave(B, 0)[:, None]
        img_tok = f.reshape(B, N * self.n_patch, -1)

        txt_tok = self.txt_emb(txt_ids)
        st_tok = self.state_mlp(state)[:, None, :]
        x = torch.cat([txt_tok, st_tok, img_tok], 1)
        for blk in self.blocks:
            x = blk(x)
        return x

    def flow_loss(self, memory, action, noise=None):
        B = action.shape[0]
        noise = torch.randn_like(action) if noise is None else noise
        t = torch.rand(B, device=action.device)
        a_t = (1 - t)[:, None, None] * noise + t[:, None, None] * action
        target = action - noise

        h = self.act_in(a_t) + self.pos_chunk[None]
        t_emb = self.time_mlp(sinusoid_embed(t, h.shape[-1]))
        h = h + t_emb[:, None, :]
        for blk in self.act_blocks:
            h = blk(h, memory)
        v = self.act_out(h)
        return ((v - target) ** 2).mean()

    @torch.no_grad()
    def sample(self, memory, steps=8):
        B = memory.shape[0]
        x = torch.randn(B, self.chunk, self.act_dim, device=memory.device)
        dt = 1.0 / steps
        for k in range(steps):
            t = torch.full((B,), k * dt, device=memory.device)
            h = self.act_in(x) + self.pos_chunk[None]
            h = h + self.time_mlp(sinusoid_embed(t, h.shape[-1]))[:, None, :]
            for blk in self.act_blocks:
                h = blk(h, memory)
            x = x + dt * self.act_out(h)
        return x


# ------------------------------------------------------------------ helpers
class Normalizer:
    def __init__(self, mean, std):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)

    def norm(self, x):
        return (x - self.mean) / self.std

    def denorm(self, x):
        return x * self.std + self.mean

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"mean": self.mean.tolist(), "std": self.std.tolist()}, f)

    @staticmethod
    def load(path):
        with open(path, "r", encoding="utf-8") as f:
            js = json.load(f)
        return Normalizer(js["mean"], js["std"])
