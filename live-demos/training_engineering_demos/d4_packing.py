"""d4_packing.py — 变长数据:padding 浪费的算力,packing 省回来(讲义 §4)。

变长序列组 batch,最朴素的办法是补齐(padding)到 batch 内最长——补出来的 pad 位置
照样过一遍网络,算力白费。浪费率 = 1 − 平均长度 / 补齐长度。packing 把多条序列首尾
拼接、不补齐,逐 token 的计算(投影 / MLP)一个 pad 都不算;注意力则用**块对角 mask**
让每个 token 只看自己那条序列(即 flash-attention 处理变长序列的那套做法)。

本 demo 用一个真的 Transformer 栈(注意力 + MLP),两条路径都走 FlexAttention:
  · padding:[B, MAXLEN],mask = 因果 且 kv 在真实长度内;计算覆盖 B×MAXLEN 个位置。
  · packing:[1, total],mask = 因果 且 同一文档(块对角);计算只覆盖 total 个真实 token,
             注意力也只在文档内(O(Σlenᵢ²) 而非 O(total²))。
对照两者的**有效 token 吞吐**(有效 token = 真实 token 数 / 墙钟)。

预言:packing 有效吞吐 / padding ≈ 1 / (1 − 浪费率)(逐 token 计算精确按此比;注意力
      省得更多,但此规模下逐 token 计算是大头,故总比值贴着 1/(1−浪费率))。

运行:  CUDA_VISIBLE_DEVICES=0 .venv/bin/python projects/training_engineering_demos/d4_packing.py
"""
import time
import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

assert torch.cuda.is_available(), "需要 GPU"
DEV = "cuda"
print(f"[GPU] {torch.cuda.get_device_name(0)}")

D, H, LAYERS, B, MAXLEN = 1024, 16, 8, 64, 1024
DH = D // H
flex = torch.compile(flex_attention)
torch.manual_seed(0)

# 长尾长度:多数短、少数接近 MAXLEN
u = torch.rand(B, generator=torch.Generator().manual_seed(1))
lengths = (MAXLEN * (u ** 3)).clamp(min=8).long()
total = int(lengths.sum())
waste = 1 - total / (B * MAXLEN)
lengths_dev = lengths.to(DEV)
doc = torch.repeat_interleave(torch.arange(B, device=DEV), lengths_dev)   # [total] 每个位置属于哪条序列


def pad_mask_mod(b, h, q, kv):
    return (q >= kv) & (kv < lengths_dev[b])           # 因果 且 kv 在第 b 条的真实长度内


def pack_mask_mod(b, h, q, kv):
    return (q >= kv) & (doc[q] == doc[kv])             # 因果 且 同一文档(块对角)


bm_pad = create_block_mask(pad_mask_mod, B, None, MAXLEN, MAXLEN, device=DEV)
bm_pack = create_block_mask(pack_mask_mod, 1, None, total, total, device=DEV)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(D, 3 * D)
        self.o = nn.Linear(D, D)
        self.mlp = nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D))

    def forward(self, x, bm):                            # x: [Bx, S, D]
        Bx, S, _ = x.shape
        q, k, v = self.qkv(x).split(D, dim=2)
        q, k, v = (t.view(Bx, S, H, DH).transpose(1, 2) for t in (q, k, v))  # [Bx,H,S,DH]
        a = flex(q, k, v, block_mask=bm)
        x = x + self.o(a.transpose(1, 2).reshape(Bx, S, D))
        return x + self.mlp(x)


net = nn.ModuleList(Block() for _ in range(LAYERS)).to(DEV).to(torch.bfloat16)


@torch.no_grad()
def forward(x, bm):
    for blk in net:
        x = blk(x, bm)
    return x


def timed(shape, bm, tokens_effective, warm=3, meas=10):
    x = torch.randn(*shape, device=DEV, dtype=torch.bfloat16)
    for _ in range(warm):
        forward(x, bm)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(meas):
        forward(x, bm)
    torch.cuda.synchronize()
    return tokens_effective / ((time.perf_counter() - t0) / meas)


print(f"\n{B} 条序列,补齐长 {MAXLEN};真实 token {total:,},补齐后 {B*MAXLEN:,}")
print(f"浪费率 = 1 − 平均/补齐 = {waste:.1%}")
print(f"\n预言: packing 有效吞吐 / padding ≈ 1 / (1 − 浪费率) = {1/(1-waste):.2f}×")
print("实测(有效 token / 秒,只计真实 token):")
tp_pad = timed((B, MAXLEN, D), bm_pad, total)
tp_pack = timed((1, total, D), bm_pack, total)
print(f"  padding   {tp_pad/1e6:8.1f} M tok/s   (算了 {B*MAXLEN:,} 个位置,其中 {B*MAXLEN-total:,} 是 pad)")
print(f"  packing   {tp_pack/1e6:8.1f} M tok/s   (只算 {total:,} 个真实 token,注意力只在文档内)")
print(f"\n  提速  packing / padding = {tp_pack/tp_pad:.2f}×   (预言 ≈ {1/(1-waste):.2f})")
print("\n读法: 吞吐表上 padding 的 tokens/s 若把 pad 也算进去会很好看——但有效 token 少一半。")
print("      预训练把变长文档拼成定长流,正是 packing 的极端形式,天然绕开这一切。")


# ── 实测输出 ──────────────────────────────────────────────────────────────
# 机器:2×RTX 5090(单卡 32 GB)· 2026-07-21 · 从 repo 根目录运行。
# 数字随机器/负载浮动,看比值不看绝对值(见 README 契约)。
#
#   [GPU] NVIDIA GeForce RTX 5090
#
#   64 条序列,补齐长 1024;真实 token 17,614,补齐后 65,536
#   浪费率 = 1 − 平均/补齐 = 73.1%
#
#   预言: packing 有效吞吐 / padding ≈ 1 / (1 − 浪费率) = 3.72×
#   实测(有效 token / 秒,只计真实 token):
#     padding        0.2 M tok/s   (算了 65,536 个位置,其中 47,922 是 pad)
#     packing        0.8 M tok/s   (只算 17,614 个真实 token,注意力只在文档内)
#
#     提速  packing / padding = 3.68×   (预言 ≈ 3.72)
#
#   读法: 吞吐表上 padding 的 tokens/s 若把 pad 也算进去会很好看——但有效 token 少一半。
#         预训练把变长文档拼成定长流,正是 packing 的极端形式,天然绕开这一切。
# ──────────────────────────────────────────────────────────────────────────
