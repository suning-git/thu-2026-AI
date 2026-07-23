"""d5_kv_cache.py — 不带 KV cache 的自回归生成是 O(T²)(讲义 §7.2)。

自回归 decode 每生成一个 token,注意力只需要"新 token 的 Q"和"所有前缀的 K、V"。
前缀的 K、V 之前算过且永不改变——缓存下来,每步的代价从 O(t) 降到 O(1),
生成 T 个 token 的总代价从 O(T²) 降到 O(T)。

本 demo 用一个自包含的单层因果注意力,分别测"每步重算全前缀"与"带 KV cache",
在若干个序列长度上计时,拟合耗时随长度的增长阶。

预言:不带 cache 的总耗时随生成长度约二次增长(拟合指数 ≈ 2);带 cache 约线性(≈ 1)。

运行:  CUDA_VISIBLE_DEVICES=0 .venv/bin/python projects/training_engineering_demos/d5_kv_cache.py
"""
import math
import torch
import torch.nn.functional as F

assert torch.cuda.is_available(), "需要 GPU"
DEV = "cuda"
print(f"[GPU] {torch.cuda.get_device_name(0)}")

D, H = 2048, 16            # 模型维 / 头数(取大一点,让每步计算量盖过固定的提交开销,
                          #  二次项才在这个 T 范围内显形)
DH = D // H
torch.manual_seed(0)
Wq = torch.randn(D, D, device=DEV, dtype=torch.bfloat16) / math.sqrt(D)
Wk = torch.randn(D, D, device=DEV, dtype=torch.bfloat16) / math.sqrt(D)
Wv = torch.randn(D, D, device=DEV, dtype=torch.bfloat16) / math.sqrt(D)


def proj(x, W):
    # x: [t, D] -> [H, t, DH]
    return (x @ W).view(-1, H, DH).transpose(0, 1)


@torch.no_grad()
def gen_no_cache(T):
    """每步都把整个前缀重新前向一遍——O(T²)。"""
    seq = torch.randn(1, D, device=DEV, dtype=torch.bfloat16)
    for _ in range(T):
        q, k, v = proj(seq, Wq), proj(seq, Wk), proj(seq, Wv)      # 全前缀重算
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        nxt = out.transpose(0, 1).reshape(-1, D)[-1:]              # 只取最后一步
        seq = torch.cat([seq, nxt], 0)


@torch.no_grad()
def gen_cache(T):
    """只算新 token 的 Q/K/V,K/V 追加进 cache——O(T)。"""
    x = torch.randn(1, D, device=DEV, dtype=torch.bfloat16)
    kc, vc = None, None
    for _ in range(T):
        q, k, v = proj(x, Wq), proj(x, Wk), proj(x, Wv)           # 只算 1 个新 token
        kc = k if kc is None else torch.cat([kc, k], 1)
        vc = v if vc is None else torch.cat([vc, v], 1)
        out = F.scaled_dot_product_attention(q, kc, vc)           # 新 Q × 全部 K/V
        x = out.transpose(0, 1).reshape(-1, D)[-1:]


def timed(fn, T):
    fn(8)  # 热身
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record(); fn(T); e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e)


def slope(xs, ys):
    """log-log 线性拟合的斜率 = 经验增长阶。"""
    lx = [math.log(x) for x in xs]; ly = [math.log(y) for y in ys]
    n = len(xs); mx = sum(lx) / n; my = sum(ly) / n
    return sum((a - mx) * (b - my) for a, b in zip(lx, ly)) / sum((a - mx) ** 2 for a in lx)


LENS = [256, 512, 1024, 2048]
# 全局 warmup:在最大长度上先各跑一次,把一次性的 cuBLAS/自动调优 开销从首点里挤掉。
gen_no_cache(LENS[-1]); gen_cache(LENS[-1])
torch.cuda.synchronize()

print(f"\n生成长度: {LENS}")
t_no = [timed(gen_no_cache, T) for T in LENS]
t_ca = [timed(gen_cache, T) for T in LENS]

print("\n预言: 长度每翻倍,无cache 耗时约 ×4(二次),有cache 约 ×2(线性)。")
print("实测:")
print("  T       无cache(ms)   有cache(ms)   加速")
for T, a, b in zip(LENS, t_no, t_ca):
    print(f"  {T:<6}  {a:10.1f}  {b:11.1f}   {a/b:5.1f}×")
print("\n  长度翻倍 → 耗时倍数(无cache 预言≈4,有cache 预言≈2):")
for i in range(len(LENS) - 1):
    print(f"    {LENS[i]}→{LENS[i+1]}:  无cache ×{t_no[i+1]/t_no[i]:.1f}   有cache ×{t_ca[i+1]/t_ca[i]:.1f}")
print(f"\n  log-log 斜率(整体): 无cache {slope(LENS, t_no):.2f}(→2)  有cache {slope(LENS, t_ca):.2f}(→1)")


# ── 实测输出 ──────────────────────────────────────────────────────────────
# 机器:2×RTX 5090(单卡 32 GB)· 2026-07-21 · 从 repo 根目录运行。
# 数字随机器/负载浮动,看比值不看绝对值(见 README 契约)。
#
#   [GPU] NVIDIA GeForce RTX 5090
#
#   生成长度: [256, 512, 1024, 2048]
#
#   预言: 长度每翻倍,无cache 耗时约 ×4(二次),有cache 约 ×2(线性)。
#   实测:
#     T       无cache(ms)   有cache(ms)   加速
#     256           43.8         20.8     2.1×
#     512           73.7         39.9     1.8×
#     1024         295.5         80.6     3.7×
#     2048        2150.1        164.0    13.1×
#
#     长度翻倍 → 耗时倍数(无cache 预言≈4,有cache 预言≈2):
#       256→512:  无cache ×1.7   有cache ×1.9
#       512→1024:  无cache ×4.0   有cache ×2.0
#       1024→2048:  无cache ×7.3   有cache ×2.0
#
#     log-log 斜率(整体): 无cache 1.89(→2)  有cache 1.00(→1)
# ──────────────────────────────────────────────────────────────────────────
