"""d3_logits_chain.py — 最后一层吃掉半张卡;融合交叉熵把它压回去(讲义 §2.4 / §5)。

语言模型最后一步:hidden [N, d] 经 lm_head 投到 logits [N, V],再算交叉熵。logits 这个
张量的形状是 N×V —— N=B·T 个 token 乘上词表大小 V,常常是整个 step 里最大的张量。
朴素实现把它完整算出来、整块存进显存(还要转成 fp32、留下 softmax 的中间量),峰值显存被它主导。
融合交叉熵(Liger 的 fused linear cross-entropy)分块算、logits 从不完整写进显存,峰值大幅下降。

预言:① 朴素峰值 ≈ 几个 N×V 张量(bf16 logits + 转 fp32 的副本 + CE 的 log_softmax 中间量),
         即 N×V×4 的数倍,随 V 线性膨胀;② 融合峰值 ≪ 朴素(logits 不整块存下);
      ③ 两者 loss 数值一致(只省显存,不改结果)。

运行:  CUDA_VISIBLE_DEVICES=0 .venv/bin/python projects/training_engineering_demos/d3_logits_chain.py
"""
import torch
import torch.nn.functional as F
from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss

assert torch.cuda.is_available(), "需要 GPU"
DEV = "cuda"
print(f"[GPU] {torch.cuda.get_device_name(0)}")


def gb(n):
    return n / 1024 ** 3


N, D, V = 8192, 1024, 131072      # N=B·T 个 token,d 隐藏维,V 词表(取大,让 logits 张量显形)


def fresh():
    # 固定种子:朴素与融合两次跑用**同一份**输入,loss 才可比。
    g = torch.Generator(device=DEV).manual_seed(0)
    h = torch.randn(N, D, device=DEV, dtype=torch.bfloat16, generator=g).requires_grad_(True)
    w = (torch.randn(V, D, device=DEV, dtype=torch.bfloat16, generator=g) / D ** 0.5).requires_grad_(True)
    t = torch.randint(0, V, (N,), device=DEV, generator=g)
    return h, w, t


def naive(h, w, t):
    logits = (h @ w.t()).float()          # 完整算出 [N, V] fp32 并存进显存 —— 这一步就是显存大头
    loss = F.cross_entropy(logits, t)
    loss.backward()
    return loss.detach()


def fused(h, w, t):
    loss = LigerFusedLinearCrossEntropyLoss()(w, h, t)   # logits 从不完整写进显存
    loss.backward()
    return loss.detach()


def peak_of(fn):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    h, w, t = fresh()
    base = torch.cuda.memory_allocated()          # 输入(h,w,t)的常驻,两者相同,扣掉后比链本身
    loss = fn(h, w, t)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    del h, w, t
    torch.cuda.empty_cache()
    return peak, peak - base, float(loss)


logits_bytes = N * V * 4      # 预测:朴素 logits 张量 fp32
print(f"\nN={N}(=B·T), d={D}, V={V:,}")
print(f"预言: 朴素峰值 ≈ logits 张量 N×V×4 = {gb(logits_bytes):.2f} GB 量级;融合峰值 ≪ 之;loss 一致。")
print("实测:")
p_naive, chain_naive, loss_naive = peak_of(naive)
p_fused, chain_fused, loss_fused = peak_of(fused)
print(f"  朴素   峰值 {gb(p_naive):6.2f} GB  (链上净增 {gb(chain_naive):.2f} GB)  loss={loss_naive:.4f}")
print(f"  融合   峰值 {gb(p_fused):6.2f} GB  (链上净增 {gb(chain_fused):.2f} GB)  loss={loss_fused:.4f}")
print(f"\n  省显存  朴素/融合 峰值 = {p_naive / p_fused:.1f}×")
print(f"  loss 差 = {abs(loss_naive - loss_fused):.2e}  (应≈0,只省显存不改结果)")
print("\n读法: B·T·V 这个乘积里 V 常是最大因子——词表大小是显存问题。这也是 fused CE 存在的理由。")
print("      真实模型里 NanoInfra 的 head 默认已用 fused CE(见 core/model/heads.py)。")


# ── 实测输出 ──────────────────────────────────────────────────────────────
# 机器:2×RTX 5090(单卡 32 GB)· 2026-07-21 · 从 repo 根目录运行。
# 数字随机器/负载浮动,看比值不看绝对值(见 README 契约)。
#
#   [GPU] NVIDIA GeForce RTX 5090
#
#   N=8192(=B·T), d=1024, V=131,072
#   预言: 朴素峰值 ≈ logits 张量 N×V×4 = 4.00 GB 量级;融合峰值 ≪ 之;loss 一致。
#   实测:
#     朴素   峰值  16.30 GB  (链上净增 16.03 GB)  loss=12.2697
#     融合   峰值   1.36 GB  (链上净增 1.03 GB)  loss=12.2697
#
#     省显存  朴素/融合 峰值 = 12.0×
#     loss 差 = 0.00e+00  (应≈0,只省显存不改结果)
#
#   读法: B·T·V 这个乘积里 V 常是最大因子——词表大小是显存问题。这也是 fused CE 存在的理由。
#         真实模型里 NanoInfra 的 head 默认已用 fused CE(见 core/model/heads.py)。
# ──────────────────────────────────────────────────────────────────────────
