"""d2_memory.py — OOM 是一笔能算清的显存;激活是大头、∝B·T(讲义 §2)。

先按公式预测,再在一个真实 step 的精确时刻实测,并排。关键是**在参数、梯度、
优化器状态(m,v)四项显存同时在场的那一刻量**(反向完成、optimizer.step 已分配
m/v、尚未 zero_grad),这一刻的常驻显存才对应讲义那个每参数 8 字节。然后 batch
翻倍验证"激活 ∝ B·T"。

每参数占多少字节(纯 bf16):参数 2 + 梯度 2 + AdamW 的 m,v 各 2 = 共 8。
  · 建模后常驻 ≈ N×2(只有参数)
  · step 后 / zero_grad 前 ≈ N×8(参数+梯度+m+v 全在)← 讲义的数
  · zero_grad 后常驻 ≈ N×6(梯度被释放)
  · 峰值出现在反向(参数+梯度 4B/参数 与激活同时在场):峰 − N×4 ≈ 激活

预言:① 上面三个常驻数各自命中 N×{2,8,6};② 峰 − N×4 ≈ 激活量级 32·d·L·B·T;
      ③ batch ×2 → 参数那部分不变、激活约 ×2。

运行:  CUDA_VISIBLE_DEVICES=0 .venv/bin/python projects/training_engineering_demos/d2_memory.py
"""
import torch

from harness import build, gpu_banner, gb

gpu_banner()

DEPTH = 6                      # 小尺度,几秒装配;结论与尺度无关
BATCHES = [8, 16]             # 第二个是第一个的 2×,验证 ∝B

cur = torch.cuda.memory_allocated
peak = torch.cuda.max_memory_allocated


def measure(batch):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    h = build([f"model.depth={DEPTH}", f"device_batch_size={batch}",
               "use_compile=false", "checkpoint.enabled=false",
               "evaluation.text.enabled=false", "max_steps=5"])
    system, opts, dl = h["system"], h["optimizers"], h["dataloader"]
    d, L, T = h["gpt_config"].n_embd, h["gpt_config"].n_layer, h["seq"]
    N = sum(p.numel() for p in system.parameters())
    autocast = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    it = iter(dl)

    torch.cuda.synchronize()
    m_build = cur()                                   # 常驻 ≈ N×2

    torch.cuda.reset_peak_memory_stats()
    with autocast:
        loss = system.loss(next(it))
    loss.backward()
    torch.cuda.synchronize()
    m_bwd_cur = cur()                                 # 常驻:参数+梯度(+残留激活)
    m_bwd_peak = peak()                               # 峰值:参数+梯度+激活

    for opt in opts:
        opt.step()                                    # 分配 m,v
    torch.cuda.synchronize()
    m_allfour = cur()                                 # 常驻 ≈ N×8(参数+梯度+m+v 全在)← 讲义的数

    system.zero_grad(set_to_none=True)                # 释放梯度
    torch.cuda.synchronize()
    m_resident = cur()                                # 常驻 ≈ N×6

    act = m_bwd_peak - N * 4                           # 峰 − (参数+梯度) ≈ 激活
    act_est = 32 * d * L * batch * T                   # 激活量级预测
    out = dict(N=N, d=d, L=L, T=T, m_build=m_build, m_allfour=m_allfour,
               m_resident=m_resident, m_bwd_peak=m_bwd_peak, act=act, act_est=act_est)
    del system, opts, dl, it, loss
    torch.cuda.empty_cache()
    return out


print(f"\n模型 depth={DEPTH};序列长 T 取自配置。")
res = {}
for b in BATCHES:
    r = measure(b)
    res[b] = r
    N = r["N"]
    print(f"\n=== batch={b} ===  参数量 N={N/1e6:.1f}M  (d={r['d']}, L={r['L']}, T={r['T']})")
    print(f"  预言 / 实测 常驻显存:")
    print(f"    建模后·只有参数        N×2 = {gb(N*2):5.2f}  /  实测 {gb(r['m_build']):5.2f} GB")
    print(f"    step 后·参数+梯度+m+v   N×8 = {gb(N*8):5.2f}  /  实测 {gb(r['m_allfour']):5.2f} GB   ← 讲义的数")
    print(f"    zero_grad 后·释放梯度   N×6 = {gb(N*6):5.2f}  /  实测 {gb(r['m_resident']):5.2f} GB")
    print(f"  激活:  峰值(反向) {gb(r['m_bwd_peak']):.2f} − 参数梯度 N×4 {gb(N*4):.2f} = {gb(r['act']):.2f} GB")
    print(f"         预言量级 32·d·L·B·T = {gb(r['act_est']):.2f} GB  (系数随实现浮动,看量级)")

b1, b2 = BATCHES
a1, a2 = res[b1]["act"], res[b2]["act"]
p1, p2 = res[b1]["m_resident"], res[b2]["m_resident"]
print(f"\n预言: batch {b1}→{b2}(×{b2//b1}),参数那部分不变、激活约 ×{b2//b1}。")
print(f"实测:  参数常驻 {gb(p1):.2f} → {gb(p2):.2f} GB (应几乎不变) | "
      f"激活 {gb(a1):.2f} → {gb(a2):.2f} GB (比值 {a2/a1:.2f}×,预言 ≈ {b2/b1:.0f})")
print("\n读法: 你调大 batch 撞 OOM,爆的是激活这部分,不是参数。")


# ── 实测输出 ──────────────────────────────────────────────────────────────
# 机器:2×RTX 5090(单卡 32 GB)· 2026-07-21 · 从 repo 根目录运行(否则找不到
# outputs/tokenizer 会退化到 gpt2 而 vocab 不匹配报错)。装配日志已略。
# 数字随机器/负载浮动,看比值不看绝对值(见 README 契约)。
#
#   [GPU] NVIDIA GeForce RTX 5090
#
#   模型 depth=6;序列长 T 取自配置。
#
#   === batch=8 ===  参数量 N=35.8M  (d=384, L=6, T=512)
#     预言 / 实测 常驻显存:
#       建模后·只有参数        N×2 =  0.07  /  实测  0.07 GB
#       step 后·参数+梯度+m+v   N×8 =  0.27  /  实测  0.33 GB   ← 讲义的数
#       zero_grad 后·释放梯度   N×6 =  0.20  /  实测  0.26 GB
#     激活:  峰值(反向) 0.81 − 参数梯度 N×4 0.13 = 0.68 GB
#            预言量级 32·d·L·B·T = 0.28 GB  (系数随实现浮动,看量级)
#
#   === batch=16 ===  参数量 N=35.8M  (d=384, L=6, T=512)
#     预言 / 实测 常驻显存:
#       建模后·只有参数        N×2 =  0.07  /  实测  0.13 GB
#       step 后·参数+梯度+m+v   N×8 =  0.27  /  实测  0.33 GB   ← 讲义的数
#       zero_grad 后·释放梯度   N×6 =  0.20  /  实测  0.26 GB
#     激活:  峰值(反向) 1.47 − 参数梯度 N×4 0.13 = 1.34 GB
#            预言量级 32·d·L·B·T = 0.56 GB  (系数随实现浮动,看量级)
#
#   预言: batch 8→16(×2),参数那部分不变、激活约 ×2。
#   实测:  参数常驻 0.26 → 0.26 GB (应几乎不变) | 激活 0.68 → 1.34 GB (比值 1.97×,预言 ≈ 2)
#
#   读法: 你调大 batch 撞 OOM,爆的是激活这部分,不是参数。
# ──────────────────────────────────────────────────────────────────────────
