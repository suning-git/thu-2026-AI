"""d6_util_vs_mfu.py — GPU-Util 100% 可以只用不到 1% 算力(讲义 §3.4)。

nvidia-smi 的 GPU-Util = "采样窗口内至少有一个 kernel 在跑"的时间占比。它回答
"GPU 在不在干活",不回答"干活的效率"。

本 demo 连续做一个**内存受限**的逐元素运算(在一大块张量上反复 x = x*a + b):
每个元素只有 2 个 FLOP,但要把整块张量读一遍写一遍——GPU 一直在搬内存,util 顶满,
而按 FLOPs 折算的算力利用率(MFU)近乎零。对照:换成一个真正的大矩阵乘(算力受限),
util 同样 100%,MFU 却是几十个百分点——**util 分辨不出这两种情形**。

预言:另开终端 `nvidia-smi -l 1` 看本进程,逐元素运算阶段 util≈100% 但 MFU≪1%;
      矩阵乘阶段 util 仍≈100% 而 MFU 高两个数量级。

运行:  CUDA_VISIBLE_DEVICES=0 .venv/bin/python projects/training_engineering_demos/d6_util_vs_mfu.py
       (另开终端: nvidia-smi -l 1  ,对照两个阶段的 GPU-Util —— 都会是 ~100%)
"""
import time
import torch

assert torch.cuda.is_available(), "需要 GPU"
DEV = "cuda"
name = torch.cuda.get_device_name(0)
print(f"[GPU] {name}")

PEAK_TFLOPS = {"5090": 209.5, "H100": 989.0, "A100": 312.0, "A800": 312.0, "H20": 148.0}
peak = next((v for k, v in PEAK_TFLOPS.items() if k in name.upper()), 200.0)

N = 8192                      # 大张量 8192×8192 bf16 ≈ 128 MB
SECONDS = 10                  # 每个阶段持续这么久,给你时间去看 nvidia-smi


def run_phase(fn, flops_per_call, label):
    """连续调用 fn 约 SECONDS 秒,用 events 计 GPU 时间,返回 (调用数, MFU)。"""
    fn(); torch.cuda.synchronize()          # 热身
    n = 0
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record(); t0 = time.perf_counter()
    while time.perf_counter() - t0 < SECONDS:
        for _ in range(200):
            fn()
        n += 200
    e.record(); torch.cuda.synchronize()
    gpu_s = s.elapsed_time(e) / 1e3
    mfu = 100 * n * flops_per_call / gpu_s / peak / 1e12
    shown = (f"{mfu:.3f}%" if mfu < 95
             else f"≳100%(算力受限;参考峰值 {peak:.0f}T 为保守稠密值,故可略超)")
    print(f"  [{label}] {n:,} 次,GPU 忙 {gpu_s:.1f}s → MFU = {shown}")
    return mfu


x = torch.randn(N, N, device=DEV, dtype=torch.bfloat16)
a = torch.tensor(1.0001, device=DEV, dtype=torch.bfloat16)
b = torch.tensor(0.0001, device=DEV, dtype=torch.bfloat16)
w = torch.randn(N, N, device=DEV, dtype=torch.bfloat16)


def elementwise():
    # 内存受限:每元素 2 FLOP,却要读一遍写一遍整块张量
    x.mul_(a).add_(b)


def matmul():
    # 算力受限:N×N×N 的乘加,复用片上数据,FLOP/字节 高
    torch.matmul(x, w, out=x)


print(f"\n阶段一 · 内存受限的逐元素运算(去 nvidia-smi 看 util)—— 预言 util≈100%,MFU≪1%")
mfu_ew = run_phase(elementwise, 2 * N * N, "逐元素")

print(f"\n阶段二 · 算力受限的矩阵乘(util 同样≈100%)—— 预言 MFU 高两个数量级")
mfu_mm = run_phase(matmul, 2 * N ** 3, "矩阵乘")

print(f"\n预言: 两个阶段 nvidia-smi 的 util 都≈100%,而 MFU 相差约两个数量级。")
mm_show = f"{mfu_mm:.1f}%" if mfu_mm < 95 else "≳100%(达参考峰值)"
print(f"实测: 逐元素 MFU {mfu_ew:.3f}%   vs   矩阵乘 MFU {mm_show}   "
      f"(差 {mfu_mm/max(mfu_ew,1e-9):.0f}×)")
print("\n读法: util≈100% 只说明 GPU 没空等;它分辨不出'在搬内存'还是'在算'。")
print("      '100% 但很慢'要靠 MFU 才看得出——util 低才是数据管线饿了的信号。")


# ── 实测输出 ──────────────────────────────────────────────────────────────
# 机器:2×RTX 5090(单卡 32 GB)· 2026-07-21 · 从 repo 根目录运行。
# 数字随机器/负载浮动,看比值不看绝对值(见 README 契约)。
#
#   [GPU] NVIDIA GeForce RTX 5090
#
#   阶段一 · 内存受限的逐元素运算(去 nvidia-smi 看 util)—— 预言 util≈100%,MFU≪1%
#     [逐元素] 28,600 次,GPU 忙 10.2s → MFU = 0.179%
#
#   阶段二 · 算力受限的矩阵乘(util 同样≈100%)—— 预言 MFU 高两个数量级
#     [矩阵乘] 3,000 次,GPU 忙 15.7s → MFU = ≳100%(算力受限;参考峰值 210T 为保守稠密值,故可略超)
#
#   预言: 两个阶段 nvidia-smi 的 util 都≈100%,而 MFU 相差约两个数量级。
#   实测: 逐元素 MFU 0.179%   vs   矩阵乘 MFU ≳100%(达参考峰值)   (差 559×)
#
#   读法: util≈100% 只说明 GPU 没空等;它分辨不出'在搬内存'还是'在算'。
#         '100% 但很慢'要靠 MFU 才看得出——util 低才是数据管线饿了的信号。
# ──────────────────────────────────────────────────────────────────────────
