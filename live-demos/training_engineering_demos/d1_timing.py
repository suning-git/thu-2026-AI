"""d1_timing.py — 计时的三种方式,其中两种在骗你(讲义 §3.3)。

同一段 GPU 计算,用三种方式计时:
  裸计时           —— 测的是 CPU 把 kernel 提交进队列的时间(微秒级),与 GPU 无关。错。
  synchronize 围栏 —— 强制等队列排空,测到 GPU 真实执行时间。对。
  CUDA events      —— 在 GPU 时间线上打点,也对,且不阻塞 CPU。

预言(比值,与具体 GPU 无关):裸计时 ≪ 后两者(小几十到几百倍);后两者互相接近。

运行:  CUDA_VISIBLE_DEVICES=0 .venv/bin/python projects/training_engineering_demos/d1_timing.py
"""
import time
import torch

assert torch.cuda.is_available(), "需要 GPU"
DEV = "cuda"
print(f"[GPU] {torch.cuda.get_device_name(0)}")


def work():
    """一段真有分量的 GPU 计算:200 次大矩阵乘链。"""
    a = torch.randn(4096, 4096, device=DEV)
    b = torch.randn(4096, 4096, device=DEV)
    x = a
    for i in range(200):
        x = x @ (b if i % 2 == 0 else a)
    return x


# 热身:第一次 GPU 操作带一次性的 context/cuBLAS 初始化开销(~百 ms),不计入。
torch.cuda.synchronize()
work()
torch.cuda.synchronize()

# 方式一:裸计时(错)——不 synchronize,time.time 夹住
t0 = time.perf_counter()
y = work()
naive_ms = (time.perf_counter() - t0) * 1e3

# 方式二:synchronize 围栏(对)
torch.cuda.synchronize()
t0 = time.perf_counter()
y = work()
torch.cuda.synchronize()
sync_ms = (time.perf_counter() - t0) * 1e3

# 方式三:CUDA events(对,不阻塞 CPU)
start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
start.record()
y = work()
end.record()
torch.cuda.synchronize()
event_ms = start.elapsed_time(end)

print("\n预言: 裸计时 ≪ synced ≈ events;裸计时只是 CPU 提交 kernel 的时间。")
print("实测:")
print(f"  裸计时            {naive_ms:9.2f} ms")
print(f"  synchronize 围栏   {sync_ms:9.2f} ms")
print(f"  CUDA events       {event_ms:9.2f} ms")
print(f"\n  synced / 裸计时 = {sync_ms / naive_ms:6.1f}×   (裸计时把 GPU 时间少报了这么多倍)")
print(f"  events / synced = {event_ms / sync_ms:6.3f}   (两个正确方法应接近 1)")
print("\n读法: 任何一份优化报告,若计时代码里没有 synchronize/events,其结论可直接丢弃。")


# ── 实测输出 ──────────────────────────────────────────────────────────────
# 机器:2×RTX 5090(单卡 32 GB,无 NVLink)· 2026-07-21 · 从 repo 根目录运行。
# 数字随机器/负载浮动,看比值不看绝对值(见 README 契约)。
#
#   [GPU] NVIDIA GeForce RTX 5090
#
#   预言: 裸计时 ≪ synced ≈ events;裸计时只是 CPU 提交 kernel 的时间。
#   实测:
#     裸计时                 1.14 ms
#     synchronize 围栏      406.51 ms
#     CUDA events          406.76 ms
#
#     synced / 裸计时 =  355.3×   (裸计时把 GPU 时间少报了这么多倍)
#     events / synced =  1.001   (两个正确方法应接近 1)
#
#   读法: 任何一份优化报告,若计时代码里没有 synchronize/events,其结论可直接丢弃。
# ──────────────────────────────────────────────────────────────────────────
