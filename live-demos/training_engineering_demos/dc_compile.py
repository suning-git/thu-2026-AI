"""dc_compile.py — torch.compile 做了什么,什么时候白做(讲义 §5)。

同一个模型,eager vs compile,各测:稳定后的每步时间、峰值显存、首步编译耗时。
然后往 forward 里塞一个 graph break(取一个标量并据此走 python 分支),重测 compile——
看收益缩水。

预言:
  1. compile 首步显著慢(一次性编译);稳定后每步时间 ≤ eager(融合)。
  2. compile 峰值显存 ≤ eager(算子融合让中间张量不必整块写进显存)。

(graph break 的演示见 d_graphbreak.py —— break 必须发生在被编译的函数内部才有效,
 无法从训练循环外部注入,所以单独做一个能控制 forward 的自包含 demo。)

运行:  CUDA_VISIBLE_DEVICES=0 .venv/bin/python projects/training_engineering_demos/dc_compile.py
"""
import time
import torch

from harness import build, gpu_banner, gb

gpu_banner()

DEPTH = 6
WARM = 3       # 稳定前跳过的步数
MEAS = 10      # 计时步数


def run(overrides, tag=""):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    h = build(overrides + [f"model.depth={DEPTH}", "checkpoint.enabled=false",
                           "evaluation.text.enabled=false", "max_steps=100"])
    system, opts, dl = h["system"], h["optimizers"], h["dataloader"]
    it = iter(dl)
    autocast = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

    def step():
        batch = next(it)
        with autocast:
            loss = system.loss(batch)
        loss.backward()
        for opt in opts:
            opt.step()
        system.zero_grad(set_to_none=True)
        return loss

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    step()                                       # 首步(compile 在这里编译)
    torch.cuda.synchronize()
    first_ms = (time.perf_counter() - t0) * 1e3

    for _ in range(WARM):
        step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(MEAS):
        step()
    torch.cuda.synchronize()
    steady_ms = (time.perf_counter() - t0) * 1e3 / MEAS
    peak = torch.cuda.max_memory_allocated()
    del system, opts, dl, it
    torch.cuda.empty_cache()
    print(f"  {tag:<10} 首步 {first_ms:8.0f} ms | 稳定 {steady_ms:7.1f} ms/步 | 峰值 {gb(peak):5.2f} GB")
    return first_ms, steady_ms, peak


print(f"\n模型 depth={DEPTH}。")
print("预言: compile 首步慢(编译);稳定后更快、峰值更低。")
print("实测:")
e_first, e_steady, e_peak = run(["use_compile=false"], tag="eager")
c_first, c_steady, c_peak = run(["use_compile=true"], tag="compile")

print("\n小结(比值):")
print(f"  compile 稳定加速     eager/compile = {e_steady/c_steady:.2f}×")
print(f"  compile 省显存       eager/compile = {e_peak/c_peak:.2f}×  ({gb(e_peak):.2f}→{gb(c_peak):.2f} GB)")
print(f"  首步编译代价         compile/eager 首步 = {c_first/e_first:.1f}×(一次性)")
print("\n读法: 先用 eager 把正确性调通,再加 compile,前后各测一次速度和显存——把它当需要验证的优化,不当信仰。")
print("      graph break 会毁掉这里的收益,其机制见 d_graphbreak.py。")


# ── 实测输出 ──────────────────────────────────────────────────────────────
# 机器:2×RTX 5090(单卡 32 GB)· 2026-07-21 · 从 repo 根目录运行。装配日志已略。
# 数字随机器/负载浮动,看比值不看绝对值(见 README 契约)。
#
#   [GPU] NVIDIA GeForce RTX 5090
#
#   模型 depth=6。
#   预言: compile 首步慢(编译);稳定后更快、峰值更低。
#   实测:
#     eager      首步      527 ms | 稳定    21.4 ms/步 | 峰值  1.60 GB
#     compile    首步     4170 ms | 稳定    17.9 ms/步 | 峰值  1.10 GB
#
#   小结(比值):
#     compile 稳定加速     eager/compile = 1.19×
#     compile 省显存       eager/compile = 1.46×  (1.60→1.10 GB)
#     首步编译代价         compile/eager 首步 = 7.9×(一次性)
#
#   读法: 先用 eager 把正确性调通,再加 compile,前后各测一次速度和显存——把它当需要
#         验证的优化,不当信仰。graph break 会毁掉这里的收益,其机制见 d_graphbreak.py。
# ──────────────────────────────────────────────────────────────────────────
