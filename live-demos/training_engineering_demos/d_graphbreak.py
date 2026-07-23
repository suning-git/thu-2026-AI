"""d_graphbreak.py — graph break:为什么有时 compile 了等于白 compile(讲义 §5.3)。

torch.compile 把 forward 记录成一张线性计算图,交给后端**融合**。融合的最大收益在
一长串逐元素运算上:未编译时每个都要"读全量-算-写全量"(内存受限),融合后
读一次写一次。碰到无法记录进计算图的东西——print、.item()、依赖张量值的 python 分支——
它会**断图**:融合被切碎,还可能插入 CPU-GPU 同步。断得密,收益归零。

关键:break 必须发生在**被编译的函数内部**才有效,无法从训练循环外部注入。这里用一个
自包含的、密集逐元素运算的小模块,forward 里可选地每步插一个数据依赖分支,对照:
  · 能否整图编译(`fullgraph=True`:无 break 则成功,有 break 则抛异常)—— 二元、稳的判据;
  · 稳定每步时间(融合被断图切碎后,内存受限的逐元素运算链回到未融合的慢速)。

预言:干净 forward 能整图编译、快;插了数据依赖分支的 forward 整图编译失败、显著变慢。

运行:  CUDA_VISIBLE_DEVICES=0 .venv/bin/python projects/training_engineering_demos/d_graphbreak.py
"""
import time
import torch

assert torch.cuda.is_available(), "需要 GPU"
DEV = "cuda"
print(f"[GPU] {torch.cuda.get_device_name(0)}")

N = 4096          # 张量 4096×4096 ≈ 64 MB(fp32),逐元素运算链是内存受限的
STEPS = 60        # 逐元素运算的条数;融合本应把它们并成一两个 kernel


def make_fn(do_break):
    def fn(x):
        for _ in range(STEPS):
            x = torch.sin(x) * 0.9 + 0.1        # 一条逐元素运算:未融合 = 一个内存受限的 kernel
            if do_break:
                # .item() 把张量值拉回 CPU 供 python 判断 —— 强制断图 + 一次同步
                if x.mean().item() > -1e30:
                    x = x + 0.0
        return x
    return fn


def whole_graph_ok(fn, x):
    """fullgraph=True 强制单张图:无 break 则成功,有 break 则抛异常。二元、版本无关。"""
    torch._dynamo.reset()
    try:
        torch.compile(fn, fullgraph=True)(x)
        return True
    except Exception:
        return False


def steady_ms(cfn, x, warm=5, meas=30):
    for _ in range(warm):
        cfn(x)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(meas):
        cfn(x)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e3 / meas


x = torch.randn(N, N, device=DEV)

print(f"\n张量 {N}×{N},逐元素运算链长 {STEPS}。")
print("预言: 干净 forward 能整图编译、快;插数据依赖分支的 forward 整图编译失败、显著变慢。")
print("实测:")
res = {}
for do_break in (False, True):
    ok = whole_graph_ok(make_fn(do_break), x)
    torch._dynamo.reset()
    cfn = torch.compile(make_fn(do_break))       # 常规 compile(允许断图),计稳定时间
    ms = steady_ms(cfn, x)
    res[do_break] = (ok, ms)
    tag = "插了数据依赖分支" if do_break else "干净 forward   "
    print(f"  {tag}  整图编译 = {'成功 ✓' if ok else '失败 ✗(有 break)':<14}  稳定 {ms:7.2f} ms/步")

(ok0, ms0), (ok1, ms1) = res[False], res[True]
print(f"\n  整图编译: {'成功' if ok0 else '失败'} → {'成功' if ok1 else '失败'}")
print(f"  变慢:     {ms1/ms0:.1f}×   (一条逐元素运算链本可融成一两个 kernel,断图后碎成 ~{STEPS} 个 + 同步)")
print("\n读法: 模型 forward 保持'纯张量计算'(不 print、不取标量、不做数据依赖分支),")
print("      compile 的收益才兑现。日志、监控放到 forward 外面去。")
print("      诊断: TORCH_LOGS=graph_breaks .venv/bin/python ...  或  torch._dynamo.explain(fn)(x)")


# ── 实测输出 ──────────────────────────────────────────────────────────────
# 机器:2×RTX 5090(单卡 32 GB)· 2026-07-21 · 从 repo 根目录运行。
# 数字随机器/负载浮动,看比值不看绝对值(见 README 契约)。
# 注:插分支那次,torch 会往 stderr 打印一段 "Graph break from `Tensor.item()`" 的
#     警告(正是本 demo 要触发的现象);下面只录 stdout。
#
#   [GPU] NVIDIA GeForce RTX 5090
#
#   张量 4096×4096,逐元素运算链长 60。
#   预言: 干净 forward 能整图编译、快;插数据依赖分支的 forward 整图编译失败、显著变慢。
#   实测:
#     干净 forward     整图编译 = 成功 ✓            稳定    0.61 ms/步
#     插了数据依赖分支  整图编译 = 失败 ✗(有 break)   稳定   12.89 ms/步
#
#     整图编译: 成功 → 失败
#     变慢:     21.1×   (一条逐元素运算链本可融成一两个 kernel,断图后碎成 ~60 个 + 同步)
#
#   读法: 模型 forward 保持'纯张量计算'(不 print、不取标量、不做数据依赖分支),
#         compile 的收益才兑现。日志、监控放到 forward 外面去。
#         诊断: TORCH_LOGS=graph_breaks .venv/bin/python ...  或  torch._dynamo.explain(fn)(x)
# ──────────────────────────────────────────────────────────────────────────
