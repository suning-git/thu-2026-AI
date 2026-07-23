"""d8_profiler.py — 用 profiler 看 FSDP 的通信能不能被计算盖住(讲义 §6 + §3.3)。

FSDP 每层临时 all-gather 拼回完整参数、反向再 reduce-scatter 梯度。这些通信跑在
一条**独立的 CUDA 流**上,理想情况下"藏"在计算 kernel 底下(overlap):两条流在
时间上叠着跑,wall-clock 就只由计算决定,通信白赚。藏不住的那部分(exposed comm)
直接加到每步时间上——正是 d7 里双卡总吞吐不到 2× 的那笔通信税。

profiler 的用处就在这:它把 GPU 时间线摊开,一条流是计算 kernel、另一条是 NCCL
通信 kernel。两条流叠在一起 = 通信被盖住;通信那条单独露出来一段 = 没盖住。
本 demo 用 PyTorch profiler(和 core/training/trainer.py 里同一套)录几步真 FSDP:
  1. 导出 chrome trace(可在 chrome://tracing 或 https://ui.perfetto.dev 打开);
  2. 解析 trace,把 GPU kernel 分成"通信(NCCL)"与"计算"两类,算出
     · 通信被计算**覆盖的比例** = 两类时间区间的交集 / 通信总时间;
     · **露在外面的通信** = 通信总时间 − 交集(这才是拖慢 wall-clock 的部分);
  3. 画一张时间线 PNG(上排计算流、下排通信流),让 overlap 一眼可见。

预言(比值,与具体 GPU 无关):多数 all-gather/reduce-scatter 被计算盖住(覆盖率高、
      过半);但此机无 NVLink、走 PCIe,通信慢,必有一段露在外面 —— 这段 × 每秒步数
      就是 d7 观测到的通信税的来源。

运行(必须 2 卡;单卡没有跨卡通信,overlap 无从谈起):
  .venv/bin/torchrun --nproc_per_node=2 --standalone projects/training_engineering_demos/d8_profiler.py
  # DEMO_BATCH 调小每卡 batch → 计算变少、更难盖住通信,覆盖率会掉(§6.3 的 1/(B·T))
"""
import os
import json
import time

import torch

from harness import build

DEPTH = 20                 # 与 d7 同尺度(0.5B),好让这里的 overlap 解释 d7 的吞吐
PROFILE_STEPS = 3          # 录几步就够;trace 越短越好读
OUT_DIR = os.path.expandvars("./outputs/demo_traces")


# ── 区间集合运算(kernel 的 [起, 止],单位 ms)────────────────────────────────
def merge(intervals):
    """把一堆可能重叠的区间并成互不相交的有序区间,返回 (总长度, 合并后列表)。"""
    if not intervals:
        return 0.0, []
    ivs = sorted(intervals)
    out = [list(ivs[0])]
    for s, e in ivs[1:]:
        if s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return sum(e - s for s, e in out), out


def intersect_len(a, b):
    """两个已合并(互不相交、有序)区间列表的交集总长度。"""
    i = j = 0
    tot = 0.0
    while i < len(a) and j < len(b):
        s, e = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if e > s:
            tot += e - s
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return tot


def load_gpu_kernels(trace_path):
    """从 chrome trace 里取所有 GPU kernel 事件,按名字分成 通信(nccl) / 计算 两类。
    返回各自的 [(起ms, 止ms), ...]。ts/dur 原始单位是微秒。"""
    with open(trace_path) as f:
        data = json.load(f)
    events = data["traceEvents"] if isinstance(data, dict) else data
    comm, comp = [], []
    for e in events:
        if e.get("cat") != "kernel" or e.get("ph") != "X" or "dur" not in e:
            continue
        s = e["ts"] / 1e3
        iv = (s, s + e["dur"] / 1e3)
        (comm if "nccl" in e.get("name", "").lower() else comp).append(iv)
    return comm, comp


def plot_timeline(comm, comp, out_png, overlap_frac):
    """两排泳道:上排计算 kernel、下排 NCCL 通信 kernel。叠在一起 = 通信被盖住。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    t0 = min(s for s, _ in comp + comm)
    # 只画中间那一步的窗口,避免整张图挤成一团:取全窗口中段的一个切片
    span = max(e for _, e in comp + comm) - t0
    lo, hi = t0 + span * 0.40, t0 + span * 0.72

    fig, ax = plt.subplots(figsize=(12, 2.8))
    for ivs, y, color in [(comp, 1.0, "#3b7dd8"), (comm, 0.0, "#e06666")]:
        bars = [((s - t0), (e - s)) for s, e in ivs if e > lo - t0 + t0 and s < hi]
        ax.broken_barh([(s, max(w, 0.002)) for s, w in bars], (y, 0.8),
                       facecolors=color, edgecolors="none")
    ax.set_xlim((lo - t0), (hi - t0))
    ax.set_ylim(-0.2, 2.0)
    ax.set_yticks([0.4, 1.4])
    ax.set_yticklabels(["NCCL 通信", "计算 (SM)"])
    ax.set_xlabel("时间 (ms,相对录制窗口起点)")
    ax.set_title(f"FSDP GPU 时间线 — 通信被计算覆盖 {overlap_frac:.0%}"
                 f"(上下两排在时间上叠着 = 通信藏在计算底下)")
    ax.legend(handles=[Patch(color="#3b7dd8", label="计算 kernel"),
                       Patch(color="#e06666", label="NCCL 通信 kernel")],
              loc="upper right", ncol=2, fontsize=9, framealpha=0.9)
    try:
        plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "DejaVu Sans"]
    except Exception:
        pass
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def main():
    ws = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if ws < 2:
        if rank == 0:
            print("[!] 单卡没有跨卡通信,没有 overlap 可看。请用 2 卡运行(见文件头):")
            print("    .venv/bin/torchrun --nproc_per_node=2 --standalone "
                  "projects/training_engineering_demos/d8_profiler.py")
        return

    bs = int(os.environ.get("DEMO_BATCH", "12"))
    h = build([f"model.depth={DEPTH}", f"device_batch_size={bs}", "use_compile=false",
               "checkpoint.enabled=false", "evaluation.text.enabled=false", "max_steps=100"])
    system, opts, dl = h["system"], h["optimizers"], h["dataloader"]
    seq, batch = h["seq"], h["batch"]
    autocast = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    it = iter(dl)

    def step():
        with autocast:
            loss = system.loss(next(it))
        loss.backward()
        for opt in opts:
            opt.step()
        system.zero_grad(set_to_none=True)

    for _ in range(5):       # warmup:让 cuBLAS/NCCL 初始化、优化器态分配完
        step()
    torch.cuda.synchronize()

    from torch.profiler import profile, ProfilerActivity
    prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])
    prof.start()
    t0 = time.perf_counter()
    for _ in range(PROFILE_STEPS):
        step()
    torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - t0) / PROFILE_STEPS * 1e3
    prof.stop()

    if rank != 0:
        torch.distributed.barrier()
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    trace_path = os.path.join(OUT_DIR, f"fsdp_trace_ws{ws}_d{DEPTH}.json")
    png_path = os.path.join(OUT_DIR, f"fsdp_overlap_ws{ws}_d{DEPTH}.png")
    prof.export_chrome_trace(trace_path)

    comm, comp = load_gpu_kernels(trace_path)
    comm_len, comm_m = merge(comm)
    comp_len, comp_m = merge(comp)
    overlap = intersect_len(comm_m, comp_m)
    overlap_frac = overlap / comm_len if comm_len else 0.0
    exposed = comm_len - overlap
    n = PROFILE_STEPS

    print(f"\n[world_size={ws}]  模型 depth={DEPTH}, 每卡 batch={batch}, 序列 {seq}")
    print(f"  录了 {n} 步,导出 trace:")
    print(f"    {trace_path}")
    print(f"    (在 https://ui.perfetto.dev 或 chrome://tracing 打开,看 GPU 那几条 stream)")
    print(f"\n  GPU kernel 时间(合并重叠后,{n} 步合计):")
    print(f"    计算 kernel 忙        {comp_len:8.1f} ms")
    print(f"    NCCL 通信 kernel 忙   {comm_len:8.1f} ms")
    print(f"    两者时间上的交集      {overlap:8.1f} ms  ← 通信藏在计算底下的部分")
    bubble = wall_ms - comp_len / n - exposed / n
    print(f"\n  预言: 多数通信被计算盖住(覆盖率过半),但 PCIe 慢,必有一段露在外面。")
    print(f"  实测:")
    print(f"    通信被计算覆盖率 = 交集 / 通信总时 = {overlap_frac:.0%}")
    print(f"    露在外面的通信   = {exposed/n:6.2f} ms/步   (没盖住、直接拖慢每步)")
    print(f"\n  每步 wall-clock 拆账({wall_ms:.2f} ms/步 = 三笔):")
    print(f"    计算              {comp_len/n:6.2f} ms/步")
    print(f"    + 露在外的通信     {exposed/n:6.2f} ms/步   ← 通信税(§6);盖住越多这笔越小")
    print(f"    + 发射/同步气泡     {bubble:6.2f} ms/步   ← 两条流之间的空隙")

    plot_timeline(comm, comp, png_path, overlap_frac)
    print(f"\n  时间线 PNG(上排计算、下排通信,叠着=盖住): {png_path}")
    print("\n读法: profiler 不是用来看一个数,是用来看**两条流叠没叠上**。露在外面的通信"
          "\n      × 每秒步数,就是 d7 里那笔'总吞吐 < N×'的税。想让它变小:加大 B·T(每步"
          "\n      算得多、盖得住更多通信)、或换有 NVLink 的机器(通信本身更快)。")

    torch.distributed.barrier()


if __name__ == "__main__":
    main()


# ── 实测输出 ──────────────────────────────────────────────────────────────
# 机器:2×RTX 5090(单卡 32 GB,无 NVLink、卡间走 PCIe)· 2026-07-21 ·
# .venv/bin/torchrun --nproc_per_node=2 --standalone …/d8_profiler.py(装配日志已略)。
# 数字随机器/负载浮动,看比值不看绝对值(见 README 契约)。
#
#   [world_size=2]  模型 depth=20, 每卡 batch=12, 序列 512
#   录了 3 步,导出 trace:
#     ./outputs/demo_traces/fsdp_trace_ws2_d20.json
#     (在 https://ui.perfetto.dev 或 chrome://tracing 打开,看 GPU 那几条 stream)
#
#   GPU kernel 时间(合并重叠后,3 步合计):
#     计算 kernel 忙           421.5 ms
#     NCCL 通信 kernel 忙      261.8 ms
#     两者时间上的交集         216.9 ms  ← 通信藏在计算底下的部分
#
#   预言: 多数通信被计算盖住(覆盖率过半),但 PCIe 慢,必有一段露在外面。
#   实测:
#     通信被计算覆盖率 = 交集 / 通信总时 = 83%
#     露在外面的通信   =  14.98 ms/步   (没盖住、直接拖慢每步)
#
#   每步 wall-clock 拆账(159.21 ms/步 = 三笔):
#     计算              140.50 ms/步
#     + 露在外的通信      14.98 ms/步   ← 通信税(§6);盖住越多这笔越小
#     + 发射/同步气泡       3.73 ms/步   ← 两条流之间的空隙
#
#   时间线 PNG(上排计算、下排通信,叠着=盖住): ./outputs/demo_traces/fsdp_overlap_ws2_d20.png
#
# 关键:覆盖率 83% + 露在外的通信 ~15ms/步(≈wall 的 9%)——这 9% 正好是 d7 里
# 双卡总吞吐只有 1.86× 而非 2× 的那笔税。profiler 让"通信税"从一个笼统的比值,
# 变成时间线上两条流之间那道看得见的缝。
# ──────────────────────────────────────────────────────────────────────────
