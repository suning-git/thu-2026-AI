"""d7_fsdp.py — 多卡:每卡显存 ÷N,但 2 卡不是 2 倍快(讲义 §6)。

FSDP(fully_shard)把参数/梯度/优化器状态切成 1/N 分到各卡,算某层时临时 all-gather
拼回完整、用完 reshard。于是每卡的这部分显存 ÷N;代价是每步多出参数收发的通信。

本 demo 在**同一个模型**上量两件事:每卡峰值显存、总吞吐(tokens/s)。分别跑单卡与
双卡 FSDP,对照。核心装配复用 harness(world_size>1 时 core 自动 FSDP)。

★这台机是 2×5090、**无 NVLink**(卡间只有 PCIe ~64 GB/s),正是讲义 §6 说的"无 NVLink →
通信直接主导"的机器——预期双卡总吞吐远不到 2×,甚至可能因通信瓶颈接近 1×。

预言:① 双卡 FSDP 每卡参数常驻 ≈ 单卡 ÷ 2;② 双卡总吞吐 < 2×(通信税;此机无 NVLink 尤甚)。

运行(两次,对照输出):
  单卡:  CUDA_VISIBLE_DEVICES=0 .venv/bin/python projects/training_engineering_demos/d7_fsdp.py
  双卡:  torchrun --nproc_per_node=2 --standalone projects/training_engineering_demos/d7_fsdp.py
"""
import os
import time
import torch

from harness import build, gb

DEPTH = 20                # 0.5B 量级:单卡装得下,÷2 又清晰可测
MEAS = 20


def main():
    bs = int(os.environ.get("DEMO_BATCH", "16"))   # 每卡 batch;小 B·T 会放大通信税(§6.3 的 1/(B·T))
    h = build([f"model.depth={DEPTH}", f"device_batch_size={bs}", "use_compile=false",
               "checkpoint.enabled=false", "evaluation.text.enabled=false", "max_steps=100"])
    system, opts, dl = h["system"], h["optimizers"], h["dataloader"]
    seq, batch = h["seq"], h["batch"]
    ws = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    N = sum(p.numel() for p in system.parameters())
    autocast = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    it = iter(dl)

    def step():
        with autocast:
            loss = system.loss(next(it))
        loss.backward()
        for opt in opts:
            opt.step()
        system.zero_grad(set_to_none=True)
        return loss

    # warmup + 让优化器态分配
    for _ in range(3):
        step()
    torch.cuda.synchronize()
    per_card_resident = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    for _ in range(MEAS):
        step()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / MEAS
    peak = torch.cuda.max_memory_allocated()

    tok_per_step_per_card = batch * seq
    per_card_tps = tok_per_step_per_card / dt

    if rank == 0:
        print(f"\n[world_size={ws}]  模型 depth={DEPTH}, 参数量 N={N/1e6:.0f}M")
        print(f"  每卡参数常驻(参数+优化器态,已切分)  {gb(per_card_resident):6.2f} GB")
        print(f"  每卡峰值显存                        {gb(peak):6.2f} GB")
        print(f"  每卡吞吐   {per_card_tps/1e3:8.1f} K tok/s   →   总吞吐 {ws*per_card_tps/1e3:8.1f} K tok/s")
        print(f"\n  对照单卡运行的输出:")
        print(f"    · 每卡参数常驻应 ≈ 单卡 ÷ {ws}(FSDP 把参数、梯度、优化器状态都切成 1/N)")
        print(f"    · 总吞吐应 < {ws}× 单卡(通信税;此机无 NVLink、走 PCIe,预期折损明显)")
    if ws > 1:
        torch.distributed.barrier()


if __name__ == "__main__":
    main()


# ── 实测输出 ──────────────────────────────────────────────────────────────
# 机器:2×RTX 5090(单卡 32 GB,无 NVLink、卡间走 PCIe)· 2026-07-21 ·
# 从 repo 根目录运行。装配日志已略。数字随机器/负载浮动,看比值不看绝对值。
#
#   单卡:  CUDA_VISIBLE_DEVICES=0 .venv/bin/python projects/training_engineering_demos/d7_fsdp.py
#
#     [world_size=1]  模型 depth=20, 参数量 N=477M
#       每卡参数常驻(参数+优化器态,已切分)    2.73 GB
#       每卡峰值显存                         15.94 GB
#       每卡吞吐       44.2 K tok/s   →   总吞吐     44.2 K tok/s
#
#   双卡:  .venv/bin/torchrun --nproc_per_node=2 --standalone projects/training_engineering_demos/d7_fsdp.py
#
#     Distributed training: world_size=2
#     Wrapping trunk + head with FSDP
#     [world_size=2]  模型 depth=20, 参数量 N=477M
#       每卡参数常驻(参数+优化器态,已切分)    1.40 GB
#       每卡峰值显存                         14.78 GB
#       每卡吞吐       41.1 K tok/s   →   总吞吐     82.3 K tok/s
#
#   对照:① 每卡参数常驻 2.73 → 1.40 GB ≈ ÷2(FSDP 把参数、梯度、优化器状态都切了)✓
#         ② 总吞吐 44.2 → 82.3 K tok/s = 1.86×(< 2×,通信税;此机无 NVLink 走 PCIe,
#            这已算体面;把 DEMO_BATCH 调小会放大通信税、比值进一步跌向 1×)。
# ──────────────────────────────────────────────────────────────────────────
