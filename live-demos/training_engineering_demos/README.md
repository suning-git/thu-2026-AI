# 训练工程 · 演示画廊

配套第六课 slides(课堂现场 demo 的代码)。slides 讲**心智模型**;这里的每个 demo 在
**真实 GPU 上做一次诚实测量**,把课上的断言变成你能亲手撞的数字。

## 课程学生怎么跑

本文件夹是自足的,不必放进 nanoinfra 源码树——nanoinfra 会作为**库**装进你的虚拟环境:

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt        # 装 nanoinfra(库)及其依赖
```

- **微基准**(d1 / d3 / d4 / d5 / d6 / d_graphbreak):装好即跑,要一张 GPU;单文件,`python d1_timing.py`。
- **真训练探针**(d2 / dc / d7 / d8):另需 FineWeb 数据与 tokenizer 产物。已有课程 nanoinfra(hackathon 那份)的话直接指过去:`export NANOINFRA_BASE_DIR=<你的nanoinfra>/outputs`;没有就按 [nanoinfra](https://github.com/suning-git/nanoinfra) README 下载数据、训练 tokenizer。
- **roofline.html**:浏览器直接打开,无需 GPU。

(把本文件夹拷进 nanoinfra 的 `projects/` 下运行也完全可以——课上 slides"亲手跑一次"的路径就是按那个位置写的。)

## 契约(所有 demo 必须满足)

1. **预言先行。** 每个 demo 先 `print` 出"预言 X",再 `print` 出"实测 Y",并排放。
   你不是"跑一下看看",而是"我心里有个模型,它对不对"。对不上的时候,收获最大。
2. **断言用比值,不用绝对值。** 换台 GPU 绝对毫秒/显存就变;所以预言写成
   "裸计时比 synced 小 ~100×"(稳),不写"0.1 ms"(脆)。每个 demo 开头打印 GPU 型号。
3. **打印讲义引用的那张表/那个数。** 讲义引 demo 的输出——单一真源。
4. **一行运行命令**(见每个文件头)。产图的 demo 配 hypothesis-first 的 findings。
5. **不碰 core。** 需要真实训练栈的,子类化 `Trainer`、只覆盖文档声明的扩展点
   (`harness.py` 里已封装,`adam_probe` 是原型)。

这批 demo 教的其实不是"显存/计时/FSDP",是**监督 agent 的手感**:当你让 agent"优化一下",
一个 demo 就是"一次正确的测量长什么样"的范本——它是讲义 §8「向 agent 要证据」那张表的可执行注脚。

## 三个物种(写法不同,别混)

| 物种 | 跑在哪 | 特点 | 本画廊的 demo |
|---|---|---|---|
| **微基准** | 任意一张 GPU,几秒 | 合成小张量,依赖 torch(d3 用 liger,d4 用 flex_attention) | `d1_timing` · `d5_kv_cache` · `d6_util_vs_mfu` · `d_graphbreak` · `d3_logits_chain` · `d4_packing` |
| **真训练探针** | 训练栈,一个真模型 | 用 `harness.py` 装配真模型/dataloader/optimizer | `d2_memory` · `dc_compile` · `d7_fsdp`(多卡) · `d8_profiler`(多卡,产图) |
| **交互件** | 浏览器,无 GPU | 心智模型工具,不是模拟现象 | `roofline.html`(§7 带宽墙) |

> 交互件只给 §7 的 roofline——推理的算术强度 vs 岭线是个真正的心智模型工具。
> 其余一律别做成动画:训练工程 demo 的价值在真硅片上,不在模拟里。

## 讲义章节 → demo

| 讲义 | 断言 | demo | 状态 |
|---|---|---|---|
| §3.3 计时三法 | 裸计时/util 都在骗你 | `d1_timing` | ✅ |
| §3.4 util≠MFU | util 100% 可以只用 1% 算力 | `d6_util_vs_mfu` | ✅ |
| §7.2 KV cache | 不带 cache 生成是 O(T²) | `d5_kv_cache` | ✅ |
| §2 显存五本账 | 激活是大头、∝B·T | `d2_memory` | ✅ |
| §5.1-5.2 torch.compile | 融合让稳定更快、峰值更低(首步一次性编译) | `dc_compile` | ✅ |
| §5.3 graph break | 数据依赖分支断图,融合收益归零 | `d_graphbreak` | ✅ |
| §2.4 logits 链 | fused CE 把峰值大幅压低(V 是显存问题) | `d3_logits_chain` | ✅ |
| §4 packing | padding 浪费 = 1−平均/补齐长 | `d4_packing` | ✅ |
| §7.3 带宽墙 | decode 撞 ~算力/带宽 的岭线 | `roofline.html` | ✅ |
| §6 FSDP | 每卡显存 ÷N、通信税 ∝1/(B·T) | `d7_fsdp` | ✅(需 2 卡) |
| §6 + §3.3 profiler | FSDP 通信被计算覆盖多少(overlap);wall 拆成 计算+露在外的通信+气泡 | `d8_profiler` | ✅(需 2 卡,产时间线图) |

## 运行(在本文件夹内,venv 已激活)

```bash
# 微基准 —— 任意一张空 GPU
CUDA_VISIBLE_DEVICES=0 python d1_timing.py
CUDA_VISIBLE_DEVICES=0 python d5_kv_cache.py
CUDA_VISIBLE_DEVICES=0 python d6_util_vs_mfu.py
CUDA_VISIBLE_DEVICES=0 python d_graphbreak.py
CUDA_VISIBLE_DEVICES=0 python d3_logits_chain.py   # 需 liger_kernel
CUDA_VISIBLE_DEVICES=0 python d4_packing.py         # 需 flex_attention(torch≥2.5)

# 真训练探针 —— 需 FineWeb 与 tokenizer 产物(见上节的 NANOINFRA_BASE_DIR)
CUDA_VISIBLE_DEVICES=0 python d2_memory.py
CUDA_VISIBLE_DEVICES=0 python dc_compile.py
# d7 多卡:跑两次对照(单卡 vs 双卡 FSDP);DEMO_BATCH 调每卡 batch 看通信税随 B·T 变化
CUDA_VISIBLE_DEVICES=0 python d7_fsdp.py
torchrun --nproc_per_node=2 --standalone d7_fsdp.py
# d8 profiler:2 卡,录 trace + 算通信 overlap + 出时间线 PNG(→ outputs/demo_traces/)
torchrun --nproc_per_node=2 --standalone d8_profiler.py

# 交互件 —— 浏览器直接打开
roofline.html
```

## 贡献

这是一个**画廊**:每个 demo 一个自包含文件、满足上面的契约,就能进来。
微基准直接嵌进讲义;真探针复用 `harness.py`。契约即门槛。
