"""harness.py — 真训练探针 demo 的共用装配(不碰 core)。

把《优化器讲义》adam_probe 里趟通的那套装配抽出来:hydra compose 出 text-pretrain
的配置、装配 vocab/模型/dataloader/optimizer,返回可驱动的对象。demo 只需覆盖
一件事(通常是"在一个 step 的某些位置埋下测量点"),不重写训练。

这是 demo 画廊的基础设施,不是被冻结的研究载体,所以两个真探针 demo 共享它
(与消融画廊"每个实验各自复制、冻结一份代码"的纪律不同——那是为了钉死图背后的代码,这里不是)。
"""
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import modalities.text
from modalities.text.train_text import SOURCE_TYPES, assemble_vocab, resolve_trunk
from modalities.text import get_tokenizer
from core.training.model_setup import build_system
from core.training.trainer import create_optimizers
from core.model.gpt import GPTConfig
from core.data.mixed_dataloader import MixedDataLoader

# 配置目录从已导入的包定位——无论 nanoinfra 是 pip 装的库还是本文件夹恰好
# 躺在其源码树里,都成立;数据/产物位置照常由 NANOINFRA_BASE_DIR 等环境变量决定。
CONFIG_DIR = Path(modalities.text.__file__).resolve().parent / "configs"


def build(overrides):
    """返回 dict: system / optimizers / dataloader / config / gpt_config。

    overrides: hydra CLI 覆盖列表,如 ["model.depth=6", "device_batch_size=16",
    "use_compile=false"]。demo 用它选模型尺度与 batch。
    """
    with initialize_config_dir(config_dir=str(CONFIG_DIR),
                               version_base=None):
        cfg = compose(config_name="train_text", overrides=overrides)
    config = OmegaConf.to_container(cfg, resolve=True)

    seq = config["sequence_len"]
    bs = config["device_batch_size"]
    mc = config["model"]

    tokenizer = get_tokenizer()
    layout, resolver = assemble_vocab(tokenizer)
    gpt_config = GPTConfig(
        sequence_len=seq, vocab_size=layout.vocab_size, n_layer=mc["depth"],
        n_head=mc["n_head"], n_kv_head=mc["n_kv_head"], n_embd=mc["dim"],
        n_token_types=layout.n_token_types,
    )
    trunk_cls = resolve_trunk(mc.get("trunk_class"))
    setup = build_system(trunk_cls, gpt_config,
                         use_compile=config.get("use_compile", True),
                         seed=config.get("seed", 42))

    sources = []
    for sc in config["data"]["sources"]:
        sc = dict(sc)
        sc.setdefault("sequence_len", seq)
        sc.setdefault("device", "cuda")
        sources.append(sc)
    dataloader = MixedDataLoader(
        loader_config={"batch_size": bs,
                       "data": {"sequence_len": seq, "sources": sources}},
        tokenizers={"text": tokenizer, "layout": layout, "control_resolver": resolver},
        source_types=SOURCE_TYPES, resume_state_dict=None,
    )
    optimizers = create_optimizers(setup["system"], config["optimizer"],
                                   world_size=setup["world_size"])
    return {
        "system": setup["system"], "optimizers": optimizers, "dataloader": dataloader,
        "config": config, "gpt_config": gpt_config,
        "seq": seq, "batch": bs,
    }


def gpu_banner():
    assert torch.cuda.is_available(), "需要 GPU"
    name = torch.cuda.get_device_name(0)
    print(f"[GPU] {name}")
    return name


def gb(nbytes):
    return nbytes / 1024 ** 3
