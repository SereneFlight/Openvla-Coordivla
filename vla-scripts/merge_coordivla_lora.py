"""
merge_coordivla_lora.py

训练完成后，将 LoRA adapter 融合进 base 模型，保存完整权重。

用法：
    python vla-scripts/merge_coordivla_lora.py \
        --openvla_path /root/openvla-7b \
        --adapter_dir  /root/openvla/adapter-tmp/<run_name> \
        --save_dir     /root/openvla/merged/<run_name>
"""

import argparse
import json
import torch
from pathlib import Path
from peft import PeftModel
from transformers import AutoConfig, AutoModelForVision2Seq, AutoProcessor, AutoImageProcessor

from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.extern.hf.coordivla_configuration import CoordiVLAConfig
from prismatic.extern.hf.coordivla_modeling import CoordiVLAForActionPrediction

ACTION_ENCODING = "joint_delta_gripper_cont_abs"
ACTION_MASK = [True, True, True, True, True, True, False]


def parse_optional_bool(value):
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
    return bool(value)

# 注册 HuggingFace Auto 类
try:
    AutoConfig.register("openvla", OpenVLAForActionPrediction.config_class)
    AutoImageProcessor.register(OpenVLAForActionPrediction.config_class, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAForActionPrediction.config_class, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAForActionPrediction.config_class, OpenVLAForActionPrediction)
except Exception:
    pass

try:
    AutoConfig.register("coordivla", CoordiVLAConfig)
    AutoImageProcessor.register(CoordiVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(CoordiVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(CoordiVLAConfig, CoordiVLAForActionPrediction)
except Exception:
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--openvla_path",  type=str, required=True, help="OpenVLA 原始权重路径")
    parser.add_argument("--adapter_dir",   type=str, required=True, help="LoRA adapter 保存路径")
    parser.add_argument("--save_dir",      type=str, required=True, help="merge 后完整权重保存路径")
    parser.add_argument("--stats_path",    type=str, default=None,  help="dataset_statistics.json 路径（由 compute_dataset_statistics.py 生成）")
    parser.add_argument(
        "--use_coordination",
        type=str,
        default=None,
        choices=["True", "False", "true", "false"],
        help="是否启用协调模块（与训练时一致）；None 时优先从 adapter/config.json 读取",
    )
    parser.add_argument(
        "--coordination_summary_tokens",
        type=int,
        default=None,
        help="协调模块 summary query 数；None 时优先从 adapter/config.json 读取，旧权重缺省为 0",
    )
    parser.add_argument(
        "--coordination_num_layers",
        type=int,
        default=None,
        help="协调模块层数；None 时优先从 adapter/config.json/coordination_module.pt 推断",
    )
    parser.add_argument(
        "--coordination_fusion",
        type=str,
        default=None,
        choices=["residual", "meta_query"],
        help="Coordination fusion type; None reads adapter/config.json or infers from coordination_module.pt.",
    )
    parser.add_argument(
        "--coordination_include_self_action_prefix",
        type=str,
        default=None,
        choices=["True", "False", "true", "false"],
        help="For meta_query fusion, whether target arm action queries can attend to same-arm action prefix.",
    )
    parser.add_argument(
        "--coordination_prefix_memory",
        type=str,
        default=None,
        choices=["True", "False", "true", "false"],
        help="Whether cross-arm summary tokens are carried as explicit prefixes through the remaining LLM layers.",
    )
    args = parser.parse_args()

    adapter_config_path = Path(args.adapter_dir) / "config.json"
    use_coordination = parse_optional_bool(args.use_coordination)
    coordination_summary_tokens = args.coordination_summary_tokens
    coordination_num_layers = args.coordination_num_layers
    coordination_fusion = args.coordination_fusion
    coordination_include_self_action_prefix = parse_optional_bool(args.coordination_include_self_action_prefix)
    coordination_prefix_memory = parse_optional_bool(args.coordination_prefix_memory)
    train_projector = None
    user_coordination_fusion = args.coordination_fusion
    if adapter_config_path.exists():
        with open(adapter_config_path) as f:
            adapter_model_config = json.load(f)
        if use_coordination is None:
            use_coordination = parse_optional_bool(adapter_model_config.get("use_coordination"))
        train_projector = parse_optional_bool(adapter_model_config.get("train_projector"))
        if coordination_summary_tokens is None:
            coordination_summary_tokens = adapter_model_config.get("coordination_summary_tokens")
        if coordination_num_layers is None:
            coordination_num_layers = adapter_model_config.get("coordination_num_layers")
        if coordination_fusion is None:
            coordination_fusion = adapter_model_config.get("coordination_fusion")
        if coordination_include_self_action_prefix is None:
            coordination_include_self_action_prefix = parse_optional_bool(
                adapter_model_config.get("coordination_include_self_action_prefix")
            )
        if coordination_prefix_memory is None:
            coordination_prefix_memory = parse_optional_bool(adapter_model_config.get("coordination_prefix_memory"))

    coord_ckpt_path = Path(args.adapter_dir) / "coordination_module.pt"
    if (coordination_summary_tokens is None or coordination_num_layers is None or coordination_fusion is None) and coord_ckpt_path.exists():
        coord_state_for_shape = torch.load(coord_ckpt_path, map_location="cpu")
        if coordination_summary_tokens is None:
            query = coord_state_for_shape.get("left_summary_queries")
            coordination_summary_tokens = int(query.shape[0]) if query is not None else 0
        if coordination_num_layers is None:
            layer_ids = {
                int(key.split(".")[1])
                for key in coord_state_for_shape
                if key.startswith("layers.") and len(key.split(".")) > 2 and key.split(".")[1].isdigit()
            }
            coordination_num_layers = max(layer_ids) + 1 if layer_ids else 3
        if coordination_fusion is None:
            coordination_fusion = "meta_query" if any("prefix_attn" in key for key in coord_state_for_shape) else "residual"
        del coord_state_for_shape

    if user_coordination_fusion is None and coord_ckpt_path.exists():
        coord_state_for_fusion = torch.load(coord_ckpt_path, map_location="cpu")
        inferred_fusion = "meta_query" if any("prefix_attn" in key for key in coord_state_for_fusion) else "residual"
        del coord_state_for_fusion
        if coordination_fusion is not None and coordination_fusion != inferred_fusion:
            print(
                f"Warning: coordination_fusion={coordination_fusion!r} from config does not match "
                f"coordination_module.pt ({inferred_fusion!r}); using checkpoint-inferred value."
            )
        coordination_fusion = inferred_fusion

    if coordination_summary_tokens is None:
        if coord_ckpt_path.exists():
            coordination_summary_tokens = 0
        else:
            coordination_summary_tokens = 0
    if coordination_num_layers is None:
        coordination_num_layers = 3
    if coordination_fusion is None:
        coordination_fusion = "residual"
    if coordination_include_self_action_prefix is None:
        coordination_include_self_action_prefix = True
    if coordination_prefix_memory is None:
        coordination_prefix_memory = False
    if use_coordination is None:
        use_coordination = coord_ckpt_path.exists()
    use_coordination = bool(use_coordination)
    if coordination_prefix_memory and coordination_summary_tokens <= 0:
        raise ValueError("coordination_prefix_memory=True requires coordination_summary_tokens > 0.")
    print(f"use_coordination: {use_coordination}")
    print(f"coordination_summary_tokens: {coordination_summary_tokens}")
    print(f"coordination_num_layers: {coordination_num_layers}")
    print(f"coordination_fusion: {coordination_fusion}")
    print(f"coordination_include_self_action_prefix: {coordination_include_self_action_prefix}")
    print(f"coordination_prefix_memory: {coordination_prefix_memory}")

    print(f"加载 base 模型: {args.openvla_path}")
    base_vla = CoordiVLAForActionPrediction.from_single_arm_pretrained(
        args.openvla_path,
        CoordiVLAConfig.from_pretrained(
            args.openvla_path,
            use_coordination=use_coordination,
            coordination_summary_tokens=coordination_summary_tokens,
            coordination_num_layers=coordination_num_layers,
            coordination_fusion=coordination_fusion,
            coordination_include_self_action_prefix=coordination_include_self_action_prefix,
            coordination_prefix_memory=coordination_prefix_memory,
        ),
        torch_dtype=torch.bfloat16,
    )

    print(f"加载 adapter: {args.adapter_dir}")
    merged_vla = PeftModel.from_pretrained(base_vla, args.adapter_dir)
    merged_vla = merged_vla.merge_and_unload()

    # 加载 coordination_module 权重（全参数训练，不在 LoRA adapter 里）
    coord_ckpt = Path(args.adapter_dir) / "coordination_module.pt"
    if use_coordination and coord_ckpt.exists():
        print(f"加载 coordination_module 权重: {coord_ckpt}")
        coord_state = torch.load(coord_ckpt, map_location="cpu")
        if getattr(merged_vla.coordination_module, "fusion", "residual") == "residual":
            merged_vla.coordination_module.alpha_left = torch.nn.Parameter(
                merged_vla.coordination_module.alpha_left.detach().float()
            )
            merged_vla.coordination_module.alpha_right = torch.nn.Parameter(
                merged_vla.coordination_module.alpha_right.detach().float()
            )
        merged_vla.coordination_module.load_state_dict(coord_state)
    elif use_coordination:
        raise FileNotFoundError(
            f"use_coordination=True but coordination weights are missing: {coord_ckpt}"
        )

    projector_ckpt = Path(args.adapter_dir) / "projector.pt"
    if projector_ckpt.exists():
        print(f"Loading projector weights: {projector_ckpt}")
        projector_state = torch.load(projector_ckpt, map_location="cpu")
        merged_vla.projector_left.load_state_dict(projector_state["projector_left"])
        merged_vla.projector_right.load_state_dict(projector_state["projector_right"])
        merged_vla.config.train_projector = True
    elif train_projector:
        raise FileNotFoundError(
            f"adapter config says train_projector=True but projector weights are missing: {projector_ckpt}"
        )
    else:
        print(f"No projector weights found; using base projector: {projector_ckpt}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"保存完整权重到: {save_dir}")
    # 注入真实 norm_stats（推理时反归一化用）
    if args.stats_path is None:
        raise ValueError("The OpenVLA-style delta-action model requires --stats_path from compute_dataset_statistics.py.")
    with open(args.stats_path) as f:
        dataset_statistics = json.load(f)
    for key, stats in dataset_statistics.items():
        action_stats = stats.get("action", {})
        action_encoding = action_stats.get("action_encoding")
        if action_encoding != ACTION_ENCODING:
            raise ValueError(
                f"stats_path uses action_encoding={action_encoding!r} for {key}; "
                f"expected {ACTION_ENCODING!r}. Recompute dataset statistics with the updated script."
            )
        action_mask = action_stats.get("mask")
        if action_mask != ACTION_MASK:
            raise ValueError(
                f"stats_path uses mask={action_mask!r} for {key}; expected {ACTION_MASK!r}. "
                "The gripper dimension must remain absolute, continuous, and unnormalized."
            )
    merged_vla.norm_stats = dataset_statistics
    merged_vla.config.norm_stats = dataset_statistics  # 同步到 config，save_pretrained 才能持久化
    print(f"已注入 norm_stats，key: {list(dataset_statistics.keys())}")
    processor = AutoProcessor.from_pretrained(args.openvla_path, trust_remote_code=True)
    with open(save_dir / "dataset_statistics.json", "w") as f:
        json.dump(dataset_statistics, f, indent=2)
    processor.save_pretrained(save_dir)
    merged_vla.save_pretrained(save_dir)
    if use_coordination:
        torch.save(
            merged_vla.coordination_module.state_dict(),
            save_dir / "coordination_module.pt",
        )
        if getattr(merged_vla.coordination_module, "fusion", "residual") == "residual":
            torch.save(
                {
                    "alpha_left": merged_vla.coordination_module.alpha_left.detach().cpu().float(),
                    "alpha_right": merged_vla.coordination_module.alpha_right.detach().cpu().float(),
                },
                save_dir / "coordination_alpha_fp32.pt",
            )

    if projector_ckpt.exists():
        torch.save(
            {
                "projector_left": merged_vla.projector_left.state_dict(),
                "projector_right": merged_vla.projector_right.state_dict(),
            },
            save_dir / "projector.pt",
        )

    print("完成！")


if __name__ == "__main__":
    main()
