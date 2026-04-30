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
    parser.add_argument("--use_coordination", type=str, default="True", help="是否启用协调模块（与训练时一致）")
    args = parser.parse_args()

    print(f"加载 base 模型: {args.openvla_path}")
    base_vla = CoordiVLAForActionPrediction.from_single_arm_pretrained(
        args.openvla_path,
        CoordiVLAConfig.from_pretrained(args.openvla_path, use_coordination=(args.use_coordination.lower() == "true")),
        torch_dtype=torch.bfloat16,
    )

    print(f"加载 adapter: {args.adapter_dir}")
    merged_vla = PeftModel.from_pretrained(base_vla, args.adapter_dir)
    merged_vla = merged_vla.merge_and_unload()

    # 加载 coordination_module 权重（全参数训练，不在 LoRA adapter 里）
    coord_ckpt = Path(args.adapter_dir) / "coordination_module.pt"
    if coord_ckpt.exists():
        print(f"加载 coordination_module 权重: {coord_ckpt}")
        coord_state = torch.load(coord_ckpt, map_location="cpu")
        merged_vla.coordination_module.load_state_dict(coord_state)
    elif args.use_coordination.lower() == "true":
        print("警告：use_coordination=True 但未找到 coordination_module.pt，协调模块将使用随机初始化权重")

    print(f"保存完整权重到: {args.save_dir}")
    # 注入真实 norm_stats（推理时反归一化用）
    if args.stats_path is not None:
        with open(args.stats_path) as f:
            dataset_statistics = json.load(f)
        merged_vla.norm_stats = dataset_statistics
        merged_vla.config.norm_stats = dataset_statistics  # 同步到 config，save_pretrained 才能持久化
        print(f"已注入 norm_stats，key: {list(dataset_statistics.keys())}")
    else:
        print("警告：未传 --stats_path，norm_stats 将使用 OpenVLA 原始统计量，推理时动作反归一化可能不正确")
    processor = AutoProcessor.from_pretrained(args.openvla_path, trust_remote_code=True)
    processor.save_pretrained(args.save_dir)
    merged_vla.save_pretrained(args.save_dir)

    print("完成！")


if __name__ == "__main__":
    main()
