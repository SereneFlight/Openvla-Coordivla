"""
finetune.py

Simple script for parameter-efficient fine-tuning of OpenVLA models loaded through the HuggingFace AutoClasses, using
HuggingFace PEFT library for low-rank adaptation (LoRA).

Notes & Benchmarks:
    - Requires PEFT (`pip install peft==0.11.1`)
    - LoRA fine-tuning (see parameters below -- no quantization, LoRA rank = 32, target_modules = all-linear):
        + One 48 GB GPU can fit a Batch Size of 12
        + One 80 GB GPU can fit a Batch Size of 24

Run with:
    - [Single Node Multi-GPU (= $K) ]: torchrun --standalone --nnodes 1 --nproc-per-node $K vla-scripts/finetune.py
    - [Override Config Values]: torchrun --standalone --nnodes 1 --nproc-per-node $K vla-scripts/finetune.py \
                                    --data_root_dir <PATH/TO/RLDS/DATASETS/DIRECTORY> \
                                    --dataset_name <DATASET_NAME> \
                                    --run_root_dir <PATH/TO/LOGS/DIR> \
                                    ...
"""

import os
import io
import json
import glob
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import draccus
import h5py
import torch
import torch.distributed as dist
import tqdm
from PIL import Image
from accelerate import PartialState
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from transformers import AutoConfig, AutoImageProcessor
from torchvision import transforms as T
from transformers.modeling_outputs import CausalLMOutputWithPast

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wandb
from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.robotwin_action_utils import build_joint_delta_gripper_cont_abs, standardize_gripper_actions
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.extern.hf.coordivla_configuration import CoordiVLAConfig
from prismatic.extern.hf.coordivla_modeling import CoordiVLAForActionPrediction

COORDINATION_METADATA_KEYS = (
    "use_coordination",
    "coordination_summary_tokens",
    "coordination_num_layers",
    "coordination_fusion",
    "coordination_include_self_action_prefix",
    "coordination_prefix_memory",
    "train_projector",
)

TRAIN_PROJECTOR = True


def write_coordination_metadata(model, save_dir) -> None:
    """Persist coordination config next to the LoRA adapter for merge-time reconstruction."""
    base_model = model.get_base_model() if hasattr(model, "get_base_model") else model
    metadata = {
        key: getattr(base_model.config, key)
        for key in COORDINATION_METADATA_KEYS
        if hasattr(base_model.config, key)
    }
    config_path = Path(save_dir) / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            adapter_config = json.load(f)
    else:
        adapter_config = {}
    adapter_config.update(metadata)
    with open(config_path, "w") as f:
        json.dump(adapter_config, f, indent=2)

# === HuggingFace Auto 类注册 ===
# 注册的作用：让 AutoConfig/AutoModel 通过 model_type 字符串找到对应的类
# 比如 config.json 里写 "model_type": "coordivla"，HF 就知道用 CoordiVLAConfig
#
# 为什么保留 openvla 的注册？
# from_single_arm_pretrained 里要加载原版 OpenVLA 权重，必须先注册才能 from_pretrained
#
# try/except ValueError：防止脚本多次运行时重复注册报错
try:
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
except ValueError:
    pass  # 已经注册过，忽略

try:
    AutoConfig.register("coordivla", CoordiVLAConfig)
    AutoImageProcessor.register(CoordiVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(CoordiVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(CoordiVLAConfig, CoordiVLAForActionPrediction)
except ValueError:
    pass  # 已经注册过，忽略

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"

IGNORE_INDEX = -100
ACTION_ENCODING = "joint_delta_gripper_cont_abs"
ACTION_MASK = np.array([True, True, True, True, True, True, False], dtype=bool)


def normalize_action(action: np.ndarray, q01: np.ndarray, q99: np.ndarray, mask: np.ndarray) -> np.ndarray:
    normalized = 2 * (action - q01) / (q99 - q01 + 1e-8) - 1
    normalized = np.clip(normalized, -1.0, 1.0)
    return np.where(mask, normalized, action).astype(np.float32)


# ============================================================
# RoboTwinDataset：读取 RoboTwin hdf5，返回 CoordiVLA 需要的字段
# ============================================================
class RoboTwinDataset(Dataset):

    def __init__(self, data_dir, task_name, action_tokenizer, base_tokenizer,
                 image_transform, prompt_builder_fn, image_aug=False):
        self.action_tokenizer = action_tokenizer
        self.base_tokenizer = base_tokenizer
        self.image_transform = image_transform
        self.prompt_builder_fn = prompt_builder_fn
        self.aug = T.Compose([
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
            T.RandomGrayscale(p=0.05),
        ]) if image_aug else None

        # 加载语言指令模板（只用 seen，不混入 unseen，避免评测泄漏）
        inst_path = os.path.join(data_dir, "..", "description", "task_instruction", f"{task_name}.json")
        inst_path = os.path.abspath(inst_path)
        with open(inst_path, "r") as f:
            inst_data = json.load(f)
        self.instructions = inst_data["seen"]

        # 扫描所有 episode hdf5，建立索引 [(文件路径, 时间步), ...]
        self.samples = []
        hdf5_dir = os.path.join(data_dir, task_name, "demo_clean", "data")
        for hdf5_path in sorted(glob.glob(os.path.join(hdf5_dir, "episode*.hdf5"))):
            with h5py.File(hdf5_path, "r") as ep:
                n_steps = ep["joint_action/left_arm"].shape[0]
            # obs[t] predicts the next recorded qpos target, so the last frame has no label.
            for t in range(max(0, n_steps - 1)):
                self.samples.append((hdf5_path, t))

        # 按唯一文件列表统计，避免按 self.samples 重复读同一文件 n_steps 次
        unique_files = sorted(set(hdf5_path for hdf5_path, _ in self.samples))
        all_left = []
        all_right = []
        self.left_gripper_cache = {}
        self.right_gripper_cache = {}
        for hdf5_path in unique_files:
            with h5py.File(hdf5_path, "r") as f:
                left_gripper = standardize_gripper_actions(f["joint_action/left_gripper"][:])
                right_gripper = standardize_gripper_actions(f["joint_action/right_gripper"][:])
                self.left_gripper_cache[hdf5_path] = left_gripper
                self.right_gripper_cache[hdf5_path] = right_gripper
                left = build_joint_delta_gripper_cont_abs(
                    f["joint_action/left_arm"][:],
                    left_gripper,
                )
                right = build_joint_delta_gripper_cont_abs(
                    f["joint_action/right_arm"][:],
                    right_gripper,
                )
                all_left.append(left)
                all_right.append(right)
        all_left = np.concatenate(all_left, axis=0)  # (2*N*T, 7)
        all_right = np.concatenate(all_right, axis=0)
        self.dataset_statistics = {
            task_name: {
                "action": {
                    "lq01":  np.percentile(all_left, 1,  axis=0).astype(np.float32),
                    "lq99":  np.percentile(all_left, 99, axis=0).astype(np.float32),
                    "mask": ACTION_MASK.copy(),
                    "rq01":  np.percentile(all_right, 1,  axis=0).astype(np.float32),
                    "rq99":  np.percentile(all_right, 99,  axis=0).astype(np.float32),
                    "action_encoding": ACTION_ENCODING,
                }
            }
        }
        self.lq01 = self.dataset_statistics[task_name]['action']['lq01']
        self.lq99 = self.dataset_statistics[task_name]['action']['lq99']
        self.rq01 = self.dataset_statistics[task_name]['action']['rq01']
        self.rq99 = self.dataset_statistics[task_name]['action']['rq99']

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        hdf5_path, t = self.samples[idx]

        with h5py.File(hdf5_path, "r") as ep:
            # 读取三路图像：全局 + 左腕 + 右腕
            img_global  = Image.open(io.BytesIO(bytes(ep["observation/head_camera/rgb"][t]))).convert("RGB")
            img_wrist_l = Image.open(io.BytesIO(bytes(ep["observation/left_camera/rgb"][t]))).convert("RGB")
            img_wrist_r = Image.open(io.BytesIO(bytes(ep["observation/right_camera/rgb"][t]))).convert("RGB")

            # 读取左右臂动作：arm(6) + gripper(1) = 7 维
            left_gripper = self.left_gripper_cache[hdf5_path]
            right_gripper = self.right_gripper_cache[hdf5_path]
            left_action = np.concatenate([
                ep["joint_action/left_arm"][t + 1] - ep["joint_action/left_arm"][t],
                np.array(left_gripper[t + 1]).reshape(1)
            ]).astype(np.float32)
            right_action = np.concatenate([
                ep["joint_action/right_arm"][t + 1] - ep["joint_action/right_arm"][t],
                np.array(right_gripper[t + 1]).reshape(1)
            ]).astype(np.float32)

        # 图像预处理（resize + 归一化），复用 OpenVLA 的 image_transform
        if self.aug is not None:
            img_global  = self.aug(img_global)
            img_wrist_l = self.aug(img_wrist_l)
            img_wrist_r = self.aug(img_wrist_r)
        pv_global  = self.image_transform(img_global)
        pv_wrist_l = self.image_transform(img_wrist_l)
        pv_wrist_r = self.image_transform(img_wrist_r)

        # 随机选一条语言指令
        lang = np.random.choice(self.instructions).lower()

        # 构建 prompt + tokenize + labels（左右臂各自独立）
        def build_ids_and_labels(action):
            prompt_builder = self.prompt_builder_fn("openvla")
            conversation = [
                {"from": "human", "value": f"What action should the robot take to {lang}?"},
                {"from": "gpt", "value": self.action_tokenizer(action)},
            ]
            for turn in conversation:
                prompt_builder.add_turn(turn["from"], turn["value"])
            input_ids = self.base_tokenizer(
                prompt_builder.get_prompt(), add_special_tokens=True
            ).input_ids
            labels = list(input_ids)
            input_ids = torch.tensor(input_ids)
            labels = torch.tensor(labels)
            # 只对 action token 计算 loss，其余设为 IGNORE_INDEX
            action_mask = labels > self.action_tokenizer.action_token_begin_idx
            labels[~action_mask] = IGNORE_INDEX
            return input_ids, labels

        left_action_norm = normalize_action(left_action, self.lq01, self.lq99, ACTION_MASK)
        right_action_norm = normalize_action(right_action, self.rq01, self.rq99, ACTION_MASK)
        input_ids_left,  labels_left  = build_ids_and_labels(left_action_norm)
        input_ids_right, labels_right = build_ids_and_labels(right_action_norm )

        return dict(
            pixel_values_global=pv_global,
            pixel_values_wrist_left=pv_wrist_l,
            pixel_values_wrist_right=pv_wrist_r,
            input_ids_left=input_ids_left,
            input_ids_right=input_ids_right,
            labels_left=labels_left,
            labels_right=labels_right,
        )


# ============================================================
# CoordiVLACollator：把一个 batch 的样本 pad 对齐，输出 9 个字段
# ============================================================
# 为什么需要 Collator？
# Dataset 返回的每个样本，input_ids 长度可能不同（不同指令文本长度不同）。
# DataLoader 要把多个样本拼成一个 batch，必须长度一致。
# Collator 的工作就是：短的补 padding，长的截断，同时生成 attention_mask。
class CoordiVLACollator:

    def __init__(self, max_length, pad_token_id, padding_side="right"):
        self.max_length = max_length
        self.pad_token_id = pad_token_id
        self.padding_side = padding_side

    def __call__(self, batch):
        # batch 是一个 list，每个元素是 Dataset.__getitem__ 返回的 dict
        # 例如 batch[0]["input_ids_left"] 是第 0 个样本的左臂 input_ids

        # 分别收集左右臂的 input_ids 和 labels
        ids_left  = [item["input_ids_left"]  for item in batch]
        ids_right = [item["input_ids_right"] for item in batch]
        lab_left  = [item["labels_left"]     for item in batch]
        lab_right = [item["labels_right"]    for item in batch]

        # pad 左臂（返回对齐后的 input_ids、attention_mask、labels）
        ids_left_padded, mask_left, lab_left_padded = self._pad(ids_left, lab_left)
        # pad 右臂
        ids_right_padded, mask_right, lab_right_padded = self._pad(ids_right, lab_right)

        # 图像不需要 pad，直接 stack 成 batch
        pv_global  = torch.stack([item["pixel_values_global"]      for item in batch])
        pv_wrist_l = torch.stack([item["pixel_values_wrist_left"]  for item in batch])
        pv_wrist_r = torch.stack([item["pixel_values_wrist_right"] for item in batch])

        # 返回模型 forward 需要的 9 个字段
        return dict(
            pixel_values_global=pv_global,
            pixel_values_wrist_left=pv_wrist_l,
            pixel_values_wrist_right=pv_wrist_r,
            input_ids_left=ids_left_padded,
            input_ids_right=ids_right_padded,
            attention_mask_left=mask_left,
            attention_mask_right=mask_right,
            labels_left=lab_left_padded,
            labels_right=lab_right_padded,
        )

    def _pad(self, input_ids_list, labels_list):
        """把一组不等长的 input_ids 和 labels pad 到相同长度"""
        # 取 batch 内最长的，但不超过 max_length
        max_len = min(self.max_length, max(len(ids) for ids in input_ids_list))
        padded_ids, padded_labels, masks = [], [], []

        for ids, lab in zip(input_ids_list, labels_list):
            # 超长样本截断，避免 pad_len 为负
            ids = ids[:max_len]
            lab = lab[:max_len]
            pad_len = max_len - len(ids)
            if self.padding_side == "right":
                # 右边补 padding：[真实token ... padding]
                p_ids = torch.cat([ids, torch.full((pad_len,), self.pad_token_id, dtype=ids.dtype)])
                p_lab = torch.cat([lab, torch.full((pad_len,), IGNORE_INDEX, dtype=lab.dtype)])
                mask  = torch.cat([torch.ones(len(ids), dtype=torch.long), torch.zeros(pad_len, dtype=torch.long)])
            else:
                # 左边补 padding：[padding ... 真实token]
                p_ids = torch.cat([torch.full((pad_len,), self.pad_token_id, dtype=ids.dtype), ids])
                p_lab = torch.cat([torch.full((pad_len,), IGNORE_INDEX, dtype=lab.dtype), lab])
                mask  = torch.cat([torch.zeros(pad_len, dtype=torch.long), torch.ones(len(ids), dtype=torch.long)])
            padded_ids.append(p_ids)
            padded_labels.append(p_lab)
            masks.append(mask)

        # stack 成 (B, max_len) 的 tensor
        return torch.stack(padded_ids), torch.stack(masks), torch.stack(padded_labels)


# # === Utilities ===
# # fmt: off
# def create_vision_transform(vla: nn.Module, input_size: int) -> Callable[[Image.Image], torch.Tensor]:
#     """Gets image transform for the vision encoder."""
#     data_cfg = timm.data.resolve_model_data_config(vla.vision_backbone)
#     data_cfg["input_size"] = (3, input_size, input_size)
#     return timm.data.create_transform(
#         input_size=data_cfg["input_size"],
#         interpolation=data_cfg["interpolation"],
#         mean=data_cfg["mean"],
#         std=data_cfg["std"],
#         crop_pct=1.0,           # Set to 1.0 to disable cropping
#         crop_mode="center",     # Default crop mode --> no-op when `crop_pct == 1.0`
#         is_training=False,      # Disable image_aug when loading transform; handled by RLDS dataloader
#     )
#
# # fmt: on


@dataclass
class FinetuneConfig:
    # fmt: off
    openvla_path: str = "/home/yj/desktop/openvla-7b"              # 预训练 OpenVLA 路径（用于初始化 CoordiVLA 权重）

    # Directory Paths
    data_dir: Path = Path("/home/yj/RoboTwin/data")                # RoboTwin 数据根目录
    task_name: str = "beat_block_hammer"                           # 任务名
    run_root_dir: Path = Path("runs")                               # Path to directory to store logs & checkpoints
    adapter_tmp_dir: Path = Path("adapter-tmp")                     # Temporary directory for LoRA weights before fusing

    # Fine-tuning Parameters
    batch_size: int = 16                                            # Fine-tuning batch size
    max_steps: int = 200_000                                        # Max number of fine-tuning steps
    save_steps: int = 500                                           # Interval for checkpoint saving
    learning_rate: float = 1e-4                                     # Fine-tuning learning rate
    warmup_steps: int = 500                                         # Warmup 步数，从 0 线性升到 learning_rate
    max_grad_norm: float = 1.0                                      # Gradient clipping，防止梯度爆炸
    grad_accumulation_steps: int = 1                                # Gradient accumulation steps
    image_aug: bool = True                                          # Whether to train with image augmentations
    shuffle_buffer_size: int = 100_000                              # Dataloader shuffle buffer size (can reduce if OOM)
    save_latest_checkpoint_only: bool = True                        # Whether to save only one checkpoint per run and
                                                                    #   continually overwrite the latest checkpoint
                                                                    #   (If False, saves all checkpoints)

    # LoRA Arguments
    use_lora: bool = True                                           # Whether to use LoRA fine-tuning
    lora_rank: int = 32                                             # Rank of LoRA weight matrix
    lora_dropout: float = 0.0                                       # Dropout applied to LoRA weights
    use_coordination: bool = True                                   # 是否启用协调模块
    coordination_summary_tokens: int = 16                           # 16 个 summary query；设为 0 则使用完整上下文
    coordination_num_layers: int = 1                                # 小数据任务先用 1 层，避免协调模块参数过大
    coordination_fusion: str = "meta_query"                         # meta_query: coordination prefix attention
    coordination_include_self_action_prefix: bool = False            # False: 7x(16+7); True: 7x(16+7+7)
    coordination_prefix_memory: bool = False                         # Carry cross-arm summary tokens through remaining LLM layers
    use_quantization: bool = False                                  # Whether to 4-bit quantize VLA for LoRA fine-tuning
                                                                    #   => CAUTION: Reduces memory but hurts performance

    # Tracking Parameters
    wandb_project: str = "coordivla"                                  # Name of W&B project to log to (use default!)
    wandb_entity: str = "stanford-voltron"                          # Name of entity to log under
    run_id_note: Optional[str] = None                               # Extra note for logging, Weights & Biases

    # fmt: on


@draccus.wrap()
def finetune(cfg: FinetuneConfig) -> None:
    print(f"Fine-tuning OpenVLA Model `{cfg.openvla_path}` on `{cfg.task_name}`")

    # [Validate] Ensure GPU Available & Set Device / Distributed Context
    assert torch.cuda.is_available(), "Fine-tuning assumes at least one GPU is available!"
    distributed_state = PartialState()
    torch.cuda.set_device(device_id := distributed_state.local_process_index)
    torch.cuda.empty_cache()

    # Configure Unique Experiment ID & Log Directory
    exp_id = (
        f"{cfg.openvla_path.split('/')[-1]}+{cfg.task_name}"
        f"+b{cfg.batch_size * cfg.grad_accumulation_steps}"
        f"+lr-{cfg.learning_rate}"
    )
    if cfg.use_lora:
        exp_id += f"+lora-r{cfg.lora_rank}+dropout-{cfg.lora_dropout}"
    if TRAIN_PROJECTOR:
        exp_id += "+projector"
    if cfg.use_coordination:
        exp_id += f"+coord-{cfg.coordination_fusion}"
    if cfg.use_quantization:
        exp_id += "+q-4bit"
    if cfg.run_id_note is not None:
        exp_id += f"--{cfg.run_id_note}"
    if cfg.image_aug:
        exp_id += "--image_aug"

    # Start =>> Build Directories
    run_dir, adapter_dir = cfg.run_root_dir / exp_id, cfg.adapter_tmp_dir / exp_id
    os.makedirs(run_dir, exist_ok=True)

    # Quantization Config =>> only if LoRA fine-tuning
    quantization_config = None
    if cfg.use_quantization:
        assert cfg.use_lora, "Quantized training only supported for LoRA fine-tuning!"
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4"
        )

    # 加载 processor（从原始 OpenVLA 路径）和 CoordiVLA 模型（从单臂预训练权重初始化）
    processor = AutoProcessor.from_pretrained(cfg.openvla_path, trust_remote_code=True)
    vla = CoordiVLAForActionPrediction.from_single_arm_pretrained(
        cfg.openvla_path,
        CoordiVLAConfig.from_pretrained(
            cfg.openvla_path,
            use_coordination=cfg.use_coordination,
            coordination_summary_tokens=cfg.coordination_summary_tokens,
            coordination_num_layers=cfg.coordination_num_layers,
            coordination_fusion=cfg.coordination_fusion,
            coordination_include_self_action_prefix=cfg.coordination_include_self_action_prefix,
            coordination_prefix_memory=cfg.coordination_prefix_memory,
        ),
        torch_dtype=torch.bfloat16,
    )

    # Device Placement =>> note that BitsAndBytes automatically handles for quantized training
    if cfg.use_quantization:
        vla = prepare_model_for_kbit_training(vla)
    else:
        vla = vla.to(device_id)

    # [LoRA] Wrap Model w/ PEFT `LoraConfig` =>> by default we set `target_modules=all-linear`
    if cfg.use_lora:
        # 只对 LLM 主干线性层做 LoRA，coordination_module 排除在外（后面手动解冻全参数更新）
        lora_target = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=min(cfg.lora_rank, 16),
            lora_dropout=cfg.lora_dropout,
            target_modules=lora_target,
            init_lora_weights="gaussian",
        )
        vla = get_peft_model(vla, lora_config)
        vla.print_trainable_parameters()
        # 验证 coord 模块是否被 LoRA 包住（应全为 False）
        print("=== coord module requires_grad before unfreeze ===")
        for n, p in vla.named_parameters():
            if 'coordination_module' in n:
                print(n, p.requires_grad)
        print("=== end ===")


    # 开启梯度检查点（必须在 get_peft_model 之后，通过 get_base_model() 访问双臂 LLM）
    base = vla.get_base_model() if cfg.use_lora else vla
    base.llm_left.gradient_checkpointing_enable()
    base.llm_right.gradient_checkpointing_enable()
    base.llm_left.config.use_cache = False
    base.llm_right.config.use_cache = False

    # DDP 初始化前先解冻 coordination_module，并让 alpha 保持 fp32。
    # alpha 是标量门控；如果跟随模型转成 bf16，1e-4 量级更新会在 1.0 附近被舍入掉。
    use_coordination = base.use_coordination
    if use_coordination:
        # The coordination module is newly trained. Keep it in fp32 so AdamW updates
        # do not get rounded away around small Xavier/query initializations.
        coord_mod_base = base.coordination_module.to(torch.float32)
        use_alpha_fusion = getattr(coord_mod_base, "fusion", "residual") == "residual"
        if use_alpha_fusion:
            coord_mod_base.alpha_left = torch.nn.Parameter(coord_mod_base.alpha_left.detach().float())
            coord_mod_base.alpha_right = torch.nn.Parameter(coord_mod_base.alpha_right.detach().float())
        for n, p in vla.named_parameters():
            if 'coordination_module' in n:
                p.requires_grad = True
        if not use_alpha_fusion:
            coord_mod_base.alpha_left.requires_grad = False
            coord_mod_base.alpha_right.requires_grad = False
        coord_param_count = sum(p.numel() for p in coord_mod_base.parameters())
        coord_trainable_count = sum(p.numel() for p in coord_mod_base.parameters() if p.requires_grad)
        print(f"coordination_module params: {coord_param_count:,} total, {coord_trainable_count:,} trainable, dtype=fp32")
        if getattr(coord_mod_base, "fusion", "residual") == "meta_query":
            print(
                "coordination_include_self_action_prefix: "
                f"{getattr(coord_mod_base, 'include_self_action_prefix', True)}"
            )
            print(
                "coordination_prefix_memory: "
                f"{getattr(base.config, 'coordination_prefix_memory', False)}"
            )

    base.config.train_projector = TRAIN_PROJECTOR
    if TRAIN_PROJECTOR:
        for n, p in vla.named_parameters():
            if "projector_left" in n or "projector_right" in n:
                p.requires_grad = True
        projector_param_count = sum(
            p.numel()
            for n, p in base.named_parameters()
            if "projector_left" in n or "projector_right" in n
        )
        projector_trainable_count = sum(
            p.numel()
            for n, p in base.named_parameters()
            if ("projector_left" in n or "projector_right" in n) and p.requires_grad
        )
        print(
            f"projector_left/right params: {projector_param_count:,} total, "
            f"{projector_trainable_count:,} trainable"
        )

    # Wrap VLA in PyTorch DDP Wrapper for Multi-GPU Training
    vla = DDP(vla, device_ids=[device_id], find_unused_parameters=True, gradient_as_bucket_view=True)

    if use_coordination:
        coord_mod_base = vla.module.get_base_model().coordination_module

        # Create optimizer. Alpha is only used by the legacy residual-fusion module.
        use_alpha_fusion = getattr(coord_mod_base, "fusion", "residual") == "residual"
        alpha_params = [coord_mod_base.alpha_left, coord_mod_base.alpha_right] if use_alpha_fusion else []
        coord_params = [p for n, p in vla.named_parameters()
                        if 'coordination_module' in n
                        and 'alpha_left' not in n and 'alpha_right' not in n
                        and p.requires_grad]
        projector_params = [p for n, p in vla.named_parameters()
                            if ('projector_left' in n or 'projector_right' in n)
                            and p.requires_grad]
        lora_params  = [p for n, p in vla.named_parameters()
                        if 'coordination_module' not in n
                        and 'projector_left' not in n and 'projector_right' not in n
                        and 'alpha_left' not in n and 'alpha_right' not in n
                        and p.requires_grad]
        param_groups = [
            {'params': lora_params,  'lr': cfg.learning_rate},
            {'params': coord_params, 'lr': cfg.learning_rate / 2},
            {'params': projector_params, 'lr': cfg.learning_rate / 2},
        ]
        if use_alpha_fusion:
            alpha_group_idx = len(param_groups)
            param_groups.append({'params': alpha_params, 'lr': 0.0, 'weight_decay': 0.0})
        else:
            alpha_group_idx = None
        optimizer = AdamW(param_groups)
        print(f"coordination_fusion: {getattr(coord_mod_base, 'fusion', 'residual')}")
    else:
        lora_params = [p for p in vla.parameters() if p.requires_grad]
        optimizer = AdamW(lora_params, lr=cfg.learning_rate)
        alpha_group_idx = None
    # 主干 LoRA：warmup + cosine decay
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-8, end_factor=1.0, total_iters=cfg.warmup_steps
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.max_steps - cfg.warmup_steps, eta_min=1e-6
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[cfg.warmup_steps]
    )

    # Create Action Tokenizer
    action_tokenizer = ActionTokenizer(processor.tokenizer)

    # 构建 RoboTwin 数据集（map-style，支持 shuffle）
    prompt_builder_fn = PurePromptBuilder if "v01" not in cfg.openvla_path else VicunaV15ChatPromptBuilder
    vla_dataset = RoboTwinDataset(
        data_dir=cfg.data_dir,
        task_name=cfg.task_name,
        action_tokenizer=action_tokenizer,
        base_tokenizer=processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=prompt_builder_fn,
        image_aug=cfg.image_aug,
    )

    # [Important] Save Dataset Statistics =>> used to de-normalize actions for inference!
    if distributed_state.is_main_process:
        save_dataset_statistics(vla_dataset.dataset_statistics, run_dir)

    # 构建 CoordiVLA Collator 和 DataLoader
    collator = CoordiVLACollator(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )
    dataloader = DataLoader(
        vla_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=4,
    )

    # Initialize Logging =>> W&B
    if distributed_state.is_main_process:
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=f"ft+{exp_id}")

    # Deque to store recent train metrics (used for computing smoothened metrics for gradient accumulation)
    recent_losses = deque(maxlen=cfg.grad_accumulation_steps)
    recent_action_losses = deque(maxlen=cfg.grad_accumulation_steps)
    recent_action_accuracies = deque(maxlen=cfg.grad_accumulation_steps)
    recent_l1_losses = deque(maxlen=cfg.grad_accumulation_steps)

    # Train!
    with tqdm.tqdm(total=cfg.max_steps, leave=False) as progress:
        vla.train()
        optimizer.zero_grad()
        gradient_step_idx = 0
        epoch = 0
        while True:
            epoch += 1
            for batch_idx, batch in enumerate(dataloader):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output = vla(
                        pixel_values_global=batch["pixel_values_global"].to(torch.bfloat16).to(device_id),
                        pixel_values_wrist_left=batch["pixel_values_wrist_left"].to(torch.bfloat16).to(device_id),
                        pixel_values_wrist_right=batch["pixel_values_wrist_right"].to(torch.bfloat16).to(device_id),
                        input_ids_left=batch["input_ids_left"].to(device_id),
                        input_ids_right=batch["input_ids_right"].to(device_id),
                        attention_mask_left=batch["attention_mask_left"].to(device_id),
                        attention_mask_right=batch["attention_mask_right"].to(device_id),
                        labels_left=batch["labels_left"].to(device_id),
                        labels_right=batch["labels_right"].to(device_id),
                    )
                    loss = output.loss

                # Alpha 惩罚项：1000步后才加，冻结期间不干扰其他参数
                # Normalize loss to account for gradient accumulation
                normalized_loss = loss / cfg.grad_accumulation_steps

                # Backward pass
                normalized_loss.backward()

                # Compute Accuracy and L1 Loss for Logging（左右臂各算，合并统计）
                # num_patches：全局图 + 腕部图各自的 patch 数之和
                # 用 projector_left 的实际输出维度计算，兼容 fused backbone（DINOv2+SigLIP patch 数不同）
                # logits shape: (B, num_patches + seq_len, vocab)
                # labels shape: (B, seq_len)
                # 对齐：取 logits 后 seq_len-1 个位置（shift by 1）
                seq_len = batch["labels_left"].shape[1]
                action_logits_left  = output.logits_left[:,  -(seq_len):-1, :]
                action_logits_right = output.logits_right[:, -(seq_len):-1, :]
                action_preds_left  = action_logits_left.argmax(dim=2)
                action_preds_right = action_logits_right.argmax(dim=2)
                action_gt_left  = batch["labels_left"][:, 1:].to(device_id)
                action_gt_right = batch["labels_right"][:, 1:].to(device_id)
                mask_left  = action_gt_left  > action_tokenizer.action_token_begin_idx
                mask_right = action_gt_right > action_tokenizer.action_token_begin_idx

                # Compute Accuracy
                correct_left  = (action_preds_left  == action_gt_left)  & mask_left
                correct_right = (action_preds_right == action_gt_right) & mask_right
                denom = (mask_left.sum() + mask_right.sum()).float()
                action_accuracy = (correct_left.sum() + correct_right.sum()).float() / denom.clamp(min=1)

                # Compute L1 Loss on Predicted (Continuous) Actions
                preds_cat = torch.cat([action_preds_left[mask_left], action_preds_right[mask_right]]).cpu().numpy()
                gt_cat    = torch.cat([action_gt_left[mask_left],    action_gt_right[mask_right]]).cpu().numpy()
                continuous_actions_pred = torch.tensor(action_tokenizer.decode_token_ids_to_actions(preds_cat))
                continuous_actions_gt   = torch.tensor(action_tokenizer.decode_token_ids_to_actions(gt_cat))
                action_l1_loss = torch.nn.functional.l1_loss(continuous_actions_pred, continuous_actions_gt)

                # Store recent train metrics
                recent_losses.append(loss.item())
                recent_action_losses.append(output.loss.item())
                recent_action_accuracies.append(action_accuracy.item())
                recent_l1_losses.append(action_l1_loss.item())

                # Compute smoothened train metrics
                smoothened_loss = sum(recent_losses) / len(recent_losses)
                smoothened_action_loss = sum(recent_action_losses) / len(recent_action_losses)
                smoothened_action_accuracy = sum(recent_action_accuracies) / len(recent_action_accuracies)
                smoothened_l1_loss = sum(recent_l1_losses) / len(recent_l1_losses)

                # Optimizer Step
                if (batch_idx + 1) % cfg.grad_accumulation_steps == 0:
                    alpha_left_grad_value = None
                    alpha_right_grad_value = None
                    if use_coordination and use_alpha_fusion:
                        if coord_mod_base.alpha_left.grad is not None:
                            alpha_left_grad_value = coord_mod_base.alpha_left.grad.detach().float().item()
                        if coord_mod_base.alpha_right.grad is not None:
                            alpha_right_grad_value = coord_mod_base.alpha_right.grad.detach().float().item()

                    torch.nn.utils.clip_grad_norm_(vla.parameters(), cfg.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    gradient_step_idx += 1

                    # 1000步后激活 alpha lr（scheduler 会覆盖，每步强制设回）
                    if use_coordination and use_alpha_fusion and gradient_step_idx >= 1000:
                        optimizer.param_groups[alpha_group_idx]['lr'] = cfg.learning_rate
                    if use_coordination and use_alpha_fusion and gradient_step_idx == 1000:
                        print("Step 1000: alpha lr activated")
                    if use_coordination and use_alpha_fusion and gradient_step_idx == 1001:
                        print(f"alpha_left grad before step: {alpha_left_grad_value}")
                        print(f"alpha_right grad before step: {alpha_right_grad_value}")
                        print(f"alpha_left value: {coord_mod_base.alpha_left.item()}")
                    progress.update()

                    # Push Metrics to W&B (every 10 gradient steps)
                    if distributed_state.is_main_process and gradient_step_idx % 10 == 0:
                        log_dict = {
                            "train_loss": smoothened_loss,
                            "action_accuracy": smoothened_action_accuracy,
                            "l1_loss": smoothened_l1_loss,
                        }
                        if use_coordination:
                            coord_mod = vla.module.get_base_model().coordination_module
                            if use_alpha_fusion:
                                log_dict["alpha_left"]  = coord_mod.alpha_left.item()
                                log_dict["alpha_right"] = coord_mod.alpha_right.item()
                                log_dict["alpha_lr"] = optimizer.param_groups[alpha_group_idx]["lr"]
                            for stat_name, stat_value in coord_mod.get_coord_stats().items():
                                log_dict[stat_name] = stat_value.float().item()
                            if alpha_left_grad_value is not None:
                                log_dict["alpha_left_grad"] = alpha_left_grad_value
                            if alpha_right_grad_value is not None:
                                log_dict["alpha_right_grad"] = alpha_right_grad_value

                            # 每 200 步记录一次 cross-attn 热力图
                            if gradient_step_idx % 200 == 0:
                                lw, rw = coord_mod.get_attn_weights()
                                if lw is not None:
                                    for name, w in [("left_queries_right", lw), ("right_queries_left", rw)]:
                                        arr = w[0].float().cpu().numpy()
                                        vmax = float(np.percentile(arr, 99))
                                        if not np.isfinite(vmax) or vmax <= 0:
                                            vmax = float(arr.max()) if arr.size else 1.0
                                        vmax = max(vmax, 1e-6)
                                        fig, ax = plt.subplots(figsize=(4, 4))
                                        im = ax.imshow(arr, aspect="auto", cmap="viridis", interpolation="nearest", vmin=0, vmax=vmax)
                                        ax.set_xlabel("Key (source arm)")
                                        ax.set_ylabel("Query (target arm)")
                                        title_prefix = "prefix_attn" if getattr(coord_mod, "fusion", "residual") == "meta_query" else "cross_attn"
                                        ax.set_title(f"{title_prefix}/{name} step={gradient_step_idx}, vmax={vmax:.2e}")
                                        plt.colorbar(im, ax=ax)
                                        plt.tight_layout()
                                        log_dict[f"{title_prefix}/{name}"] = wandb.Image(fig)
                                        plt.close(fig)
                        wandb.log(log_dict, step=gradient_step_idx)

                    # Save Model Checkpoint
                    if gradient_step_idx % cfg.save_steps == 0:
                        if distributed_state.is_main_process:
                            print(f"Saving Model Checkpoint for Step {gradient_step_idx}")
                            save_dir = adapter_dir if cfg.use_lora else run_dir
                            processor.save_pretrained(run_dir)
                            vla.module.save_pretrained(save_dir)
                            write_coordination_metadata(vla.module, save_dir)
                            if use_coordination:
                                coord_state = vla.module.get_base_model().coordination_module.state_dict()
                                torch.save(coord_state, f"{save_dir}/coordination_module.pt")
                                print(f"Saved coordination_module weights to {save_dir}/coordination_module.pt")
                            if TRAIN_PROJECTOR:
                                base_model = vla.module.get_base_model()
                                projector_state = {
                                    "projector_left": base_model.projector_left.state_dict(),
                                    "projector_right": base_model.projector_right.state_dict(),
                                }
                                torch.save(projector_state, f"{save_dir}/projector.pt")
                                print(f"Saved projector weights to {save_dir}/projector.pt")

                        dist.barrier()

                        if cfg.use_lora:
                            if distributed_state.is_main_process:
                                print(f"Saved adapter checkpoint for Step {gradient_step_idx} at: {adapter_dir}")

                        dist.barrier()

                # Stop training when max_steps is reached
                if gradient_step_idx >= cfg.max_steps:
                    print(f"Max step {cfg.max_steps} reached! Stopping training...")
                    break

            if gradient_step_idx >= cfg.max_steps:
                break

    # 训练结束后强制保存最终 checkpoint（避免 max_steps 不是 save_steps 整数倍时丢失最后几步）
    if distributed_state.is_main_process:
        print(f"Saving final checkpoint at step {gradient_step_idx}")
        save_dir = adapter_dir if cfg.use_lora else run_dir
        processor.save_pretrained(run_dir)
        vla.module.save_pretrained(save_dir)
        write_coordination_metadata(vla.module, save_dir)
        if use_coordination:
            coord_state = vla.module.get_base_model().coordination_module.state_dict()
            torch.save(coord_state, f"{save_dir}/coordination_module.pt")
            print(f"Saved coordination_module weights to {save_dir}/coordination_module.pt")
        if TRAIN_PROJECTOR:
            base_model = vla.module.get_base_model()
            projector_state = {
                "projector_left": base_model.projector_left.state_dict(),
                "projector_right": base_model.projector_right.state_dict(),
            }
            torch.save(projector_state, f"{save_dir}/projector.pt")
            print(f"Saved projector weights to {save_dir}/projector.pt")
    dist.barrier()


if __name__ == "__main__":
    finetune()
