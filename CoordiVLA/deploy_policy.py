"""
CoordiVLA deploy_policy.py

RoboTwin 评估接口，加载 merge 后的 CoordiVLA 权重，对接仿真环境。
"""

import sys
import os
import csv
import json
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from transformers import AutoConfig, AutoModelForVision2Seq, AutoProcessor, AutoImageProcessor

# 把 openvla 代码路径加进来
OPENVLA_PATH = "/root/openvla"
sys.path.insert(0, OPENVLA_PATH)

from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.extern.hf.coordivla_configuration import CoordiVLAConfig
from prismatic.extern.hf.coordivla_modeling import CoordiVLAForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.robotwin_action_utils import clip_gripper_qpos
from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder

ACTION_ENCODING = "joint_delta_gripper_cont_abs"
ACTION_MASK = [True, True, True, True, True, True, False]
ARM_DIMS = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
RB_SWAP_FOR_LEGACY_HDF5 = os.environ.get("COORDIVLA_RB_SWAP", "1").lower() not in {
    "0",
    "false",
    "no",
    "off",
}

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


def maybe_swap_rb(image):
    image = np.asarray(image)
    if not RB_SWAP_FOR_LEGACY_HDF5:
        return image
    return image[..., [2, 1, 0]].copy()


def encode_obs(obs: dict) -> dict:
    """从 RoboTwin observation 里取出三路图像"""
    return {
        "full_image":        maybe_swap_rb(obs["observation"]["head_camera"]["rgb"]),
        "left_wrist_image":  maybe_swap_rb(obs["observation"]["left_camera"]["rgb"]),
        "right_wrist_image": maybe_swap_rb(obs["observation"]["right_camera"]["rgb"]),
    }


def extract_current_qpos(TASK_ENV, observation: dict):
    joint_action = observation.get("joint_action", {})
    if "vector" in joint_action:
        return np.asarray(joint_action["vector"], dtype=np.float32).reshape(-1)
    if all(k in joint_action for k in ("left_arm", "left_gripper", "right_arm", "right_gripper")):
        return np.concatenate([
            np.asarray(joint_action["left_arm"], dtype=np.float32).reshape(-1),
            np.asarray(joint_action["left_gripper"], dtype=np.float32).reshape(-1)[:1],
            np.asarray(joint_action["right_arm"], dtype=np.float32).reshape(-1),
            np.asarray(joint_action["right_gripper"], dtype=np.float32).reshape(-1)[:1],
        ])
    try:
        left = np.asarray(TASK_ENV.robot.get_left_arm_jointState(), dtype=np.float32).reshape(-1)
        right = np.asarray(TASK_ENV.robot.get_right_arm_jointState(), dtype=np.float32).reshape(-1)
        return np.concatenate([left, right])
    except Exception:
        return None


def convert_delta_action_to_qpos(action: np.ndarray, current_qpos):
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    if current_qpos is None:
        raise RuntimeError("Delta-action inference requires current qpos, but it could not be read from observation/env.")
    current_qpos = np.asarray(current_qpos, dtype=np.float32).reshape(-1)
    if current_qpos.shape != action.shape:
        raise RuntimeError(f"Current qpos shape {current_qpos.shape} does not match action shape {action.shape}.")

    qpos_action = action.copy()
    qpos_action[ARM_DIMS] = current_qpos[ARM_DIMS] + action[ARM_DIMS]
    return clip_gripper_qpos(qpos_action)


def log_action_debug(TASK_ENV, action: np.ndarray, current_qpos, raw_action=None):
    log_dir = getattr(TASK_ENV, "eval_video_path", None)
    if log_dir is None:
        return

    action = np.asarray(action, dtype=np.float32).reshape(-1)
    raw_action = action if raw_action is None else np.asarray(raw_action, dtype=np.float32).reshape(-1)
    if current_qpos is not None:
        current_qpos = np.asarray(current_qpos, dtype=np.float32).reshape(-1)
    if current_qpos is None or current_qpos.shape != action.shape:
        current_qpos = np.full_like(action, np.nan, dtype=np.float32)

    delta = action - current_qpos
    finite_delta = delta[np.isfinite(delta)]
    if finite_delta.size == 0:
        l2_delta = mean_abs_delta = max_abs_delta = np.nan
    else:
        l2_delta = float(np.linalg.norm(finite_delta))
        mean_abs_delta = float(np.mean(np.abs(finite_delta)))
        max_abs_delta = float(np.max(np.abs(finite_delta)))

    log_path = Path(log_dir) / "action_debug.csv"
    write_header = not log_path.exists()
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(
                ["step", "l2_delta", "mean_abs_delta", "max_abs_delta"]
                + [f"pred_{i}" for i in range(action.shape[0])]
                + [f"raw_{i}" for i in range(raw_action.shape[0])]
                + [f"current_{i}" for i in range(action.shape[0])]
                + [f"delta_{i}" for i in range(action.shape[0])]
            )
        writer.writerow(
            [getattr(TASK_ENV, "take_action_cnt", 0), l2_delta, mean_abs_delta, max_abs_delta]
            + action.astype(float).tolist()
            + raw_action.astype(float).tolist()
            + current_qpos.astype(float).tolist()
            + delta.astype(float).tolist()
        )


class CoordiVLAModel:
    def __init__(self, checkpoint_path: str, openvla_path: str, unnorm_key: str = None):
        print(f"加载 CoordiVLA 权重: {checkpoint_path}")
        print("正在加载模型，请稍候（29GB，需要几分钟）...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.unnorm_key = unnorm_key

        # 根据权重路径选 prompt builder（与训练脚本保持一致）
        self.prompt_builder_fn = VicunaV15ChatPromptBuilder if "v01" in openvla_path else PurePromptBuilder

        # 加载 processor 和 action tokenizer
        self.processor = AutoProcessor.from_pretrained(openvla_path, trust_remote_code=True)
        self.action_tokenizer = ActionTokenizer(self.processor.tokenizer)

        # 加载 merge 后的完整模型（device_map=auto 自动分配 CPU+GPU，避免 OOM）
        self.model = CoordiVLAForActionPrediction.from_pretrained(
            checkpoint_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        ).eval()
        if (
            getattr(self.model, "use_coordination", False)
            and getattr(self.model.coordination_module, "fusion", "residual") == "residual"
        ):
            self.model.coordination_module.alpha_left = torch.nn.Parameter(
                self.model.coordination_module.alpha_left.detach().float()
            )
            self.model.coordination_module.alpha_right = torch.nn.Parameter(
                self.model.coordination_module.alpha_right.detach().float()
            )
            alpha_path = Path(checkpoint_path) / "coordination_alpha_fp32.pt"
            if alpha_path.exists():
                alpha_state = torch.load(alpha_path, map_location="cpu")
                self.model.coordination_module.alpha_left.data.copy_(alpha_state["alpha_left"].float())
                self.model.coordination_module.alpha_right.data.copy_(alpha_state["alpha_right"].float())
        stats_path = Path(checkpoint_path) / "dataset_statistics.json"
        if (not isinstance(getattr(self.model, "norm_stats", None), dict)) and stats_path.exists():
            with open(stats_path) as f:
                self.model.norm_stats = json.load(f)
            self.model.config.norm_stats = self.model.norm_stats
        self.device = self._infer_input_device()
        self.action_encoding = self._get_action_encoding()
        if self.action_encoding != ACTION_ENCODING:
            raise RuntimeError(
                f"Expected action_encoding={ACTION_ENCODING!r}, got {self.action_encoding!r}. "
                "Please merge with the updated dataset statistics before evaluation."
            )
        self._check_action_mask()

    def _infer_input_device(self):
        if hasattr(self.model, "hf_device_map"):
            for module_name in ("vision_backbone", "projector_left", "llm_left"):
                device = self.model.hf_device_map.get(module_name)
                if device is not None and str(device) not in {"cpu", "disk"}:
                    return torch.device(device)
        return next(self.model.parameters()).device

    def _get_action_encoding(self):
        norm_stats = getattr(self.model, "norm_stats", None) or getattr(self.model.config, "norm_stats", None)
        if not isinstance(norm_stats, dict):
            return "absolute_qpos"
        if self.unnorm_key is not None and self.unnorm_key not in norm_stats:
            raise RuntimeError(
                f"unnorm_key/task_name {self.unnorm_key!r} not found in norm_stats; available keys: {list(norm_stats.keys())}"
            )
        key = self.unnorm_key if self.unnorm_key in norm_stats else next(iter(norm_stats), None)
        if key is None:
            return "absolute_qpos"
        return norm_stats.get(key, {}).get("action", {}).get("action_encoding", "absolute_qpos")

    def _check_action_mask(self):
        norm_stats = getattr(self.model, "norm_stats", None) or getattr(self.model.config, "norm_stats", None)
        if not isinstance(norm_stats, dict):
            raise RuntimeError("Merged checkpoint is missing norm_stats; merge with --stats_path before evaluation.")
        if self.unnorm_key is not None and self.unnorm_key not in norm_stats:
            raise RuntimeError(
                f"unnorm_key/task_name {self.unnorm_key!r} not found in norm_stats; available keys: {list(norm_stats.keys())}"
            )
        key = self.unnorm_key if self.unnorm_key in norm_stats else next(iter(norm_stats), None)
        action_mask = norm_stats.get(key, {}).get("action", {}).get("mask")
        if action_mask != ACTION_MASK:
            raise RuntimeError(
                f"Expected action mask {ACTION_MASK!r}, got {action_mask!r}. "
                "The gripper dimension must remain absolute, continuous, and unnormalized."
            )

    def get_action(self, obs: dict, instruction: str):
        """推理一步，返回左右臂动作各 7 维"""
        encoded = encode_obs(obs)

        img_global  = Image.fromarray(encoded["full_image"]).convert("RGB")
        img_wrist_l = Image.fromarray(encoded["left_wrist_image"]).convert("RGB")
        img_wrist_r = Image.fromarray(encoded["right_wrist_image"]).convert("RGB")

        pv_global  = self.processor.image_processor.apply_transform(img_global).unsqueeze(0).to(torch.bfloat16).to(self.device)
        pv_wrist_l = self.processor.image_processor.apply_transform(img_wrist_l).unsqueeze(0).to(torch.bfloat16).to(self.device)
        pv_wrist_r = self.processor.image_processor.apply_transform(img_wrist_r).unsqueeze(0).to(torch.bfloat16).to(self.device)

        # 构建 prompt 并 tokenize（和训练时的 human turn 格式保持一致，不加 gpt turn，避免 </s> 截断生成）
        prompt_builder = self.prompt_builder_fn("openvla")
        prompt_text = prompt_builder.add_turn(
            role="human",
            message=f"What action should the robot take to {instruction.lower()}?",
        )
        input_ids = torch.tensor(
            self.processor.tokenizer(prompt_text, add_special_tokens=True).input_ids
        ).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action_left, action_right = self.model.predict_action(
                pixel_values_global=pv_global,
                pixel_values_wrist_left=pv_wrist_l,
                pixel_values_wrist_right=pv_wrist_r,
                input_ids_left=input_ids,
                input_ids_right=input_ids,
                unnorm_key=self.unnorm_key,
            )

        # 拼成 14 维：左臂7 + 右臂7
        action = np.concatenate([action_left, action_right]).astype(np.float32)
        return action


def get_model(usr_args: dict) -> CoordiVLAModel:
    return CoordiVLAModel(
        checkpoint_path=usr_args["checkpoint_path"],
        openvla_path=usr_args.get("openvla_path", os.path.expanduser("~/desktop/openvla-7b")),
        unnorm_key=usr_args.get("task_name", None),
    )


def reset_model(model: CoordiVLAModel):
    pass  # 无状态，不需要 reset


def eval(TASK_ENV, model: CoordiVLAModel, observation: dict):
    instruction = TASK_ENV.get_instruction()
    raw_action = model.get_action(observation, instruction)
    current_qpos = extract_current_qpos(TASK_ENV, observation)
    if model.action_encoding == ACTION_ENCODING:
        action = convert_delta_action_to_qpos(raw_action, current_qpos)
    else:
        action = raw_action
    log_action_debug(TASK_ENV, action, current_qpos, raw_action=raw_action)
    # RoboTwin expects absolute qpos: left arm + left gripper + right arm + right gripper.
    TASK_ENV.take_action(action, action_type="qpos")
