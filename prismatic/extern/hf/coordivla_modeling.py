"""
coordivla_modeling.py

CoordiVLA HuggingFace-style 模型定义。
继承自 PrismaticPreTrainedModel，实现双臂协调的 VLA 推理与训练接口。
包含：CrossArmCoordinationModule（跨臂协调注意力）、CoordiVLAForActionPrediction（主模型）。

References [LLaVa, IDEFICS-2]:
    => https://github.com/huggingface/transformers/blob/main/src/transformers/models/llava/modeling_llava.py
    => https://github.com/huggingface/transformers/blob/main/src/transformers/models/idefics2/modeling_idefics2.py
"""

import logging
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, ClassVar, Dict, List, Optional, Tuple, Union
# 以下两行为 CoordiVLA 新增导入
from .coordivla_configuration import CoordiVLAConfig
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

import numpy as np
import timm
import tokenizers
import torch
import torch.nn as nn
import transformers
from timm.models.vision_transformer import LayerScale
from transformers import AutoModelForCausalLM, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import ModelOutput

from .configuration_prismatic import OpenVLAConfig, PrismaticConfig

# Get Logger
logger = logging.getLogger(__name__)


# === PyTorch/HuggingFace Default IGNORE_INDEX (for CrossEntropyLoss labels)
IGNORE_INDEX = -100


# === Utility Functions for Monkey-Patching ===
def unpack_tuple(fn: Callable[[Any], Tuple[Any]]) -> Callable[[Any], Any]:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        result = fn(*args, **kwargs)
        return result[0] if isinstance(result, tuple) else result

    return wrapper


# HF Transformers overwrites parameters with names containing `gamma`; we're going to patch VisionBackbone.LayerScale.
#   =>> TIMM :: https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py#L109
#   =>> Transformers :: https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py#L3960
def _ls_new_forward(self, x: torch.Tensor) -> torch.Tensor:
    return x.mul_(self.scale_factor) if self.inplace else x * self.scale_factor


def ls_apply_patch(ls_module: LayerScale):
    ls_module.scale_factor = nn.Parameter(ls_module.gamma.clone())
    ls_module.forward = _ls_new_forward.__get__(ls_module, LayerScale)
    del ls_module.gamma

# === Prismatic Vision Backbone (nn.Module) Definitions (w/ Fused Backbone Support) ===
class PrismaticVisionBackbone(nn.Module):
    def __init__(
        self,
        use_fused_vision_backbone: bool,
        image_sizes: List[int],
        timm_model_ids: List[str],
        timm_override_act_layers: List[Optional[str]],
    ) -> None:
        super().__init__()
        self.use_fused_vision_backbone = use_fused_vision_backbone

        # [Contract] Validate number of (fused) vision backbones, create "alpha" featurizer and Instantiate
        #   =>> Note :: Monkey-Patch the `forward()` function of the backbone to ensure FSDP-compatibility
        #               Hardcodes `get_intermediate_layers` to return the **SECOND-TO-LAST** layer patches!
        assert len(timm_model_ids) <= 2, "Prismatic models only support up to 2 (fused) vision backbones!"
        self.featurizer = timm.create_model(
            timm_model_ids[0],
            pretrained=False,
            num_classes=0,
            img_size=image_sizes[0],
            act_layer=timm_override_act_layers[0],
        )
        self.featurizer.forward = unpack_tuple(
            partial(self.featurizer.get_intermediate_layers, n={len(self.featurizer.blocks) - 2})
        )
        self.embed_dim = self.featurizer.embed_dim

        # If `use_fused_vision_backbone` =>> create "beta" featurizer
        if self.use_fused_vision_backbone:
            self.fused_featurizer = timm.create_model(
                timm_model_ids[1],
                pretrained=False,
                num_classes=0,
                img_size=image_sizes[1],
                act_layer=timm_override_act_layers[1],
            )
            self.fused_featurizer.forward = unpack_tuple(
                partial(self.fused_featurizer.get_intermediate_layers, n={len(self.fused_featurizer.blocks) - 2})
            )
            self.embed_dim += self.fused_featurizer.embed_dim

        # Patch `vision_backbone.featurizer` and `vision_backbone.fused_featurizer` with HF-Compatible LayerScale
        for module in self.featurizer.modules():
            if isinstance(module, LayerScale):
                ls_apply_patch(module)

        if self.use_fused_vision_backbone:
            for module in self.fused_featurizer.modules():
                if isinstance(module, LayerScale):
                    ls_apply_patch(module)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Run image (`pixel_values`) through featurizer; if channel-stacked, then dispatch and sequence stack."""
        if not self.use_fused_vision_backbone:
            return self.featurizer(pixel_values)

        # Split `pixel_values :: [bsz, 2 * 3, resolution, resolution]` =>> featurize =>> channel stack
        img, img_fused = torch.split(pixel_values, [3, 3], dim=1)
        patches, patches_fused = self.featurizer(img), self.fused_featurizer(img_fused)

        return torch.cat([patches, patches_fused], dim=2)


# === Prismatic Projector (nn.Module) Definitions ===
class PrismaticProjector(nn.Module):
    def __init__(self, use_fused_vision_backbone: bool, vision_dim: int, llm_dim: int) -> None:
        super().__init__()
        self.use_fused_vision_backbone = use_fused_vision_backbone
        self.vision_dim, self.llm_dim = vision_dim, llm_dim


        # Switch on `use_fused_vision_backbone` =>> use slightly different MLPs and projection factors!
        if not self.use_fused_vision_backbone:
            self.fc1 = nn.Linear(self.vision_dim, self.llm_dim, bias=True)
            self.fc2 = nn.Linear(self.llm_dim, self.llm_dim, bias=True)
            self.act_fn1 = nn.GELU()
        else:
            initial_projection_dim = 4 * vision_dim
            self.fc1 = nn.Linear(self.vision_dim, initial_projection_dim, bias=True)
            self.fc2 = nn.Linear(initial_projection_dim, self.llm_dim, bias=True)
            self.fc3 = nn.Linear(self.llm_dim, self.llm_dim, bias=True)
            self.act_fn1 = nn.GELU()
            self.act_fn2 = nn.GELU()

    def forward(self, img_patches: torch.Tensor) -> torch.Tensor:
        if not self.use_fused_vision_backbone:
            projected_features = self.fc1(img_patches)
            projected_features = self.act_fn1(projected_features)
            projected_features = self.fc2(projected_features)
        else:
            projected_features = self.fc1(img_patches)
            projected_features = self.act_fn1(projected_features)
            projected_features = self.fc2(projected_features)
            projected_features = self.act_fn2(projected_features)
            projected_features = self.fc3(projected_features)

        return projected_features


# === Main HF Class Definitions ===
@dataclass
class PrismaticCausalLMOutputWithPast(ModelOutput):
    """Base class for Prismatic casual (visually-conditioned) language model outputs; also exposes visual features."""

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

    # Additions for VLMs
    projector_features: Optional[torch.FloatTensor] = None


class PrismaticPreTrainedModel(PreTrainedModel):
    config_class: PretrainedConfig = PrismaticConfig
    base_model_prefix: str = "model"
    supports_gradient_checkpointing: bool = True

    _no_split_modules: ClassVar[List[str]] = ["PrismaticProjector"]
    _skip_keys_device_placement: str = "past_key_values"
    _supports_flash_attn_2: bool = True

    def _init_weights(self, module: nn.Module) -> None:
        # Important :: this HF ported version is *not* meant for training from scratch; only inference and fine-tuning!
        #   => As such, this init_weights code is not correct; if training VLMs from scratch, use the main codebase at
        #      https://github.com/TRI-ML/prismatic-vlms
        std = (
            self.config.initializer_range
            if hasattr(self.config, "initializer_range")
            else self.config.text_config.initializer_range
        )

        if hasattr(module, "class_embedding"):
            module.class_embedding.data.normal_(mean=0.0, std=std)

        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    @property
    def _supports_sdpa(self) -> bool:
        """Check LLM supports SDPA Attention"""
        return self.language_model._supports_sdpa


class PrismaticForConditionalGeneration(PrismaticPreTrainedModel):
    def __init__(self, config: PrismaticConfig) -> None:
        super().__init__(config)

        # [Validation] Lightweight Validate on `config` Fields + Dependency Versions
        if config.use_fused_vision_backbone is None:
            raise ValueError("Missing config field `use_fused_vision_backbone`")

        if timm.__version__ not in {"0.9.10", "0.9.11", "0.9.12", "0.9.16"}:
            raise NotImplementedError(
                "TIMM Version must be >= 0.9.10 and < 1.0.0 (breaking); please raise a GitHub Issue "
                "if you urgently need support for latest TIMM versions."
            )

        if (transformers.__version__ != "4.40.1") or (tokenizers.__version__ != "0.19.1"):
            logger.warning(
                f"Expected `transformers==4.40.1` and `tokenizers==0.19.1` but got "
                f"`transformers=={transformers.__version__}` and `tokenizers=={tokenizers.__version__}`; "
                f"there might be inference-time regressions due to dependency changes. If in doubt, please"
                f"use the above versions."
            )

        # Instantiate PrismaticVisionBackbone (w/ Potential Fused Backbone)
        self.vision_backbone = PrismaticVisionBackbone(
            config.use_fused_vision_backbone, config.image_sizes, config.timm_model_ids, config.timm_override_act_layers
        )

        # Create Multimodal Projector
        self.projector = PrismaticProjector(
            config.use_fused_vision_backbone,
            vision_dim=self.vision_backbone.embed_dim,
            llm_dim=config.text_config.hidden_size,
        )

        # Instantiate LLM Backbone
        self.language_model = AutoModelForCausalLM.from_config(
            config.text_config, attn_implementation=config._attn_implementation
        )
        self.vocab_size = config.text_config.vocab_size
        self.pad_token_id = config.pad_token_id

        # HF Boilerplate =>> initializes weights via `_init_weights()` and sets gradient checkpointing
        self.post_init()

    # === `PreTrainedModel` Boilerplate ===
    def get_input_embeddings(self) -> nn.Module:
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self) -> nn.Module:
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings: nn.Module) -> None:
        self.language_model.set_output_embeddings(new_embeddings)

    def get_decoder(self) -> nn.Module:
        return self.language_model.get_decoder()

    def set_decoder(self, decoder: nn.Module) -> None:
        self.language_model.set_decoder(decoder)

    def tie_weights(self) -> None:
        self.language_model.tie_weights()  # Note: `Llama-2` and `Mistral` don't tie weights (no-op)

    def resize_token_embeddings(
        self, new_num_tokens: Optional[int] = None, pad_to_multiple_of: Optional[int] = None
    ) -> nn.Embedding:
        updated_embeddings = self.language_model.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)

        # Update config/instance variables
        self.config.text_config.vocab_size = updated_embeddings.num_embeddings
        self.vocab_size = updated_embeddings.num_embeddings

        return updated_embeddings

    # === Core Prismatic VLM `forward()` Logic ===
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_projector_features: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, PrismaticCausalLMOutputWithPast]:
        """Run a forward pass through the VLM, returning a PrismaticCausalLMOutputWithPast instance."""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_projector_features = output_projector_features if output_projector_features is not None else False
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Respect `use_cache` only if not training (even if `gradient_checkpointing` is off)
        use_cache = use_cache and not self.training

        # Instantiate Placeholder for Projector Features
        projected_patch_embeddings = None

        # Note :: We only support forward passes with the following cases:
        #   => Cached Generation :: (input_ids.shape[1] == 1) and (past_key_values is not None)
        #   => Unimodal Forward :: (pixel_values is None)
        #   => Multimodal Forward :: (pixel_values is not None) and (input_ids/embeds.shape[0] == pixel_values.shape[0])

        # === Handle Generation with Cache (`input_ids.shape[1] == 1`) =>> requires `past_keys_values` ===
        if input_ids.shape[1] == 1:
            assert input_ids.shape[0] == 1, "Generation is only currently supported for batch size of 1!"
            assert past_key_values is not None, "You must provide `past_key_values` during cached generation!"
            assert labels is None, "Unexpected key `labels` provided during cached generation!"

            language_model_output = self.language_model(
                input_ids=input_ids,
                attention_mask=None,
                position_ids=None,
                past_key_values=past_key_values,
                inputs_embeds=None,
                labels=None,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        # === Handle Unimodal Forward ===
        elif pixel_values is None:
            assert (input_ids is not None) and (inputs_embeds is None), "Missing `input_ids` in language-only forward!"
            assert past_key_values is None, "Unexpected key `past_key_values` provided during language-only forward!"

            language_model_output = self.language_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=None,
                past_key_values=None,
                inputs_embeds=None,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        # === Handle Multimodal Forward ===
        elif (input_ids.shape[0] == pixel_values.shape[0]) or (inputs_embeds.shape[0] == pixel_values.shape[0]):
            assert past_key_values is None, "Unexpected key `past_key_values` provided during language-only forward!"

            # Visual Feature Extraction
            patch_features = self.vision_backbone(pixel_values)

            # Projection Logic =>> Update Attention Mask
            projected_patch_embeddings = self.projector(patch_features)
            projected_patch_attention_mask = None
            if attention_mask is not None:
                projected_patch_attention_mask = torch.full(
                    (projected_patch_embeddings.shape[0], projected_patch_embeddings.shape[1]),
                    fill_value=True,
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )

            # Get Input Embeddings (from Language Model Embeddings)
            input_embeddings = self.get_input_embeddings()(input_ids)

            # Build Multimodal Embeddings & Attention Mask =>> Prismatic defaults to inserting after <BOS> token (1:)
            multimodal_embeddings = torch.cat(
                [input_embeddings[:, :1, :], projected_patch_embeddings, input_embeddings[:, 1:, :]], dim=1
            )
            multimodal_attention_mask = None
            if attention_mask is not None:
                multimodal_attention_mask = torch.cat(
                    [attention_mask[:, :1], projected_patch_attention_mask, attention_mask[:, 1:]], dim=1
                )

            # Build Labels (if specified) =>> Ignore Labels for Patch Embeddings
            multimodal_labels = None
            if labels is not None:
                projected_patch_labels = torch.full(
                    (projected_patch_embeddings.shape[0], projected_patch_embeddings.shape[1]),
                    fill_value=IGNORE_INDEX,
                    dtype=labels.dtype,
                    device=labels.device,
                )
                multimodal_labels = torch.cat([labels[:, :1], projected_patch_labels, labels[:, 1:]], dim=1)

            # Dispatch to Language Model
            language_model_output = self.language_model(
                input_ids=None,
                attention_mask=multimodal_attention_mask,
                position_ids=None,
                past_key_values=None,
                inputs_embeds=multimodal_embeddings,
                labels=multimodal_labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        # === Otherwise =>> Assume Invalid! ===
        elif (input_ids.shape[0] != pixel_values.shape[0]) or (inputs_embeds.shape[0] != pixel_values.shape[0]):
            raise ValueError("Non-homogenous batch of (text, image) input -- forward() does not support mixed batches!")

        else:
            raise ValueError(
                "Invalid PrismaticForConditionalGeneration `forward()` call with provided arguments:\n"
                f"=> `input_ids` = {input_ids is not None}\n"
                f"=> `attention_mask` = {attention_mask is not None}\n"
                f"=> `pixel_values` = {pixel_values is not None}\n"
                f"=> `labels` = {labels is not None}\n"
                f"=> `input_embeds` = {inputs_embeds is not None}\n"
                f"=> `past_key_values` = {past_key_values is not None}\n"
                f"=> `use_cache` = {use_cache}"
            )

        # Unpack `language_model_output` and return PrismaticCausalLMOutputWithPast (or tuple if not `return_dict`)
        if not return_dict:
            if output_projector_features and (projected_patch_embeddings is not None):
                return *language_model_output, projected_patch_embeddings

            return language_model_output

        return PrismaticCausalLMOutputWithPast(
            loss=language_model_output.loss,
            logits=language_model_output.logits,
            past_key_values=language_model_output.past_key_values,
            hidden_states=language_model_output.hidden_states,
            attentions=language_model_output.attentions,
            projector_features=projected_patch_embeddings,
        )

    # === GenerationMixin Methods ===
    def prepare_inputs_for_generation(
        self,
        input_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: str,
    ) -> Dict[str, torch.Tensor]:
        """Borrowed from `LlamaForCausalLM` and simplified for batch size = 1; mirrors original PrismaticVLM logic."""
        if ((input_ids is not None) and (input_ids.shape[0] > 1)) or (
            (inputs_embeds is not None) and (inputs_embeds.shape[0] > 1)
        ):
            raise ValueError("Generation with batch size > 1 is not currently supported!")

        # Handle `past_key_values` (cache) =>> assume `input_ids` just has unprocessed tokens
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        # If `input_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"input_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        # Make sure `pixel_values` are preserved in `model_inputs`
        model_inputs.update(
            {
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
            }
        )

        return model_inputs

    # Defer to Language Model (all handle this differently, with different return types)
    def _reorder_cache(self, *args, **kwargs) -> Any:
        return self.language_model._reorder_cache(*args, **kwargs)


class OpenVLAForActionPrediction(PrismaticForConditionalGeneration):
    config_class: PretrainedConfig = OpenVLAConfig

    def __init__(self, config: OpenVLAConfig) -> None:
        super().__init__(config)
        self.norm_stats = config.norm_stats

        # Compute action bins
        self.bins = np.linspace(-1, 1, config.n_action_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0

        # Compute vocab size for de-tokenization -- revert added "multiple of"
        self.vocab_size = self.config.text_config.vocab_size - self.config.pad_to_multiple_of

    def predict_action(
        self, input_ids: Optional[torch.LongTensor] = None, unnorm_key: Optional[str] = None, **kwargs: str
    ) -> np.ndarray:
        """Thin wrapper around .generate() that decodes predicted actions and unnormalizes them."""
        # If the special empty token ('') does not already appear after the colon (':') token in the prompt
        # (after "OUT:" or "ASSISTANT:"), insert it to match the inputs seen at training time
        if not torch.all(input_ids[:, -1] == 29871):
            input_ids = torch.cat(
                (input_ids, torch.unsqueeze(torch.Tensor([29871]).long(), dim=0).to(input_ids.device)), dim=1
            )

        # Run VLA inference
        generated_ids = self.generate(input_ids, max_new_tokens=self.get_action_dim(unnorm_key), **kwargs)

        # Extract predicted action tokens and translate into (normalized) continuous actions
        predicted_action_token_ids = generated_ids[0, -self.get_action_dim(unnorm_key) :].cpu().numpy()
        discretized_actions = self.vocab_size - predicted_action_token_ids
        discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1)
        normalized_actions = self.bin_centers[discretized_actions]

        # Unnormalize actions
        action_norm_stats = self.get_action_stats(unnorm_key)
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )

        return actions

    @staticmethod
    def _check_unnorm_key(norm_stats: Dict[str, Dict[str, Any]], unnorm_key: Optional[str]) -> str:
        if unnorm_key is None:
            assert len(norm_stats) == 1, (
                f"Your model was trained on more than one dataset, "
                f"please pass a `unnorm_key` from the following options to choose the statistics "
                f"used for un-normalizing actions: {norm_stats.keys()}"
            )
            unnorm_key = next(iter(norm_stats.keys()))

        assert unnorm_key in norm_stats, (
            f"The `unnorm_key` you chose is not in the set of available dataset statistics, "
            f"please choose from: {norm_stats.keys()}"
        )
        return unnorm_key

    def get_action_dim(self, unnorm_key: Optional[str] = None) -> int:
        """Get the dimensionality of the policy's action space."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return len(self.norm_stats[unnorm_key]["action"]["q01"])

    def get_action_stats(self, unnorm_key: Optional[str] = None) -> Dict[str, Any]:
        """Get all the logged statistics for the given dataset."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return self.norm_stats[unnorm_key]["action"]
@dataclass
class CoordiVLAOutput(ModelOutput):
    """CoordiVLA输出格式,兼容HuggingFace ModelOutput"""
    loss: Optional[torch.FloatTensor] = None #左右臂loss之和
    logits_left: Optional[torch.FloatTensor] = None #左臂logits（B,seq,vocab)
    logits_right:Optional[torch.FloatTensor] = None #右臂logits（B,seq,vocab)

# 单层协调块：self-attn + cross-attn
class CoordinationLayer(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        self.self_attn_left  = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.self_attn_right = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.cross_attn_left  = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.cross_attn_right = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)

    def forward(self, left_action, right_action, causal_mask):
        left_self,  _ = self.self_attn_left(
            left_action, left_action, left_action, attn_mask=causal_mask, need_weights=False)
        right_self, _ = self.self_attn_right(
            right_action, right_action, right_action, attn_mask=causal_mask, need_weights=False)
        left_out,  _ = self.cross_attn_left(
            left_self, right_action, right_action, need_weights=False)
        right_out, _ = self.cross_attn_right(
            right_self, left_action, left_action, need_weights=False)
        # 残差
        left_action  = left_action  + left_out
        right_action = right_action + right_out
        return left_action, right_action


# CrossArmCoordinationModule:多层跨臂协调模块
class CrossArmCoordinationModule(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, num_layers: int = 3) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [CoordinationLayer(hidden_size, num_heads) for _ in range(num_layers)]
        )
        self.alpha_left  = nn.Parameter(torch.ones(1))   # 初始值 1.0
        self.alpha_right = nn.Parameter(torch.ones(1))   # 初始值 1.0

    def forward(self, left_hidden, right_hidden, action_token_start_idx):
        left_action  = left_hidden[:, action_token_start_idx:, :]
        right_action = right_hidden[:, action_token_start_idx:, :]
        K = left_action.shape[1]

        if K == 0:
            return left_hidden, right_hidden

        causal_mask = torch.triu(
            torch.ones(K, K, dtype=torch.bool, device=left_hidden.device), diagonal=1)

        left_orig  = left_action
        right_orig = right_action
        for layer in self.layers:
            left_action, right_action = layer(left_action, right_action, causal_mask)

        # 最终残差融合
        left_new_action  = left_orig  + self.alpha_left  * left_action
        right_new_action = right_orig + self.alpha_right * right_action

        left_out  = left_hidden.clone()
        right_out = right_hidden.clone()
        left_out[:,  action_token_start_idx:, :] = left_new_action
        right_out[:, action_token_start_idx:, :] = right_new_action
        return left_out, right_out

# CoordiVLAForActionPrediction:主模型
# 【整体架构】
# - 共享 vision_backbone（左右臂看同一个视觉编码器，节省显存）
# - 独立 projector_left / projector_right（各自投影到 LLM 空间）
# - 独立 llm_left / llm_right（各自的 7B LLM）
# - 共享 coordination_module（在 coordination_layer 层之后插入）
#
# 【为什么 vision_backbone 共享？】
# 双臂任务的视觉输入通常来自同一场景，共享编码器减少参数量，
# 且视觉特征本身不需要区分左右臂
#
# 【为什么 projector 不共享？】
# 左右臂的 LLM 权重不同（各自微调），projector 需要各自适配
class CoordiVLAForActionPrediction(PrismaticPreTrainedModel):
    config_class = CoordiVLAConfig
    
    def __init__(self, config:CoordiVLAConfig) -> None:
        super().__init__(config)

        #共享视觉编码器（DINOv2 —— SigLip融合)
        self.vision_backbone = PrismaticVisionBackbone(
            config.use_fused_vision_backbone, config.image_sizes, config.timm_model_ids, config.timm_override_act_layers,
        )

        #独立投影层：视觉特征->LLM隐空间
        self.projector_left = PrismaticProjector(
            config.use_fused_vision_backbone, vision_dim=self.vision_backbone.embed_dim,llm_dim=config.text_config.hidden_size,
        )
        self.projector_right = PrismaticProjector(
            config.use_fused_vision_backbone, vision_dim=self.vision_backbone.embed_dim,llm_dim=config.text_config.hidden_size,
        )
        self.use_coordination = config.use_coordination
        if self.use_coordination:
            self.coordination_module = CrossArmCoordinationModule(
                hidden_size=config.text_config.hidden_size,
                num_heads=config.coordination_num_heads,
            )

        # 独立 LLM（各自的 7B 参数）
        self.llm_left  = AutoModelForCausalLM.from_config(
            config.text_config, attn_implementation=config._attn_implementation
        )
        self.llm_right = AutoModelForCausalLM.from_config(
            config.text_config, attn_implementation=config._attn_implementation
        )

        self.coordination_layer = config.coordination_layer  # 在第 N 层之后插入协调

        # 动作解码相关
        self.norm_stats  = config.norm_stats
        self.bins        = np.linspace(-1, 1, config.n_action_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0
        self.vocab_size  = config.text_config.vocab_size - config.pad_to_multiple_of

        self.post_init()

    def get_input_embeddings(self):
        # PEFT 需要此方法；双臂共用同一词表，返回左臂 embedding 即可
        return self.llm_left.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.llm_left.set_input_embeddings(value)
        self.llm_right.set_input_embeddings(value)

    def _build_multimodal_embeds(self, llm, projector, pixel_values_global, pixel_values_wrist, input_ids, attention_mask):
        assert attention_mask is not None, "attention_mask 不能为空"
        patch_global      = self.vision_backbone(pixel_values_global)
        patch_wrist       = self.vision_backbone(pixel_values_wrist)
        projected_global  = projector(patch_global)
        projected_wrist   = projector(patch_wrist)
        projected = torch.cat([projected_global, projected_wrist], dim=1)
        input_embeds      = llm.get_input_embeddings()(input_ids)
        multimodal_embeds = torch.cat(
            [input_embeds[:, :1, :], projected, input_embeds[:, 1:, :]], dim=1
        )
        patch_mask = torch.ones(
            projected.shape[0], projected.shape[1],
            dtype=attention_mask.dtype, device=attention_mask.device
        )
        multimodal_mask = torch.cat(
            [attention_mask[:, :1], patch_mask, attention_mask[:, 1:]], dim=1
        )
        return multimodal_embeds, multimodal_mask

    def _build_multimodal_labels(self, labels, n_patch):
        patch_ignore = torch.full(
            (labels.shape[0], n_patch), fill_value=IGNORE_INDEX,
            dtype=labels.dtype, device=labels.device
        )
        return torch.cat([labels[:, :1], patch_ignore, labels[:, 1:]], dim=1)

    @staticmethod
    def _get_action_token_start_idx(mm_labels):
        # 返回每个样本各自的 action 起点，List[int]，长度=B
        starts = []
        for i in range(mm_labels.shape[0]):
            non_ignore = (mm_labels[i] != IGNORE_INDEX).nonzero(as_tuple=True)[0]
            if len(non_ignore) == 0:
                raise ValueError(f"sample {i}: labels 中没有找到非 IGNORE_INDEX 的 token")
            starts.append(non_ignore[0].item())
        return starts

    def _apply_coordination_per_sample(self, hidden_left, hidden_right, action_starts):
        # 对 batch 内每个样本单独做 coordination，避免不同样本 action_start 不一致
        outs_left, outs_right = [], []
        for b, s in enumerate(action_starts):
            hl, hr = self.coordination_module(
                hidden_left[b:b+1], hidden_right[b:b+1], s
            )
            outs_left.append(hl)
            outs_right.append(hr)
        return torch.cat(outs_left, dim=0), torch.cat(outs_right, dim=0)

    def _llm_partial_forward(self, llm, inputs_embeds, attention_mask, stop_layer):
        model = llm.model
        B, seq_len, _ = inputs_embeds.shape
        causal_mask  = _prepare_4d_causal_attention_mask(
            attention_mask, (B, seq_len), inputs_embeds, past_key_values_length=0
        )
        position_ids = torch.arange(seq_len, device=inputs_embeds.device).unsqueeze(0).expand(B, -1)
        hidden_states = inputs_embeds
        for idx in range(stop_layer):
            hidden_states = model.layers[idx](
                hidden_states, attention_mask=causal_mask, position_ids=position_ids
            )[0]
        return hidden_states

    def _llm_final_forward(self, llm, hidden_states, attention_mask, labels, start_layer):
        model = llm.model
        B, seq_len, _ = hidden_states.shape
        causal_mask  = _prepare_4d_causal_attention_mask(
            attention_mask, (B, seq_len), hidden_states, past_key_values_length=0
        )
        position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0).expand(B, -1)
        for idx in range(start_layer, len(model.layers)):
            hidden_states = model.layers[idx](
                hidden_states, attention_mask=causal_mask, position_ids=position_ids
            )[0]
        hidden_states = model.norm(hidden_states)
        logits = llm.lm_head(hidden_states)
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous().to(logits.device)
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=IGNORE_INDEX
            )
        return loss, logits

    def forward(
        self,
        input_ids_left=None, input_ids_right=None,
        attention_mask_left=None, attention_mask_right=None,
        pixel_values_global=None,
        pixel_values_wrist_left=None, pixel_values_wrist_right=None,
        labels_left=None, labels_right=None,
        return_dict=None, **kwargs
    ) -> CoordiVLAOutput:
        assert input_ids_left is not None and input_ids_right is not None
        assert attention_mask_left is not None and attention_mask_right is not None
        assert pixel_values_wrist_left is not None and pixel_values_wrist_right is not None
        assert pixel_values_global is not None 

        embeds_left,  mask_left  = self._build_multimodal_embeds(
            self.llm_left,  self.projector_left, pixel_values_global, pixel_values_wrist_left, input_ids_left,  attention_mask_left
        )
        embeds_right, mask_right = self._build_multimodal_embeds(
            self.llm_right, self.projector_right, pixel_values_global, pixel_values_wrist_right, input_ids_right, attention_mask_right
        )

        n_patch = embeds_left.shape[1] - input_ids_left.shape[1]
        mm_labels_left  = self._build_multimodal_labels(labels_left,  n_patch) if labels_left  is not None else None
        mm_labels_right = self._build_multimodal_labels(labels_right, n_patch) if labels_right is not None else None

        if mm_labels_left is not None:
            action_starts = self._get_action_token_start_idx(mm_labels_left)  # List[int], len=B
        else:
            action_starts = [embeds_left.shape[1]] * embeds_left.shape[0]

        N = self.coordination_layer
        hidden_left  = self._llm_partial_forward(self.llm_left,  embeds_left,  mask_left,  stop_layer=N + 1)
        hidden_right = self._llm_partial_forward(self.llm_right, embeds_right, mask_right, stop_layer=N + 1)

        if self.use_coordination:
            hidden_left, hidden_right = self._apply_coordination_per_sample(hidden_left, hidden_right, action_starts)

        loss_left,  logits_left  = self._llm_final_forward(
            self.llm_left,  hidden_left,  mask_left,  mm_labels_left,  start_layer=N + 1
        )
        loss_right, logits_right = self._llm_final_forward(
            self.llm_right, hidden_right, mask_right, mm_labels_right, start_layer=N + 1
        )

        total_loss = (
            (loss_left + loss_right)
            if (loss_left is not None and loss_right is not None)
            else None
        )
        return CoordiVLAOutput(loss=total_loss, logits_left=logits_left, logits_right=logits_right)

    def predict_action(
        self, input_ids_left, input_ids_right, pixel_values_global,
        pixel_values_wrist_left, pixel_values_wrist_right,
        unnorm_key=None
    ):
        empty = torch.tensor([[29871]], dtype=torch.long)
        if not torch.all(input_ids_left[:, -1] == 29871):
            input_ids_left  = torch.cat([input_ids_left,  empty.to(input_ids_left.device)],  dim=1)
        if not torch.all(input_ids_right[:, -1] == 29871):
            input_ids_right = torch.cat([input_ids_right, empty.to(input_ids_right.device)], dim=1)

        action_dim = self.get_action_dim(unnorm_key)
        mask_l = torch.ones(input_ids_left.shape,  dtype=torch.long, device=input_ids_left.device)
        mask_r = torch.ones(input_ids_right.shape, dtype=torch.long, device=input_ids_right.device)

        embeds_left,  mask_left  = self._build_multimodal_embeds(
            self.llm_left,  self.projector_left,  pixel_values_global, pixel_values_wrist_left,  input_ids_left,  mask_l
        )
        embeds_right, mask_right = self._build_multimodal_embeds(
            self.llm_right, self.projector_right, pixel_values_global, pixel_values_wrist_right, input_ids_right, mask_r
        )

        assert mask_left.shape[0] == 1, "predict_action 只支持 batch size = 1"

        action_start_idx = embeds_left.shape[1]
        B = mask_left.shape[0]
        N = self.coordination_layer
        generated_left, generated_right = [], []

        for _ in range(action_dim):
            h_left  = self._llm_partial_forward(self.llm_left,  embeds_left,  mask_left,  stop_layer=N + 1)
            h_right = self._llm_partial_forward(self.llm_right, embeds_right, mask_right, stop_layer=N + 1)
            if self.use_coordination:
                h_left, h_right = self.coordination_module(h_left, h_right, action_start_idx)
            _, logits_left  = self._llm_final_forward(self.llm_left,  h_left,  mask_left,  None, start_layer=N + 1)
            _, logits_right = self._llm_final_forward(self.llm_right, h_right, mask_right, None, start_layer=N + 1)

            # 只允许从合法 action token 里选，屏蔽文字 token 避免极值动作
            action_token_begin = self.vocab_size - self.bin_centers.shape[0]
            logits_left[:,  -1, :action_token_begin] = float('-inf')
            logits_right[:, -1, :action_token_begin] = float('-inf')
            next_left  = logits_left[:,  -1, :].argmax(dim=-1, keepdim=True)
            next_right = logits_right[:, -1, :].argmax(dim=-1, keepdim=True)
            generated_left.append(next_left[0].item())
            generated_right.append(next_right[0].item())

            embeds_left  = torch.cat([embeds_left,  self.llm_left.get_input_embeddings()(next_left)],  dim=1)
            embeds_right = torch.cat([embeds_right, self.llm_right.get_input_embeddings()(next_right)], dim=1)
            mask_left  = torch.cat([mask_left,  torch.ones(B, 1, dtype=mask_left.dtype,  device=mask_left.device)],  dim=1)
            mask_right = torch.cat([mask_right, torch.ones(B, 1, dtype=mask_right.dtype, device=mask_right.device)], dim=1)

        return (
            self._decode_action(generated_left,  unnorm_key, arm='left'),
            self._decode_action(generated_right, unnorm_key, arm='right')
        )

    def _decode_action(self, token_ids, unnorm_key, arm):
        predicted   = np.array(token_ids)
        discretized = np.clip(
            self.vocab_size - predicted - 1,
            a_min=0, a_max=self.bin_centers.shape[0] - 1
        )
        normalized  = self.bin_centers[discretized]
        stats  = self.get_action_stats(unnorm_key)
        mask   = stats.get("mask", np.ones_like(stats["lq01"], dtype=bool))
        left_hi, left_lo = np.array(stats["lq99"]), np.array(stats["lq01"])
        right_hi, right_lo = np.array(stats["rq99"]), np.array(stats["rq01"])
        return np.where(mask, 0.5 * (normalized + 1) * (left_hi - left_lo) + left_lo, normalized) if arm == 'left' else \
            np.where(mask, 0.5 * (normalized + 1) * (right_hi - right_lo) + right_lo, normalized)

    def get_action_dim(self, unnorm_key=None):
        unnorm_key = OpenVLAForActionPrediction._check_unnorm_key(self.norm_stats, unnorm_key)
        return len(self.norm_stats[unnorm_key]["action"]["lq01"])

    def get_action_stats(self, unnorm_key=None):
        unnorm_key = OpenVLAForActionPrediction._check_unnorm_key(self.norm_stats, unnorm_key)
        return self.norm_stats[unnorm_key]["action"]

    @staticmethod
    def _check_load_coverage(module, missing_keys, threshold, module_name):
        total_keys = list(module.state_dict().keys())
        if len(total_keys) == 0:
            raise RuntimeError(f"[CoordiVLA] 模块 '{module_name}' 没有可检查参数/缓冲区。")
        hit_rate = 1.0 - len(missing_keys) / len(total_keys)
        if hit_rate < threshold:
            raise RuntimeError(
                f"[CoordiVLA] 权重加载失败：'{module_name}' 命中率 {hit_rate:.1%} < 要求 {threshold:.1%}\n"
                f"未命中参数（前5个）: {missing_keys[:5]}"
            )

    @classmethod
    def from_single_arm_pretrained(cls, pretrained_model_name_or_path, config, **kwargs):
        dtype = kwargs.pop("torch_dtype", torch.float32)
        source = OpenVLAForActionPrediction.from_pretrained(pretrained_model_name_or_path, torch_dtype=dtype, **kwargs)
        model  = cls(config).to(dtype=dtype)

        res = model.vision_backbone.load_state_dict(source.vision_backbone.state_dict(), strict=False)
        cls._check_load_coverage(model.vision_backbone, res.missing_keys, 1.0, "vision_backbone")

        res = model.projector_left.load_state_dict(source.projector.state_dict(), strict=False)
        cls._check_load_coverage(model.projector_left, res.missing_keys, 1.0, "projector_left")

        res = model.projector_right.load_state_dict(source.projector.state_dict(), strict=False)
        cls._check_load_coverage(model.projector_right, res.missing_keys, 1.0, "projector_right")

        res = model.llm_left.load_state_dict(source.language_model.state_dict(), strict=False)
        cls._check_load_coverage(model.llm_left, res.missing_keys, 0.95, "llm_left")

        res = model.llm_right.load_state_dict(source.language_model.state_dict(), strict=False)
        cls._check_load_coverage(model.llm_right, res.missing_keys, 0.95, "llm_right")

        model.norm_stats = source.norm_stats
        return model