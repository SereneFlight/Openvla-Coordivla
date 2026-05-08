# OpenVLA-CoordiVLA

CoordiVLA is a dual-arm adaptation of [OpenVLA](https://github.com/openvla/openvla) for RoboTwin bimanual manipulation tasks. It keeps the OpenVLA-style vision-language-action pipeline, splits the action side into left/right branches, and adds a cross-arm coordination module for dual-arm information exchange.

This repository is a research prototype. The main focus is the path from OpenVLA to RoboTwin dual-arm manipulation: action representation, dataset statistics, LoRA fine-tuning, coordination module design, model merging, inference, and debug visualization.

## What This Repo Covers

- Dual-arm OpenVLA baseline with separate left/right action branches.
- CoordiVLA coordination module inserted into intermediate LLM hidden states.
- Meta-query coordination that compresses opposite-arm visual context into summary tokens.
- Optional coordination prefix memory that carries opposite-arm summary tokens through later LLM layers.
- Joint-delta action representation with continuous absolute gripper state.
- RoboTwin training, merge, evaluation, and action-debug workflow.

## Core Idea

The baseline extends OpenVLA from one arm to two arms. Each branch predicts 7 action tokens per arm: 6 joint dimensions and 1 gripper dimension.

CoordiVLA adds a coordination layer between the two branches. In the main `summary16 7x23` setting, each arm first summarizes the opposite-arm visual context with 16 learned meta queries. Those 16 summary tokens are then concatenated with the opposite arm's 7 visible action-prefix tokens, so the coordination module reads a 23-token cross-arm memory when updating the current arm's action hidden states.

The implementation also supports `coordination_prefix_memory=True`. In that mode, the opposite-arm summary tokens are prepended to the hidden sequence for the remaining LLM layers after the coordination module. That is the explicit prefix-memory path. The added prefix tokens are removed again before final action logits are computed.

This is the important distinction:

- `7x23` means 16 opposite-arm summary tokens + 7 visible opposite-arm action-prefix tokens.
- `7x30` means the same 16 summary tokens + 7 opposite-arm action-prefix tokens + 7 same-side action-prefix tokens.
- `prefixmem` means the opposite-arm summary tokens are also carried through later LLM layers.

## Architecture Details

CoordiVLA keeps the OpenVLA multimodal input format, but builds two action prediction branches:

- a shared visual backbone
- separate left/right projectors
- separate left/right LLM branches
- one shared coordination module inserted after `coordination_layer`

The global camera image is paired with each arm's wrist image. The left branch receives global image + left wrist image, while the right branch receives global image + right wrist image. Both branches receive the same language instruction.

The coordination module is applied only to the hidden states that predict action tokens. In causal language-model training, the hidden position before an action token predicts that token, so the implementation first finds the action prediction span from non-ignored labels and then updates only that span.

Training loss is computed on action-token positions only. Prompt tokens, image patch positions, and padding positions are assigned `IGNORE_INDEX`. The final loss is the average of the left-arm and right-arm cross-entropy losses:

```text
loss = 0.5 * (loss_left + loss_right)
```

## Repository Layout

| Path | Description |
|---|---|
| `prismatic/extern/hf/coordivla_configuration.py` | CoordiVLA Hugging Face config. |
| `prismatic/extern/hf/coordivla_modeling.py` | CoordiVLA model and cross-arm coordination module. |
| `vla-scripts/coordivia_finetune.py` | LoRA fine-tuning entry point for RoboTwin dual-arm data. |
| `vla-scripts/compute_dataset_statistics.py` | Computes action normalization statistics from RoboTwin episodes. |
| `vla-scripts/merge_coordivla_lora.py` | Merges LoRA, projector, and coordination weights for inference. |
| `prismatic/vla/robotwin_action_utils.py` | RoboTwin action encoding and decoding helpers. |
| `CoordiVLA/deploy_policy.py` | RoboTwin deployment policy wrapper. |
| `figures/` | Architecture diagrams and figure placeholders. |

## Action Representation

The code uses the OpenVLA-style action encoding:

- joint dimensions: frame-to-frame joint deltas
- gripper dimension: continuous absolute opening state

The encoding name in the code is:

```text
joint_delta_gripper_cont_abs
```

The corresponding mask is:

```text
[True, True, True, True, True, True, False]
```

The gripper dimension is kept absolute and continuous before tokenization and after decoding. It is still generated as one action token by the language model, then de-tokenized back to a continuous value.

During inference, the decoded action has 14 dimensions:

```text
[left_arm_delta(6), left_gripper_abs(1), right_arm_delta(6), right_gripper_abs(1)]
```

The deployment policy adds the joint deltas back to the current qpos and clips the continuous gripper dimensions into the valid range before sending the action to RoboTwin.

## Dataset Format

Training samples are built from RoboTwin HDF5 episodes.

Typical fields used by the loader:

- `observation/head_camera/rgb`
- `observation/left_camera/rgb`
- `observation/right_camera/rgb`
- `joint_action/left_arm`
- `joint_action/left_gripper`
- `joint_action/right_arm`
- `joint_action/right_gripper`

Each sample uses the observation at time `t` and the action target at time `t+1`.

Instructions are sampled from the task's `seen` instructions JSON file.

The current dataset pipeline builds one action label per arm:

- `left_arm[t+1] - left_arm[t]` plus the gripper value at `t+1`
- `right_arm[t+1] - right_arm[t]` plus the gripper value at `t+1`

If a task's demonstrations only contain binary gripper values, the continuous pipeline naturally keeps those values near open/closed states. If a task contains half-open gripper motion, such as `lift_pot`, the same pipeline preserves that continuous openness instead of forcing it into a binary label.

## Dataset Statistics

Action normalization is based on per-arm quantiles:

- left arm: `lq01`, `lq99`
- right arm: `rq01`, `rq99`

The statistics script and the merge script must agree on the same action encoding and mask. If they do not, the decoded action scale will be wrong at inference time.

Example:

```bash
python vla-scripts/compute_dataset_statistics.py \
  --data_dir /path/to/RoboTwin/data/handover_block/demo_clean/data \
  --task_name handover_block \
  --save_path dataset_statistics_handover_block.json
```

## Coordination Modes

The code supports multiple coordination settings.

Main config fields:

| Config | Meaning |
|---|---|
| `use_coordination` | Enables or disables the cross-arm coordination module. |
| `coordination_layer` | LLM layer index after which the coordination module is inserted. |
| `coordination_fusion` | `meta_query` for summary-prefix attention, or `residual` for the legacy residual module. |
| `coordination_summary_tokens` | Number of learned summary query tokens. `16` gives the `summary16` setting. |
| `coordination_num_layers` | Number of coordination layers. Current small-data experiments usually use `1`. |
| `coordination_include_self_action_prefix` | Adds same-side action prefix into the meta-query memory when enabled. |
| `coordination_prefix_memory` | Carries opposite-arm summary tokens through later LLM layers when enabled. |

### `meta_query`

This is the main setting used in the current experiments.

- `coordination_summary_tokens=16`
- opposite-arm visual context is compressed by learned summary queries
- action hidden states read the summary tokens plus the opposite arm's visible action prefix

### `coordination_include_self_action_prefix`

This flag controls whether same-side action prefix tokens are also visible inside the meta-query coordination memory.

- `False` -> `7x23`
- `True` -> `7x30`

### `coordination_prefix_memory`

This flag controls whether the opposite-arm summary tokens are also carried through the remaining LLM layers after the coordination module.

- `False` -> cross-arm information acts only inside the inserted coordination module
- `True` -> opposite-arm summary tokens become explicit prefix memory for later LLM layers

### `residual`

The code also keeps a legacy residual fusion path for comparison.

## Common Experiment Settings

### Dual-arm baseline

The baseline keeps the dual-arm action branches but disables the explicit coordination module:

```bash
torchrun --standalone --nnodes 1 --nproc-per-node 1 \
  vla-scripts/coordivia_finetune.py \
  --openvla_path /path/to/openvla-7b \
  --data_dir /path/to/RoboTwin/data \
  --task_name handover_block \
  --use_coordination False \
  --run_id_note handover_baseline_20k
```

### CoordiVLA `summary16 7x23`

This is the cleaner cross-arm setting:

```bash
torchrun --standalone --nnodes 1 --nproc-per-node 1 \
  vla-scripts/coordivia_finetune.py \
  --openvla_path /path/to/openvla-7b \
  --data_dir /path/to/RoboTwin/data \
  --task_name handover_block \
  --use_coordination True \
  --coordination_fusion meta_query \
  --coordination_summary_tokens 16 \
  --coordination_num_layers 1 \
  --coordination_include_self_action_prefix False \
  --coordination_prefix_memory False \
  --run_id_note handover_summary16_7x23_20k
```

In this mode, each action-query side attends to:

```text
16 opposite-arm visual summary tokens + 7 visible opposite-arm action-prefix tokens
```

### CoordiVLA `summary16 7x23 prefixmem`

This keeps the `7x23` coordination memory and also carries the 16 opposite-arm summary tokens through the remaining LLM layers:

```bash
torchrun --standalone --nnodes 1 --nproc-per-node 1 \
  vla-scripts/coordivia_finetune.py \
  --openvla_path /path/to/openvla-7b \
  --data_dir /path/to/RoboTwin/data \
  --task_name lift_pot \
  --use_coordination True \
  --coordination_fusion meta_query \
  --coordination_summary_tokens 16 \
  --coordination_num_layers 1 \
  --coordination_include_self_action_prefix False \
  --coordination_prefix_memory True \
  --run_id_note lift_pot_summary16_7x23_prefixmem_20k
```

In this mode, the `7x23` attention map still describes the inserted coordination module. The explicit prefix-memory path carries only the summary tokens through later LLM layers.

### CoordiVLA `summary16-self 7x30`

This setting lets each side read same-side action prefix tokens in addition to opposite-arm memory:

```bash
torchrun --standalone --nnodes 1 --nproc-per-node 1 \
  vla-scripts/coordivia_finetune.py \
  --openvla_path /path/to/openvla-7b \
  --data_dir /path/to/RoboTwin/data \
  --task_name handover_block \
  --use_coordination True \
  --coordination_fusion meta_query \
  --coordination_summary_tokens 16 \
  --coordination_num_layers 1 \
  --coordination_include_self_action_prefix True \
  --coordination_prefix_memory False \
  --run_id_note handover_summary16_self_7x30_20k
```

In this mode, each action-query side attends to:

```text
16 opposite-arm visual summary tokens
+ 7 visible opposite-arm action-prefix tokens
+ 7 visible same-side action-prefix tokens
```

## Typical Workflow

1. Compute dataset statistics.
2. Fine-tune OpenVLA with the dual-arm dataset.
3. Merge LoRA, projector, and coordination weights.
4. Run RoboTwin evaluation.
5. Inspect videos, action debug logs, and attention maps.

### Fine-tuning

```bash
cd /path/to/openvla

torchrun --standalone --nnodes 1 --nproc-per-node 1 \
  vla-scripts/coordivia_finetune.py \
  --openvla_path /path/to/openvla-7b \
  --data_dir /path/to/RoboTwin/data \
  --task_name handover_block
```

The fine-tuning script saves:

- LoRA adapter checkpoints
- projector weights if projector training is enabled
- coordination module weights
- metadata describing the coordination settings

Important default training settings in the current script:

| Setting | Default |
|---|---|
| `learning_rate` | `1e-4` |
| `lora_rank` | `32` |
| `lora_dropout` | `0.0` |
| `image_aug` | `True` |
| `TRAIN_PROJECTOR` | `True` |
| `coordination_fusion` | `meta_query` |
| `coordination_summary_tokens` | `16` |
| `coordination_include_self_action_prefix` | `False` |
| `coordination_prefix_memory` | `False` |

### Merge Weights

```bash
python vla-scripts/merge_coordivla_lora.py \
  --openvla_path /path/to/openvla-7b \
  --adapter_dir /path/to/adapter-tmp/<run_name> \
  --save_dir /path/to/merged/<run_name> \
  --stats_path dataset_statistics_handover_block.json
```

The merged package should include:

- merged model weights
- `dataset_statistics.json`
- processor files
- `coordination_module.pt` when coordination is enabled
- `coordination_alpha_fp32.pt` for the legacy residual fusion path

The merge script tries to read coordination metadata from the adapter config. For older checkpoints, pass the coordination flags explicitly during merge so that the merged config matches the training run:

```bash
python vla-scripts/merge_coordivla_lora.py \
  --openvla_path /path/to/openvla-7b \
  --adapter_dir /path/to/adapter-tmp/<run_name> \
  --save_dir /path/to/merged/<run_name> \
  --stats_path dataset_statistics_handover_block.json \
  --use_coordination True \
  --coordination_fusion meta_query \
  --coordination_summary_tokens 16 \
  --coordination_num_layers 1 \
  --coordination_include_self_action_prefix False \
  --coordination_prefix_memory True
```

Change the last two boolean flags to match the exact training run. For example, plain `summary16 7x23` uses `coordination_include_self_action_prefix=False` and `coordination_prefix_memory=False`, while `summary16-self 7x30` uses `coordination_include_self_action_prefix=True`.

### RoboTwin Evaluation

```bash
cd /path/to/RoboTwin

bash policy/CoordiVLA/eval.sh \
  handover_block demo_clean /path/to/merged/<run_name> 0 0
```

The deployment wrapper expects absolute qpos in the order:

- left arm
- left gripper
- right arm
- right gripper

At inference time, the policy converts the decoded delta actions back to qpos before sending them to RoboTwin.

## Debug Outputs

The training and evaluation workflow can be checked with several debug signals:

- `action_accuracy`: token-level action accuracy on supervised action-token positions.
- `l1_loss`: approximate token/action-space L1 metric used for training inspection.
- `prefix_attn/left_queries_right`: attention map for left action queries reading right-side memory.
- `prefix_attn/right_queries_left`: attention map for right action queries reading left-side memory.
- `coord_delta_norm_left/right`: magnitude of the coordination update.
- `coord_hidden_norm_left/right`: magnitude of the original action hidden states.
- `coord_effective_ratio_left/right`: relative size of the effective coordination update.
- `action_debug.csv`: inference-time qpos, decoded action, and raw action checks.

For `summary16 7x23`, the attention map source length is 23.
For `summary16-self 7x30`, the attention map source length is 30.

## Important Files

- `prismatic/vla/robotwin_action_utils.py` handles action encoding and gripper clipping.
- `CoordiVLA/deploy_policy.py` validates `action_encoding` and `mask` before evaluation.
- `vla-scripts/coordivia_finetune.py` builds the dual-arm dataset and training loop.
- `vla-scripts/merge_coordivla_lora.py` reconstructs the merged checkpoint for inference.

## Experiment Naming

Run names usually encode the main settings.

- `handover_block` or `lift_pot` indicates the task.
- `coord-meta_query` indicates meta-query coordination.
- `summary16` indicates 16 summary tokens.
- `7x23` or `7x30` indicates the coordination source shape.
- `prefixmem` indicates explicit prefix memory is enabled.
- `image_aug` indicates image augmentation is enabled during training.

## Current Status

The current experiments mainly focus on:

- `handover_block`
- `lift_pot`

The baseline can already learn part of the approach motion. The harder parts are still gripper timing, gripper holding, fine end-effector positioning, and left/right phase synchronization. On small RoboTwin datasets, CoordiVLA does not always improve clearly over the baseline, which suggests the coordination module still needs more careful validation and tuning.

## Practical Checks

Before running evaluation, check the following:

- `dataset_statistics.json` exists in the merged checkpoint directory.
- `action_encoding` is `joint_delta_gripper_cont_abs`.
- action `mask` is `[True, True, True, True, True, True, False]`.
- the merge-time coordination flags match the training-time flags.
- `coordination_prefix_memory=True` is used only with `coordination_summary_tokens > 0`.
- the RoboTwin task name used as `unnorm_key` exists in `norm_stats`.
- the decoded gripper values stay in the expected continuous range.

## Notes

- This repository is under active research and cleanup.
- Script paths may need small adjustments depending on the local OpenVLA and RoboTwin setup.
- The code assumes access to OpenVLA weights and RoboTwin datasets.
- For fair comparison, keep task configuration, random seeds, maximum inference steps, action encoding, and dataset statistics consistent.

## Acknowledgements

This project builds on OpenVLA and RoboTwin. It also borrows ideas from vision-language-action models, bimanual imitation learning, action chunking, and cross-arm coordination work.
