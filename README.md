# OpenVLA-CoordiVLA

CoordiVLA is a dual-arm adaptation of [OpenVLA](https://github.com/openvla/openvla) for RoboTwin bimanual manipulation tasks. The project extends the original single-arm VLA pipeline into left/right action prediction branches and adds a lightweight cross-arm coordination module for studying information exchange between two robot arms.

This repository is mainly a research prototype. It focuses on the engineering path from OpenVLA to RoboTwin dual-arm tasks: dataset statistics, dual-arm action tokenization, LoRA fine-tuning, model merging, RoboTwin inference, and attention/debug visualization.

## Highlights

- Dual-arm OpenVLA baseline with separate left/right action branches.
- CoordiVLA cross-arm coordination module inserted into intermediate LLM hidden states.
- Meta-query summary design: 16 learned queries compress opposite-arm visual context into summary tokens.
- Main `summary16 7x23` setting: 16 opposite-arm visual summary tokens plus 7 visible opposite-arm action-prefix tokens.
- Cross-arm information is used inside the coordination module only; it is not kept as an explicit prefix through later LLM layers.
- Support for `7x23` and `7x30` attention map analysis.
- Joint-delta action representation with absolute gripper state.
- RoboTwin training, merge, inference, and action debug workflow.

## Method Overview

The baseline keeps the OpenVLA visual-language-action pipeline but expands the action side into two branches. Each branch predicts 7 action tokens for one arm: 6 joint dimensions and 1 gripper dimension.

CoordiVLA adds a coordination layer between the two branches. In the main `summary16 7x23` version, each arm first summarizes the opposite-arm visual context with learned meta queries. The resulting 16 summary tokens are concatenated with the opposite arm's 7 visible action-prefix tokens. The current arm reads this 23-token coordination memory inside the coordination module when updating its action hidden states. A causal visibility mask prevents access to future action tokens during training.

This implementation does not use the explicit-prefix variant where cross-arm summary tokens are appended to later LLM layers. Cross-arm information acts only at the inserted coordination module and is then written back into the current action hidden states.

For action representation, the joint dimensions use frame-to-frame joint increments, while the gripper dimension keeps an absolute state. For tasks with continuous gripper motion such as `lift_pot`, the gripper can be kept as continuous absolute openness.

## Main Files

| Path | Description |
|---|---|
| `prismatic/extern/hf/coordivla_configuration.py` | CoordiVLA Hugging Face configuration. |
| `prismatic/extern/hf/coordivla_modeling.py` | CoordiVLA model and cross-arm coordination module. |
| `vla-scripts/coordivia_finetune.py` | LoRA fine-tuning script for RoboTwin dual-arm data. |
| `vla-scripts/compute_dataset_statistics.py` | Computes action normalization statistics. |
| `vla-scripts/merge_coordivla_lora.py` | Merges LoRA/projector/coordination weights for inference. |
| `prismatic/vla/robotwin_action_utils.py` | RoboTwin action encoding and decoding utilities. |
| `CoordiVLA/deploy_policy.py` | RoboTwin deployment policy wrapper. |
| `figures/` | Architecture diagrams and experiment figure placeholders. |

## Training

```bash
cd /path/to/openvla

torchrun --standalone --nnodes 1 --nproc-per-node 1 \
  vla-scripts/coordivia_finetune.py \
  --openvla_path /path/to/openvla-7b \
  --data_dir /path/to/RoboTwin/data \
  --task_name handover_block
```

## Dataset Statistics

```bash
python vla-scripts/compute_dataset_statistics.py \
  --data_dir /path/to/RoboTwin/data/handover_block/demo_clean/data \
  --task_name handover_block \
  --save_path dataset_statistics_handover_block.json
```

The training and inference pipelines must use the same action encoding and dataset statistics. A mismatch here can decode action tokens into the wrong physical scale.

## Merge Weights

```bash
python vla-scripts/merge_coordivla_lora.py \
  --openvla_path /path/to/openvla-7b \
  --adapter_dir /path/to/adapter-tmp/<run_name> \
  --save_dir /path/to/merged/<run_name> \
  --stats_path dataset_statistics_handover_block.json
```

The merged model should include:

- LoRA adapter weights
- projector weights
- coordination module weights
- dataset statistics

## RoboTwin Evaluation

```bash
cd /path/to/RoboTwin

bash policy/CoordiVLA/eval.sh \
  handover_block demo_clean /path/to/merged/<run_name> 0 0
```

During inference, the project can save videos and `action_debug.csv` files for checking joint deltas, gripper values, and failure stages.

## Experiments

The current experiments mainly focus on:

- `handover_block`
- `lift_pot`

Recorded variants include:

| Variant | Coordination Source | Attention Shape |
|---|---|---|
| Dual-arm baseline | No explicit cross-arm module | None |
| CoordiVLA residual | Opposite-arm action hidden states | `7x7` |
| CoordiVLA long context | Opposite-arm long context and prefix | `7xmany` |
| CoordiVLA summary16 | Opposite-arm summary tokens + opposite-arm action prefix | `7x23` |
| CoordiVLA summary16-self | Opposite-arm summary/action prefix + same-side action prefix | `7x30` |

The current results suggest that the baseline can already learn part of the approach motion, while gripper timing, gripper holding, fine end-effector positioning, and left/right phase synchronization remain difficult. The improvement from CoordiVLA is not yet consistently obvious, possibly because OpenVLA is already a strong base model and the newly initialized coordination module can behave like noisy perturbation on small RoboTwin datasets.

## Notes

- This repository is under active research and cleanup.
- Paths in scripts may need to be adapted to local OpenVLA and RoboTwin installations.
- The code assumes access to OpenVLA weights and RoboTwin datasets.
- For reproducible comparison, keep task configuration, random seeds, maximum inference steps, action encoding, and dataset statistics consistent.

## Acknowledgements

This project builds on OpenVLA and RoboTwin. It also takes inspiration from recent work on vision-language-action models, bimanual imitation learning, action chunking, and cross-arm coordination.
