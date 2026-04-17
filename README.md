# OpenVLA-CoordiVLA

基于 [OpenVLA](https://github.com/openvla/openvla) 的双臂协调 VLA 模型（CoordiVLA），用于 RoboTwin 双臂机器人任务。

---

## 新增文件

| 文件路径 | 说明 |
|---|---|
| `prismatic/extern/hf/coordivla_configuration.py` | CoordiVLA 配置类，继承 OpenVLAConfig，增加双臂协调参数 |
| `prismatic/extern/hf/coordivla_modeling.py` | CoordiVLA 模型定义，包含 CrossArmCoordinationModule 和 CoordiVLAForActionPrediction |
| `vla-scripts/coordivia_finetune.py` | CoordiVLA LoRA 微调脚本，读取 RoboTwin hdf5 数据，双臂联合训练 |
| `vla-scripts/merge_coordivla_lora.py` | 训练完成后将 LoRA adapter 融合进 base 模型，保存完整权重 |
| `vla-scripts/compute_dataset_statistics.py` | 计算数据集动作归一化统计量，生成 dataset_statistics.json |
| `dataset_statistics_handover_block.json` | handover_block 任务的动作归一化统计量 |

## 修改文件

| 文件路径 | 修改内容 |
|---|---|
| `prismatic/extern/hf/coordivla_configuration.py` | 修复缩进问题 |
| `prismatic/extern/hf/coordivla_modeling.py` | 修复文件头注释；修复 num_patches 计算兼容 fused backbone |
| `vla-scripts/coordivia_finetune.py` | 重写为 CoordiVLA 双臂训练流程，替换原 OpenVLA 单臂逻辑 |

---

## 使用方法

### 1. 微调

```bash
cd /path/to/openvla
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/coordivia_finetune.py \
    --openvla_path /path/to/openvla-7b \
    --data_dir /path/to/RoboTwin/data \
    --task_name handover_block
```

### 2. 计算数据集统计量

```bash
python vla-scripts/compute_dataset_statistics.py \
    --data_dir /path/to/RoboTwin/data/handover_block/demo_clean/data \
    --task_name handover_block \
    --save_path dataset_statistics_handover_block.json
```

### 3. 合并 LoRA 权重

```bash
python vla-scripts/merge_coordivla_lora.py \
    --openvla_path /path/to/openvla-7b \
    --adapter_dir /path/to/adapter-tmp/<run_name> \
    --save_dir /path/to/merged/<run_name> \
    --stats_path dataset_statistics_handover_block.json
```

### 4. RoboTwin 评估

```bash
cd /path/to/RoboTwin
bash policy/CoordiVLA/eval.sh handover_block demo_clean /path/to/merged/<run_name> 0 0
```
