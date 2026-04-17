"""
compute_dataset_statistics.py

遍历 RoboTwin hdf5 数据，计算动作的真实 q01/q99 统计量，
生成 dataset_statistics.json，供 merge 脚本注入 norm_stats。

用法：
    python vla-scripts/compute_dataset_statistics.py \
        --data_dir /path/to/RoboTwin/data/handover_block/demo_clean/data \
        --task_name handover_block \
        --save_path /path/to/dataset_statistics.json
"""

import argparse
import glob
import json
import numpy as np
import h5py


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",  type=str, required=True, help="hdf5 文件所在目录")
    parser.add_argument("--task_name", type=str, required=True, help="任务名，作为 norm_stats 的 key")
    parser.add_argument("--save_path", type=str, required=True, help="输出 dataset_statistics.json 路径")
    args = parser.parse_args()

    hdf5_files = sorted(glob.glob(f"{args.data_dir}/*.hdf5"))
    assert len(hdf5_files) > 0, f"未找到 hdf5 文件：{args.data_dir}"
    print(f"共找到 {len(hdf5_files)} 个 episode")

    left_actions, right_actions = [], []

    for path in hdf5_files:
        with h5py.File(path, "r") as f:
            left_arm     = f["joint_action/left_arm"][:]      # (T, 6)
            left_gripper = f["joint_action/left_gripper"][:]  # (T,)
            right_arm    = f["joint_action/right_arm"][:]     # (T, 6)
            right_gripper = f["joint_action/right_gripper"][:] # (T,)

            left  = np.concatenate([left_arm,  left_gripper[:, None]],  axis=1)  # (T, 7)
            right = np.concatenate([right_arm, right_gripper[:, None]], axis=1)  # (T, 7)

            left_actions.append(left)
            right_actions.append(right)

    left_all  = np.concatenate(left_actions,  axis=0)  # (N, 7)
    right_all = np.concatenate(right_actions, axis=0)  # (N, 7)
    # 左右臂拼在一起统计（训练时两臂共用同一套 norm_stats）
    all_actions = np.concatenate([left_all, right_all], axis=0)  # (2N, 7)

    q01 = np.percentile(all_actions, 1,  axis=0).tolist()
    q99 = np.percentile(all_actions, 99, axis=0).tolist()
    mask = [True] * 7

    print(f"q01: {[f'{v:.4f}' for v in q01]}")
    print(f"q99: {[f'{v:.4f}' for v in q99]}")

    dataset_statistics = {
        args.task_name: {
            "action": {
                "q01":  q01,
                "q99":  q99,
                "mask": mask,
            }
        }
    }

    with open(args.save_path, "w") as f:
        json.dump(dataset_statistics, f, indent=2)
    print(f"已保存到 {args.save_path}")


if __name__ == "__main__":
    main()
