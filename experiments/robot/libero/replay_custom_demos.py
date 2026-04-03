"""
replay_custom_demos.py

【脚本功能】
用键盘采集的原始 demo（只有 states + actions，没有图像）无法直接用于训练 OpenVLA，
因为 OpenVLA 需要图像观测。本脚本的作用是：
  1. 读取原始 HDF5（demo_clean.hdf5）里的 states 和 actions
  2. 在 LIBERO 仿真环境中重新"回放"每条 demo 的 action 序列
  3. 每一步截取图像（第三人称 agentview + 手腕 eye_in_hand）
  4. 把图像、机器人状态、actions 一起存入新的 HDF5，供后续 TFDS builder 使用

【输入 HDF5 结构（原始采集）】
  data/
    demo_1/
      actions   (T, 7)   每步的机器人动作：[dx, dy, dz, droll, dpitch, dyaw, gripper]
      states    (T, D)   每步的完整仿真状态（MuJoCo flattened state）

【输出 HDF5 结构（带图像）】
  data/
    demo_0/
      obs/
        agentview_rgb    (T, 256, 256, 3)  第三人称相机图像
        eye_in_hand_rgb  (T, 256, 256, 3)  手腕相机图像
        ee_states        (T, 6)            末端执行器状态：[pos(3) + axis_angle(3)]
        ee_pos           (T, 3)            末端执行器位置
        ee_ori           (T, 3)            末端执行器朝向（axis-angle）
        gripper_states   (T, 2)            夹爪关节角度
        joint_states     (T, 7)            机械臂关节角度
      actions            (T, 7)            动作序列（与原始一致）
      states             (T, D)            仿真状态
      robot_states       (T, 9)            [gripper(2) + eef_pos(3) + eef_quat(4)]
      rewards            (T,)              只有最后一步为 1，其余为 0
      dones              (T,)              只有最后一步为 1，其余为 0
    [attrs] language_instruction, num_demos

【使用方法】
    cd /home/yj/desktop/openvla
    python experiments/robot/libero/replay_custom_demos.py \
        --input_hdf5 /home/yj/desktop/LIBERO/demonstration_data/.../demo_clean.hdf5 \
        --output_hdf5 /home/yj/desktop/LIBERO/demonstration_data/pick_orange_juice.hdf5 \
        --bddl_file /home/yj/desktop/LIBERO/libero/libero/bddl_files/libero_object/pick_up_the_orange_juice_and_place_it_in_the_basket.bddl \
        --task_language "pick up the orange juice and place it in the basket"
"""

import argparse
import os
import sys

# ── 渲染后端设置 ──────────────────────────────────────────────────────────────
# 必须在 import robosuite 之前设置，否则 robosuite 会默认尝试 EGL（GPU headless），
# 而本机 EGL 驱动不支持 PLATFORM_DEVICE 扩展，会报 ImportError。
# osmesa 是纯软件渲染，不依赖 GPU，兼容性最好。
os.environ["MUJOCO_GL"] = "osmesa"

import h5py
import numpy as np
import robosuite.utils.transform_utils as T  # 四元数 → axis-angle 等工具函数
import tqdm

sys.path.insert(0, "/home/yj/desktop/LIBERO")

# robosuite 的 macros.py 里 MUJOCO_GPU_RENDERING 默认为 True，
# 会强制覆盖 MUJOCO_GL 环境变量，改回 False 才能让 osmesa 生效。
import robosuite.macros as macros
macros.MUJOCO_GPU_RENDERING = False

from libero.libero.envs import OffScreenRenderEnv  # LIBERO 的离屏渲染环境

# 图像分辨率，OpenVLA 训练用 256x256
IMAGE_RESOLUTION = 256


def main(args):
    # ── 1. 读取原始 HDF5 ──────────────────────────────────────────────────────
    src = h5py.File(args.input_hdf5, "r")
    orig_data = src["data"]
    num_demos = len(orig_data)
    print(f"Found {num_demos} demos in {args.input_hdf5}")

    # ── 2. 创建 LIBERO 仿真环境 ───────────────────────────────────────────────
    # OffScreenRenderEnv：离屏渲染，不弹出窗口，可以截图
    # bddl_file_name：任务描述文件，定义场景物体、初始位置、成功条件
    # control_freq=20：控制频率 20Hz，与采集时一致
    env = OffScreenRenderEnv(
        bddl_file_name=args.bddl_file,
        camera_heights=IMAGE_RESOLUTION,
        camera_widths=IMAGE_RESOLUTION,
        control_freq=20,
    )
    env.seed(0)

    # ── 3. 创建输出 HDF5 ──────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(args.output_hdf5)), exist_ok=True)
    dst = h5py.File(args.output_hdf5, "w")
    grp = dst.create_group("data")

    num_success = 0  # 成功回放并保存的 demo 数量

    # --max_demos 参数：只处理前 N 条，用于测试
    if args.max_demos > 0:
        num_demos = min(num_demos, args.max_demos)

    # ── 4. 逐条回放 demo ─────────────────────────────────────────────────────
    for demo_idx in tqdm.tqdm(range(num_demos)):
        # 原始采集的 demo 从 demo_1 开始编号（1-indexed）
        demo_key = f"demo_{demo_idx + 1}"
        if demo_key not in orig_data:
            # 兼容 0-indexed 的情况
            demo_key = f"demo_{demo_idx}"
        if demo_key not in orig_data:
            print(f"Warning: {demo_key} not found, skipping")
            continue

        demo_data = orig_data[demo_key]
        orig_actions = demo_data["actions"][()]   # (T, 7) 原始动作序列
        orig_states  = demo_data["states"][()]    # (T, D) 原始仿真状态序列

        # ── 4a. 初始化环境状态 ────────────────────────────────────────────────
        # env.reset() 先重置到默认初始状态
        # env.set_init_state() 再把仿真状态设置为采集时的第一帧，
        #   这样物体位置、机器人姿态都和原始 demo 完全一致。
        #   注意：不能用 env.sim.set_state_from_flattened()，那个不更新 observation。
        env.reset()
        env.set_init_state(orig_states[0])

        # ── 4b. Warm-up：执行 10 步空动作 ────────────────────────────────────
        # 与 LIBERO 官方回放脚本保持一致。
        # gripper=-1 表示夹爪保持张开，xyz/rpy 全为 0 表示不移动。
        # warm-up 的作用是让物理仿真稳定下来（消除初始状态的数值抖动），
        # 同时让渲染器完成初始化，确保第一帧图像正常。
        dummy_action = [0, 0, 0, 0, 0, 0, -1]
        obs = None
        for _ in range(10):
            obs, _, _, _ = env.step(dummy_action)
        # warm-up 结束后 obs 已经是有效的观测，不为 None

        # ── 4c. 回放所有 action，逐帧收集数据 ────────────────────────────────
        states          = []   # 仿真状态（MuJoCo flattened）
        actions         = []   # 动作
        ee_states       = []   # 末端执行器状态（pos + axis-angle，6D）
        gripper_states  = []   # 夹爪关节角度（2D）
        joint_states    = []   # 机械臂关节角度（7D）
        robot_states    = []   # 综合机器人状态（gripper + eef_pos + eef_quat，9D）
        agentview_images    = []   # 第三人称相机图像
        eye_in_hand_images  = []   # 手腕相机图像

        done = False
        for action in orig_actions:
            # 记录当前步的仿真状态（在 step 之前）
            # 第一步用原始初始状态，后续从仿真器实时读取
            if len(states) == 0:
                states.append(orig_states[0])
            else:
                states.append(env.sim.get_state().flatten())

            actions.append(action)

            # 记录当前步的观测（obs 是上一步 env.step() 返回的结果）
            # warm-up 结束后 obs 不为 None，所以这里每步都会执行
            if obs is not None:
                # 夹爪关节角度，shape (2,)
                gripper_states.append(obs["robot0_gripper_qpos"])

                # 机械臂 7 个关节角度，shape (7,)
                joint_states.append(obs["robot0_joint_pos"])

                # 末端执行器状态：位置(3D) + 朝向转为 axis-angle(3D) = 6D
                # quat2axisangle：四元数 → 轴角表示，更适合作为网络输入
                ee_states.append(np.hstack((
                    obs["robot0_eef_pos"],
                    T.quat2axisangle(obs["robot0_eef_quat"]),
                )))

                # 综合机器人状态：夹爪(2) + 末端位置(3) + 末端四元数(4) = 9D
                robot_states.append(np.concatenate([
                    obs["robot0_gripper_qpos"],
                    obs["robot0_eef_pos"],
                    obs["robot0_eef_quat"],
                ]))

                # 图像旋转 180°：LIBERO 相机坐标系导致图像上下左右颠倒，
                # [::-1, ::-1] 同时翻转行和列，等效于旋转 180°
                agentview_images.append(obs["agentview_image"][::-1, ::-1])
                eye_in_hand_images.append(obs["robot0_eye_in_hand_image"][::-1, ::-1])

            # 执行动作，推进仿真一步
            # done=True 表示任务成功（物体放入篮子）
            obs, _, done, _ = env.step(action.tolist())

        # ── 4d. 检查是否成功 ──────────────────────────────────────────────────
        # done=False 说明回放完所有 action 后任务仍未完成，这条 demo 不保存。
        # 可能原因：warm-up 改变了初始状态导致轨迹偏移，或原始 demo 本身有问题。
        if not done or len(actions) == 0:
            print(f"  Demo {demo_key}: not successful or empty, skipping")
            continue

        # ── 4e. 对齐长度 ──────────────────────────────────────────────────────
        # obs 列表比 actions 少 1 帧：
        #   第 t 步：记录 obs（来自第 t-1 步的结果），然后执行 action[t]
        #   最后一步 action 执行后的 obs 没有被记录（循环结束了）
        # 所以 agentview_images 长度 = T-1，actions 长度 = T
        # 用图像数量 n 来截断 actions 和 states，保证所有数组等长。
        # （warm-up 后第一帧 obs 不为 None，所以 n = T-1，不是 T-2）
        n = len(agentview_images)
        actions = actions[:n]
        states  = states[:n]

        # rewards 和 dones：只有最后一步为 1（稀疏奖励）
        dones   = np.zeros(n, dtype=np.uint8); dones[-1]   = 1
        rewards = np.zeros(n, dtype=np.uint8); rewards[-1] = 1

        # ── 4f. 写入输出 HDF5 ─────────────────────────────────────────────────
        # 输出 demo 重新从 0 编号（demo_0, demo_1, ...），跳过失败的 demo
        ep_grp  = grp.create_group(f"demo_{num_success}")
        obs_grp = ep_grp.create_group("obs")
        obs_grp.create_dataset("agentview_rgb",   data=np.stack(agentview_images))   # (n, 256, 256, 3)
        obs_grp.create_dataset("eye_in_hand_rgb", data=np.stack(eye_in_hand_images)) # (n, 256, 256, 3)
        obs_grp.create_dataset("ee_states",       data=np.stack(ee_states))          # (n, 6)
        obs_grp.create_dataset("ee_pos",          data=np.stack(ee_states)[:, :3])   # (n, 3)
        obs_grp.create_dataset("ee_ori",          data=np.stack(ee_states)[:, 3:])   # (n, 3)
        obs_grp.create_dataset("gripper_states",  data=np.stack(gripper_states))     # (n, 2)
        obs_grp.create_dataset("joint_states",    data=np.stack(joint_states))       # (n, 7)
        ep_grp.create_dataset("actions",      data=np.array(actions))                # (n, 7)
        ep_grp.create_dataset("states",       data=np.stack(states))                 # (n, D)
        ep_grp.create_dataset("robot_states", data=np.stack(robot_states))           # (n, 9)
        ep_grp.create_dataset("rewards",      data=rewards)                          # (n,)
        ep_grp.create_dataset("dones",        data=dones)                            # (n,)

        num_success += 1
        print(f"  Demo {demo_key}: {n} steps (saved as demo_{num_success - 1})")

    # ── 5. 写入数据集级别的元数据 ─────────────────────────────────────────────
    grp.attrs["language_instruction"] = args.task_language  # 任务语言描述
    grp.attrs["num_demos"] = num_success                    # 实际保存的 demo 数量

    src.close()
    dst.close()
    env.close()

    print(f"\nDone! {num_success}/{num_demos} demos saved to {args.output_hdf5}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_hdf5",     type=str, required=True,  help="原始采集的 HDF5 路径")
    parser.add_argument("--output_hdf5",    type=str, required=True,  help="输出带图像的 HDF5 路径")
    parser.add_argument("--bddl_file",      type=str, required=True,  help="LIBERO 任务 BDDL 文件路径")
    parser.add_argument("--task_language",  type=str,
                        default="pick up the orange juice and place it in the basket",
                        help="任务语言描述，写入输出 HDF5 的 attrs")
    parser.add_argument("--max_demos",      type=int, default=0,
                        help="只处理前 N 条 demo（0 = 全部），用于测试")
    args = parser.parse_args()
    main(args)
