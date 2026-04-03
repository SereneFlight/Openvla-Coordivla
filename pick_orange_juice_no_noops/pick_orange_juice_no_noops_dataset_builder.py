"""
pick_orange_juice_no_noops_dataset_builder.py

【脚本功能】
把 replay_custom_demos.py 生成的带图像 HDF5，转换成 OpenVLA 训练所需的 RLDS 格式。

【什么是 RLDS？】
RLDS（Reinforcement Learning Datasets）是 Google 提出的机器人数据集标准格式，
基于 TensorFlow Datasets（TFDS）。OpenVLA 的训练代码只认这个格式。
RLDS 把每条 demo 存成一个 "episode"，每个 episode 包含若干 "step"，
每个 step 包含：观测（图像+状态）、动作、奖励、语言指令等。

【什么是 TFDS Builder？】
TFDS Builder 是一个 Python 类，告诉 TFDS 框架：
  - 数据长什么样（_info：定义字段名、类型、shape）
  - 数据从哪来（_split_generators：指定数据文件路径）
  - 怎么读数据（_generate_examples：逐条读取并 yield）
写好之后运行 `tfds build`，TFDS 会自动调用这个类，把数据转成 TFRecord 文件。

【什么是 language_embedding？】
OpenVLA 不直接处理文字，而是用 Universal Sentence Encoder（USE）把语言指令
预先编码成 512 维的向量，训练时直接用这个向量。
USE 是 Google 的预训练语言模型，相同的句子每次编码结果一样。

【使用方法】
  1. 先跑 replay_custom_demos.py 生成带图像的 HDF5
  2. 把 REPLAYED_HDF5_PATH 改成实际路径
  3. 在 autodl 上运行：
       cd /root/openvla/pick_orange_juice_no_noops
       tfds build --overwrite
  4. 生成的数据集在 ~/tensorflow_datasets/pick_orange_juice_no_noops/

【微调时指定数据集】
    --data_root_dir ~/tensorflow_datasets
    --dataset_name pick_orange_juice_no_noops
"""

from typing import Iterator, Tuple, Any

import h5py
import numpy as np
import tensorflow_datasets as tfds

# ── 全局配置 ──────────────────────────────────────────────────────────────────
# replay_custom_demos.py 生成的带图像 HDF5 路径（在 autodl 上运行时改这里）
REPLAYED_HDF5_PATH = "/home/yj/desktop/LIBERO/demonstration_data/pick_orange_juice_replayed.hdf5"

# 任务语言描述，每个 step 都会附上这句话
LANGUAGE_INSTRUCTION = "pick up the orange juice and place it in the basket"

# 图像分辨率，必须和 replay 时一致
IMAGE_SIZE = 256


class PickOrangeJuiceNoNoops(tfds.core.GeneratorBasedBuilder):
    """
    TFDS 数据集构建器：抓橙汁放篮子任务。

    类名必须和文件名（去掉 _dataset_builder.py 后缀）的驼峰形式一致，
    TFDS 框架通过这个规则自动找到对应的 Builder 类。
    文件名：pick_orange_juice_no_noops_dataset_builder.py
    类名：PickOrangeJuiceNoNoops  ✓
    """

    # 数据集版本号，格式为 "主版本.次版本.补丁版本"
    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {"1.0.0": "Initial release."}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _info(self) -> tfds.core.DatasetInfo:
        """
        定义数据集的结构（schema）：每个字段叫什么名字、是什么类型、shape 是多少。
        这相当于数据库的表结构定义，TFDS 会根据这个来验证和存储数据。

        OpenVLA 训练代码会按照这里定义的字段名来读取数据，
        所以字段名必须和 OpenVLA 期望的完全一致。
        """
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({

                # ── episode 级别：一条完整的 demo ────────────────────────────
                "steps": tfds.features.Dataset({
                    # Dataset 表示这是一个序列，里面每个元素是一个 step

                    # ── 观测（机器人在这一步"看到"和"感知到"的信息）────────
                    "observation": tfds.features.FeaturesDict({

                        # 第三人称相机图像（agentview）
                        # shape=(256, 256, 3)：高256 × 宽256 × RGB三通道
                        # encoding_format="jpeg"：存储时压缩为 JPEG，节省空间
                        "image": tfds.features.Image(
                            shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
                            dtype=np.uint8,
                            encoding_format="jpeg",
                        ),

                        # 手腕相机图像（eye_in_hand）
                        # 安装在机械臂末端，看到的是夹爪视角
                        "wrist_image": tfds.features.Image(
                            shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
                            dtype=np.uint8,
                            encoding_format="jpeg",
                        ),

                        # 机器人本体状态，8 维：
                        #   前 6 维：末端执行器状态（ee_states）
                        #     - 前 3 维：末端位置 xyz（单位：米）
                        #     - 后 3 维：末端朝向（axis-angle 表示）
                        #   后 2 维：夹爪关节角度（gripper_states）
                        "state": tfds.features.Tensor(shape=(8,), dtype=np.float32),
                    }),

                    # 动作，7 维：[dx, dy, dz, droll, dpitch, dyaw, gripper]
                    #   前 6 维：末端执行器的位移和旋转增量（delta EEF）
                    #   第 7 维：夹爪控制（-1=张开，+1=闭合）
                    "action": tfds.features.Tensor(shape=(7,), dtype=np.float32),

                    # 折扣因子，强化学习用，这里固定为 1.0（不折扣）
                    "discount": tfds.features.Scalar(dtype=np.float32),

                    # 奖励信号：只有最后一步（任务完成）为 1.0，其余为 0.0
                    # OpenVLA 用稀疏奖励，不依赖中间奖励
                    "reward": tfds.features.Scalar(dtype=np.float32),

                    # 标志位：标记这一步在 episode 中的位置
                    "is_first":    tfds.features.Scalar(dtype=np.bool_),  # 是否是第一步
                    "is_last":     tfds.features.Scalar(dtype=np.bool_),  # 是否是最后一步
                    "is_terminal": tfds.features.Scalar(dtype=np.bool_),  # 是否是终止步（同 is_last）

                    # 语言指令文本，每一步都重复存储同一句话
                    "language_instruction": tfds.features.Text(),
                }),

                # ── episode 级别的元数据 ──────────────────────────────────────
                "episode_metadata": tfds.features.FeaturesDict({
                    # 数据来源文件路径，方便追溯
                    "file_path": tfds.features.Text(),
                }),
            })
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """
        定义数据集的划分（split）：训练集、验证集等。
        这里只有训练集，直接指向 HDF5 文件。

        dl_manager 是 TFDS 提供的下载管理器，这里用不到（数据是本地文件）。
        """
        return {
            "train": self._generate_examples(REPLAYED_HDF5_PATH),
        }

    def _generate_examples(self, hdf5_path: str) -> Iterator[Tuple[str, Any]]:
        """
        核心函数：逐条读取 HDF5 里的 demo，转换成 TFDS 期望的格式，然后 yield 出去。

        这是一个 Python generator 函数（用 yield 而不是 return），
        TFDS 框架会不断调用它来获取下一条数据。

        【重要：为什么要先全部读入内存再 yield？】
        Python generator 是"懒执行"的，yield 之后函数会暂停，
        等 TFDS 框架来取数据时再继续。
        如果在 yield 之前没有关闭 h5py 文件，Python 的垃圾回收（GC）
        可能在 generator 暂停期间关闭文件句柄，导致后续读取报错。
        解决方法：先把所有数据读进内存（Python list），关闭文件，再 yield。
        """

        def _load_all_episodes(path):
            """把 HDF5 里所有 demo 读入内存，返回 list。"""
            episodes = []
            f = h5py.File(path, "r")
            try:
                data = f["data"]

                # 按 demo 编号排序：demo_0, demo_1, demo_2, ...
                # sorted(..., key=lambda x: int(x.split("_")[1]))
                #   x.split("_") 把 "demo_0" 拆成 ["demo", "0"]
                #   [1] 取 "0"，int(...) 转成整数 0，用于数字排序
                #   （不排序的话 "demo_10" 会排在 "demo_2" 前面）
                for key in sorted(data.keys(), key=lambda x: int(x.split("_")[1])):
                    ep  = data[key]
                    obs = ep["obs"]

                    # [()] 是 h5py 的语法，表示把数据集全部读入内存（numpy 数组）
                    # 不加 [()] 的话得到的是 h5py Dataset 对象，文件关闭后就失效了
                    actions          = ep["actions"][()]           # (T, 7)
                    agentview_imgs   = obs["agentview_rgb"][()]    # (T, 256, 256, 3)
                    eye_in_hand_imgs = obs["eye_in_hand_rgb"][()]  # (T, 256, 256, 3)
                    ee_states        = obs["ee_states"][()]        # (T, 6)  末端执行器状态
                    gripper_states   = obs["gripper_states"][()]   # (T, 2)  夹爪状态

                    T_len = len(actions)  # 这条 demo 的总步数

                    # 把 ee_states(T,6) 和 gripper_states(T,2) 拼成 (T,8)
                    # np.concatenate(..., axis=-1) 沿最后一个维度拼接
                    # 例如：[0.1, 0.2, 0.3, 0.0, 0.0, 0.0] + [0.04, 0.04]
                    #      → [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.04, 0.04]
                    states = np.concatenate(
                        [ee_states, gripper_states], axis=-1
                    ).astype(np.float32)  # (T, 8)

                    # 逐步构建这条 demo 的所有 step
                    steps = []
                    for t in range(T_len):
                        steps.append({
                            "observation": {
                                "image":       agentview_imgs[t],    # (256, 256, 3) uint8
                                "wrist_image": eye_in_hand_imgs[t],  # (256, 256, 3) uint8
                                "state":       states[t],            # (8,) float32
                            },
                            "action":   actions[t].astype(np.float32),  # (7,) float32

                            # 折扣因子固定 1.0（不衰减未来奖励）
                            "discount": 1.0,

                            # 稀疏奖励：只有最后一步为 1.0
                            # (t == T_len - 1) 是布尔表达式，True→1.0，False→0.0
                            "reward":   float(t == T_len - 1),

                            # 标志位
                            "is_first":    t == 0,           # 第一步为 True
                            "is_last":     t == T_len - 1,   # 最后一步为 True
                            "is_terminal": t == T_len - 1,   # 同上（任务完成即终止）

                            # 每一步都附上语言指令
                            "language_instruction": LANGUAGE_INSTRUCTION,
                        })

                    # 把这条 demo 加入列表
                    # key 是 demo 的唯一标识符（如 "demo_0"），TFDS 用它去重
                    episodes.append((key, {
                        "steps": steps,
                        "episode_metadata": {"file_path": path},
                    }))

            finally:
                # 无论是否出错，都要关闭文件
                # 必须在 yield 之前关闭，否则 GC 可能提前回收文件句柄
                f.close()

            return episodes

        # 所有数据读入内存、文件已关闭，现在可以安全地逐条 yield
        for key, sample in _load_all_episodes(hdf5_path):
            yield key, sample
