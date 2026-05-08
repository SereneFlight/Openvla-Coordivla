"""
Coordivia_configuration.py

Coordivla的配置类，继承自OpenVLAConfig,增加双臂协调相关参数
"""
from typing import Any, Optional

from .configuration_prismatic import OpenVLAConfig


class CoordiVLAConfig(OpenVLAConfig):
    """
    CoordiVLA 配置类,双臂机器人vla模型
    在 OpenVLAConfig基础上增:
    - coordination_layer : 在哪一层插入跨臂协调模块(默认中间层)
    - coordination_num_heads: 协调模块的注意力头数
    - use_residual_injection: 是否使用残差注入（α 初始化为 0)
    - coordination_summary_tokens: 用少量可学习 query 压缩另一侧上下文的 token 数，0 表示保留完整上下文
    - coordination_num_layers: 协调模块层数
    - left_action_dim(int, 默认7) 左臂动作维度
    - right_action_dim(int, 默认7) 右臂动作维度
    **kwargs: 传递给 OpenVLAConfig 的其他参数（如 norm_stats、n_action_bins).
    """
    model_type: str = "coordivla"

    def __init__(
        self,
        coordination_layer: Optional[int] = None,
        coordination_num_heads: int = 8,
        coordination_summary_tokens: int = 0,
        coordination_num_layers: int = 3,
        coordination_fusion: str = "residual",
        coordination_include_self_action_prefix: bool = True,
        coordination_prefix_memory: bool = False,
        use_residual_injection: bool = True,
        use_coordination: bool = True,
        left_action_dim: int = 7,
        right_action_dim: int = 7,
        **kwargs: Any,
    ) -> None:
# 保存双臂协调参数
        self.coordination_layer = coordination_layer
        self.coordination_num_heads = coordination_num_heads
        self.coordination_summary_tokens = coordination_summary_tokens
        self.coordination_num_layers = coordination_num_layers
        self.coordination_fusion = coordination_fusion
        self.coordination_include_self_action_prefix = coordination_include_self_action_prefix
        self.coordination_prefix_memory = coordination_prefix_memory
        self.use_residual_injection = use_residual_injection
        self.use_coordination = use_coordination
        self.left_action_dim = left_action_dim
        self.right_action_dim = right_action_dim
        
        super().__init__(**kwargs)
# 如果没有指定协调层，那么默认选择中间
        if self.coordination_layer is None:
            num_layers = self.text_config.num_hidden_layers
            self.coordination_layer = num_layers // 2

# 验证coordination_layer 在有效范围内
        if not(0 <= self.coordination_layer < self.text_config.num_hidden_layers):
            raise ValueError(
                f"coordination_layer 必须在 [0, {self.text_config.num_hidden_layers}) 范围内，"
                f"当前值为 {self.coordination_layer}"
            )
        if self.coordination_fusion not in {"residual", "meta_query"}:
            raise ValueError(
                f"coordination_fusion must be 'residual' or 'meta_query', got {self.coordination_fusion!r}"
            )

        if self.left_action_dim <= 0 or self.right_action_dim <= 0:
            raise ValueError(
                f"动作维度必须为正数，当前 left={self.left_action_dim},right={self.right_action_dim}"
            )
        if self.coordination_num_heads <= 0:
            raise ValueError(f"coordination_num_heads 必须为正数，当前值为 {self.coordination_num_heads}")
        if self.coordination_summary_tokens < 0:
            raise ValueError(
                f"coordination_summary_tokens 不能为负数，当前值为 {self.coordination_summary_tokens}"
            )
        if self.coordination_num_layers <= 0:
            raise ValueError(
                f"coordination_num_layers 必须为正整数，当前值为 {self.coordination_num_layers}"
            )
