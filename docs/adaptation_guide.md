# verl 多芯适配指南

本文档面向硬件厂商，指导如何基于 [verl-hardware-plugin](https://github.com/verl-project/verl-hardware-plugin) 仓库完成自家芯片对 verl 的适配工作。

## 1. 整体架构

verl 通过 **Platform + Engine** 插件机制实现多芯支持：

```
verl (主框架, PR#6086)
    │
    └── entry_points: verl.plugins → verl_hardware_plugin
            │
            ├── platforms/  → 硬件平台抽象（设备管理、通信、内存、Ray 等）
            └── engines/    → 训练引擎扩展（FSDP/Megatron 的硬件特化实现）
```

- **Platform**：统一硬件接口层，封装 torch 设备操作、vLLM 推理适配、Ray 资源管理、分布式通信等
- **Engine**：训练引擎的硬件特化扩展，通过继承 + `initialize()` 注入厂商逻辑

### 1.1 设计原则

1. **条件导入**：各 platform/engine 模块在 `try/except` 中导入，缺少 SDK 不影响其他平台
2. **后注册覆盖**：同名 platform 后注册的覆盖先注册的，plugin 可覆盖内置实现
3. **自动检测**：第一个 `is_platform_available(use_smi_check=True)` 返回 True 的 platform 被选中，也可通过 `VERL_PLATFORM` 强制指定
4. **最小侵入**：Engine 通过继承 + `initialize()` 注入逻辑，不修改基类行为
5. **二维引擎键**：`(device, vendor)` 允许共享同一 device type 的多个厂商拥有独立 engine

### 1.2 核心仓库与 PR

| 仓库 | 说明 |
|------|------|
| [verl PR#6086](https://github.com/verl-project/verl/pull/6086) | verl 主框架的 Platform & Engine Registry 机制 |
| [verl-hardware-plugin](https://github.com/verl-project/verl-hardware-plugin) | 多芯适配参考实现仓库（本仓库） |

---

## 2. Platform 接口说明

Platform 类需继承 `PlatformBase` 并实现以下接口（参考 `verl/plugin/platform/platform_base.py`）：

| 分类 | 方法 | 是否必须 | 说明 |
|------|------|----------|------|
| **核心设备管理** | `device_name` | 是 | 设备类型字符串，如 `"cuda"`, `"xpu"`, `"mlu"` |
| | `vendor_name` | 是 | 厂商标识，如 `"nvidia"`, `"metax"`, `"intel"` |
| | `device_module` | 是 | `torch.<device>` 命名空间模块 |
| | `is_available()` | 是 | 检测设备是否可用（直接调用 `torch.<device>.is_available()`） |
| | `is_platform_available(use_smi_check)` | 否（有默认实现） | 平台自动检测，默认委托给 `is_available()`；可重写以支持 SMI 命令等检测方式 |
| | `current_device()` | 是 | 当前设备索引 |
| | `device_count()` | 是 | 设备数量 |
| | `set_device(device_index)` | 是 | 设置当前设备 |
| | `synchronize(device_index)` | 是 | 设备同步 |
| **随机数** | `manual_seed(seed)` | 是 | 设置当前设备随机种子 |
| | `manual_seed_all(seed)` | 是 | 设置所有设备随机种子 |
| **内存管理** | `set_allocator_settings(settings)` | 是 | 内存分配器配置 |
| | `empty_cache()` | 是 | 清空缓存 |
| **设备属性** | `get_device_capability(device_index)` | 是 | 设备计算能力 |
| **分布式通信** | `communication_backend_name()` | 是 | 通信后端名称，如 `"nccl"`, `"xccl"`, `"cncl"` |
| | `visible_devices_envvar()` | 是 | 设备可见性环境变量名 |
| **Ray 集成** | `ray_resource_name()` | 是 | Ray 资源名称 |
| | `ray_resource_options(num_gpus)` | 是 | Ray 资源配置字典 |
| | `ray_noset_envvars()` | 是 | Ray 不应自动设置的环境变量 |
| **IPC** | `is_ipc_supported()` | 是 | 是否支持 IPC tensor 共享 |
| **Profiling** | `nvtx_range(msg)` | 是 | 性能标记（不支持则 no-op） |
| | `profiler_start()` | 是 | 启动 profiler（不支持则 no-op） |
| | `profiler_stop()` | 是 | 停止 profiler（不支持则 no-op） |
| **模型 Patch** | `apply_model_patches(model_type)` | 是 | 应用厂商特定的模型 patch |
| **Rollout** | `rollout_env_vars()` | 否 | rollout 引擎启动时注入的环境变量 |
| **集合通信** | `get_collective_module()` | 否 | 集合通信模块（如 `cupy.cuda.nccl`） |
| **底层运行时** | `cudart()` | 是 | CUDA runtime API 对象，不适用则返回 None |

> **注意**：尽管目前已覆盖 verl 中用到的大部分接口，但适配过程中可能发现遗漏，请及时与我们沟通协调。

---

## 3. Engine 接口说明

Engine 通过装饰器注册，继承基类并在 `initialize()` 中注入硬件特化逻辑：

```python
@EngineRegistry.register(
    model_type="language_model",   # "language_model" 或 "value_model"
    backend=["fsdp", "fsdp2"],     # 训练后端
    device="cuda",                 # 设备类型
    vendor="my_vendor"             # 厂商标识
)
class FSDPMyVendorEngineWithLMHead(FSDPEngineWithLMHead):
    def initialize(self):
        # 在此注入厂商特化逻辑（如启用特定算子库、设置通信参数等）
        super().initialize()
```

Engine 查找优先级：
1. 精确匹配 `(device, vendor)` — 厂商特定 engine
2. 回退到 device-only — 该设备类型的基础 engine
3. 对于 CUDA 兼容设备，回退到基础 CUDA engine

环境变量覆盖：
- `VERL_ENGINE_DEVICE` — 覆盖检测到的设备名
- `VERL_ENGINE_VENDOR` — 覆盖检测到的厂商名

---

## 4. 分阶段适配路线

### 阶段一：原生软件栈适配

目标：使用厂商原生 PyTorch 后端 + 原生通信库，跑通 verl GRPO 训练。

#### Step 1：添加 Platform 和 Engine

1. Fork 或 clone [verl-hardware-plugin](https://github.com/verl-project/verl-hardware-plugin)

2. 创建 Platform 文件 `verl_hardware_plugin/platforms/platform_<vendor>.py`：

```python
import torch
from verl.plugin.platform.platform_base import PlatformBase
from verl.plugin.platform.platform_manager import PlatformRegistry

@PlatformRegistry.register(platform="<vendor_name>")
class PlatformMyDevice(PlatformBase):

    @property
    def device_name(self) -> str:
        return "<device_type>"  # "cuda", "xpu", "mlu" 等

    @property
    def vendor_name(self) -> str:
        return "<vendor_name>"

    @property
    def device_module(self):
        return torch.<device_type>

    def is_available(self) -> bool:
        return torch.<device_type>.is_available()

    def is_platform_available(self, use_smi_check: bool = False) -> bool:
        if use_smi_check:
            return self.check_smi_command("<vendor-smi>")
        return torch.<device_type>.is_available()

    # ... 实现其余所有抽象方法
```

3. 创建 Engine 文件 `verl_hardware_plugin/engines/fsdp_<vendor>.py`：

```python
from verl.workers.engine.base import EngineRegistry
from verl.workers.engine.fsdp import FSDPEngineWithLMHead
from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithValueHead

@EngineRegistry.register(
    model_type="language_model",
    backend=["fsdp", "fsdp2"],
    device="<device_type>",
    vendor="<vendor_name>"
)
class FSDPMyVendorEngineWithLMHead(FSDPEngineWithLMHead):
    def initialize(self):
        # 厂商特化初始化逻辑
        super().initialize()

@EngineRegistry.register(
    model_type="value_model",
    backend=["fsdp", "fsdp2"],
    device="<device_type>",
    vendor="<vendor_name>"
)
class FSDPMyVendorEngineWithValueHead(FSDPEngineWithValueHead):
    def initialize(self):
        super().initialize()
```

4. 在 `verl_hardware_plugin/platforms/__init__.py` 的 `register_all_platforms()` 中添加导入：

```python
try:
    from verl_hardware_plugin.platforms import platform_<vendor>  # noqa: F401
    logger.info("Registered platform: <vendor>")
except Exception as e:
    logger.debug("<Vendor> platform not registered: %s", e)
```

5. 在 `verl_hardware_plugin/engines/__init__.py` 的 `register_all_engines()` 中添加导入：

```python
try:
    from verl_hardware_plugin.engines import fsdp_<vendor>  # noqa: F401
    logger.info("Registered engines: fsdp_<vendor>")
except Exception as e:
    logger.debug("<Vendor> FSDP engines not registered: %s", e)
```

6. 安装 plugin 仓库：

```bash
pip install --no-build-isolation -e .
```

#### Step 2：安装 verl 主框架

拉取 verl PR#6086 对应分支并安装：

```bash
git clone https://github.com/verl-project/verl.git
cd verl
git fetch origin pull/6086/head:pr-6086
git checkout pr-6086
pip install --no-build-isolation -e .
```

#### Step 3：运行 GRPO 测试

修改示例脚本 `examples/grpo_trainer/run_qwen3_0.6b_fsdp.sh`：

```bash
#!/bin/bash
set -x

# ============ 设备配置 ============
export CUDA_VISIBLE_DEVICES=0,1,2,3  # 或对应厂商的设备可见性变量
export HYDRA_FULL_ERROR=1

# ============ 平台选择 ============
export VERL_PLATFORM=<vendor_name>   # 强制指定平台（或依赖自动检测）

# ============ 数据与模型路径 ============
DATA_DIR=/path/to/gsm8k/main
MODEL_DIR=/path/to/Qwen3-0.6B

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_batch_size=64 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$MODEL_DIR \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name='verl_grpo_<vendor>' \
    trainer.experiment_name='qwen3_0.6b_<vendor>' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    +actor_rollout_ref.rollout.enable_sleep_mode=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    trainer.total_epochs=15 \
    $@
```

验证标准：训练正常启动，reward 曲线随 epoch 上升，无 OOM 或通信错误。

---

### 阶段二：FlagOS 软件栈适配

目标：在阶段一基础上，接入 FlagOS 统一软件栈（FlagGems 算子库、FlagCX 通信库、Transformer-Engine-FL 等），获得更优性能。

#### Step 1：安装 FlagOS 软件栈

在阶段一适配完 verl-hardware-plugin 后，额外安装以下组件：

| 组件 | 说明 |
|------|------|
| `flag_gems` | FlagOS 统一算子库 |
| `flag_tree` | FlagOS 算子调度树 |
| `flagcx` | FlagOS 统一通信库 |
| `vllm-plugin-fl` | vLLM 的 FlagOS 插件 |
| `megatron-lm-fl` | Megatron-LM 的 FlagOS 适配版本 |
| `transformer-engine-fl` | Transformer Engine 的 FlagOS 适配版本 |

安装顺序建议：底层依赖 → 上层组件

```bash
# 1. 底层依赖
pip install flag_gems flag_tree flagcx

# 2. 上层组件（按各组件文档安装）
pip install vllm-plugin-fl
pip install megatron-lm-fl
pip install transformer-engine-fl
```

> 具体安装方式请参考各组件的官方文档，版本需与厂商 PyTorch 版本匹配。

#### Step 2：修改脚本使能 FlagOS 软件栈

在阶段一的测试脚本基础上，添加 FlagOS 相关环境变量：

```bash
#!/bin/bash
set -x

# ============ 设备配置 ============
export CUDA_VISIBLE_DEVICES=0,1,2,3
export HYDRA_FULL_ERROR=1

# ============ 平台选择 ============
export VERL_PLATFORM=flagos  # 使用 FlagOS 平台

# ============ FlagGems 算子库 ============
export USE_FLAGGEMS=true
# 可选：算子白名单/黑名单
# export TRAINING_FL_FLAGOS_WHITELIST=rmsnorm,layernorm,softmax
# export TRAINING_FL_FLAGOS_BLACKLIST=flash_attention

# ============ FlagCX 通信库 ============
export USE_FLAGCX=1
# export FLAGCX_PATH=/path/to/FlagCX
# export PYTHONPATH=/path/to/FlagCX/plugin/torch:${PYTHONPATH}

# ============ Transformer-Engine-FL ============
export TE_FL_PREFER=flagos        # flagos / vendor / reference
export TE_FL_PREFER_VENDOR=0
export TE_FL_STRICT=0             # 非严格模式，允许 fallback

# ============ vLLM FlagOS 插件 ============
export VLLM_FL_OOT_ENABLED=1
export VLLM_FL_FLAGOS_BLACKLIST="where_scalar_other,where_scalar_self,where_self,where_self_out,pad"

# ============ 日志 ============
export VERL_LOGGING_LEVEL=INFO
export TEFL_LOG_LEVEL=INFO
export FLAGCX_LOG_LEVEL=INFO

# ============ 数据与模型路径 ============
DATA_DIR=/path/to/gsm8k/main
MODEL_DIR=/path/to/Qwen3-0.6B

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_batch_size=64 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$MODEL_DIR \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name='verl_grpo_flagos_<vendor>' \
    trainer.experiment_name='qwen3_0.6b_flagos_<vendor>' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    +actor_rollout_ref.rollout.enable_sleep_mode=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    trainer.total_epochs=15 \
    $@
```

---

## 5. 适配要点与常见问题

### 5.1 CUDA 兼容设备的特殊处理

如果厂商芯片兼容 CUDA（如 MetaX），`torch.cuda.is_available()` 会返回 True。此时需要：
- `device_name` 返回 `"cuda"`
- 重写 `is_platform_available(use_smi_check=True)` 通过厂商 SMI 命令区分硬件
- 在 Engine 注册时使用 `device="cuda", vendor="<vendor_name>"` 区分

### 5.2 通信后端适配

- `communication_backend_name()` 返回厂商通信后端名称（如 `"nccl"`, `"xccl"`, `"cncl"`, `"hccl"`）
- 如果通信后端不支持 `ReduceOp.AVG`，需在 Engine 的 `initialize()` 中启用 sum-based reduction
- FlagCX 场景下通信后端由 FlagCX 统一管理

### 5.3 vLLM Rollout 适配

- `rollout_env_vars()` 返回启动 vLLM rollout 进程时需要注入的环境变量
- 如果厂商有 vLLM 的定制版本，可通过此接口传递配置

### 5.4 Ray 资源管理

- 大多数情况下 `ray_resource_name()` 返回 `"GPU"` 即可
- `ray_noset_envvars()` 返回 Ray 不应自动设置的环境变量列表（防止 Ray 覆盖厂商设备可见性设置）

### 5.5 IPC 支持

- `is_ipc_supported()` 决定是否使用 IPC 方式在进程间共享 tensor
- 不支持 IPC 的平台返回 False，verl 会回退到其他数据传输方式

### 5.6 模型 Patch

- `apply_model_patches(model_type)` 用于在模型加载后应用厂商特定的 monkey-patch
- 典型场景：替换不兼容的算子实现、修改 attention 实现等

---

## 6. 目录结构参考

适配完成后，仓库结构应类似：

```
verl-hardware-plugin/
├── verl_hardware_plugin/
│   ├── __init__.py
│   ├── platforms/
│   │   ├── __init__.py
│   │   ├── platform_xpu.py          # Intel XPU 参考
│   │   ├── platform_mlu.py          # Cambricon MLU 参考
│   │   ├── platform_cuda_metax.py   # MetaX 参考（CUDA 兼容）
│   │   └── platform_<vendor>.py     # 厂商新增
│   ├── engines/
│   │   ├── __init__.py
│   │   ├── fsdp_xpu.py              # Intel FSDP 参考
│   │   ├── fsdp_mlu.py              # Cambricon FSDP 参考
│   │   ├── fsdp_metax.py            # MetaX FSDP 参考
│   │   ├── fsdp_<vendor>.py         # 厂商新增
│   │   └── megatron_<vendor>.py     # 厂商新增（可选）
│   └── utils/
│       ├── __init__.py
│       └── config_manager.py
├── tests/
│   └── test_plugin_registration.py
├── docs/
│   └── development.md
└── pyproject.toml
```

---

## 7. 验证清单

| 检查项 | 说明 |
|--------|------|
| `pip install -e .` 成功 | plugin 包可正常安装 |
| `pytest tests/ -v` 通过 | 注册机制正常工作 |
| `VERL_PLATFORM=<vendor> python -c "from verl.plugin.platform import get_platform; print(get_platform().vendor_name)"` | 平台正确加载 |
| GRPO 训练启动无报错 | 设备初始化、通信初始化正常 |
| 训练 reward 曲线上升 | 训练逻辑正确 |
| 多节点训练正常（如适用） | 跨节点通信正常 |

---

## 8. 沟通与支持

适配过程中如遇到以下情况，请及时联系我们：

- `PlatformBase` 接口缺少厂商需要的方法
- Engine 基类行为与厂商硬件不兼容
- vLLM rollout 在厂商硬件上需要额外适配
- Ray 资源管理与厂商设备模型不匹配

联系方式：通过 [verl-hardware-plugin](https://github.com/verl-project/verl-hardware-plugin) 仓库 Issue 或直接沟通。
