# Part 2:FlagOS 社区 VeRL-plugin-FL 介绍

## 2.1 背景与定位

VeRL-plugin-FL 是由**字节 verl 团队与 FlagOS 社区联合开发**的多芯片硬件插件。FlagOS 是面向异构 AI 芯片的全栈开源 AI 系统软件栈,秉持"一次开发,处处运行"的理念——AI 模型开发一次,即可以最小的迁移成本无缝移植到多种 AI 硬件上。

在强化学习后训练场景中,verl 需要同时驱动训练(FSDP / Megatron)与推理(rollout)两类负载,对硬件适配的要求远高于纯训练框架。VeRL-plugin-FL 通过 verl 的插件机制,将 FlagOS 的跨芯片能力接入 verl,让同一份 RL 训练代码能够运行在 NVIDIA、沐曦、天数、寒武纪、Intel 等多种硬件上,实现**跨硬件平台的统一训练与推理**。

VeRL-plugin-FL 已在多种国产及异构芯片上完成验证——**沐曦(MetaX)、天数智芯(Iluvatar)、寒武纪(Cambricon MLU)** 均已跑通 verl 的端到端 RL 后训练流程,证明"一次开发、多芯片运行"在 RL 后训练场景下的可行性。

插件通过 Python `entry_points`(`verl.plugins` 组)被 verl 自动发现,用户只需 `pip install` 即可启用,无需改动 verl 主框架的任何配置。

## 2.2 技术架构

VeRL-plugin-FL 采用基于装饰器的双注册表设计,与 verl 主框架解耦:

```
verl (主框架)
    │
    └── entry_points: verl.plugins → verl_hardware_plugin
            │
            ├── platforms/  硬件无关平台抽象层
            │     @PlatformRegistry.register(platform="vendor")
            │     ├── PlatformFlagOS   (device=cuda, vendor=flagos)
            │     ├── PlatformMetaX    (device=cuda, vendor=metax)
            │     ├── PlatformMLU      (device=mlu,  vendor=cambricon)
            │     └── PlatformXPU      (device=xpu,  vendor=intel)
            │
            ├── utils/  面向阶段的环境管理器 (FLEnvManager)
            │     ├── training 阶段:FlagGems 算子加速 / 白名单 / 黑名单
            │     └── rollout  阶段:vLLM + FlagGems / FlagCX 通信
            │
            └── engines/  专用 FlagOS 引擎
                  @EngineRegistry.register(device=..., vendor=...)
                  ├── FSDPFlagOSEngine     (LMHead / ValueHead)
                  └── MegatronFlagOSEngine (LMHead)
```

引擎查找采用 `(device, vendor)` 二级键:优先精确匹配厂商专用引擎,未命中时回退到设备级基础引擎;对 CUDA 兼容设备再回退到基础 CUDA 引擎。这一设计让新硬件的接入成本降到最低。

## 2.3 三大核心特性

### ① 硬件无关的平台抽象层(Platform Abstraction Layer)

通过统一的 `PlatformBase` 接口,将设备管理、集合通信、内存管理、profiler、rollout 环境变量等硬件相关逻辑抽象为标准方法。厂商只需实现一个平台类并用 `@PlatformRegistry.register` 注册,即可接入 verl。

针对 CUDA 兼容硬件(如沐曦、天数)`torch.cuda.is_available()` 在多种芯片上均返回 True 的问题,平台层引入 `vendor_name` 厂商标识与基于 SMI 命令的硬件探测(如 `mx-smi`),在首次自动检测时精确区分实际硬件,避免误匹配到 NVIDIA 引擎。

### ② 面向不同阶段配置的环境管理器(FLEnvManager)

RL 后训练的训练阶段与推理(rollout)阶段对算子加速和通信后端的需求不同。环境管理器按阶段(training / rollout)分别管理 FlagGems 与 FlagCX 的配置:

- **FlagGems 算子加速**:支持按阶段独立配置算子**白名单 / 黑名单**,精细控制哪些算子走 FlagGems 加速路径,并支持算子命中记录,便于调优与问题定位。
- **FlagCX 统一通信**:通过 `USE_FLAGCX` 开关启用 FlagCX 异构通信库,在多芯片环境下提供统一的集合通信,未启用时自动回退到设备原生后端(如 NCCL)。

### ③ 针对 Megatron 和 FSDP 的专用 FlagOS 引擎

在 verl 原生 FSDP 与 Megatron 引擎基础上派生 FlagOS 专用引擎,覆盖 RL 训练的关键角色:

- `FSDPFlagOSEngineWithLMHead` / `FSDPFlagOSEngineWithValueHead`——支持 fsdp / fsdp2,覆盖策略模型与价值模型;
- `MegatronFlagOSEngineWithLMHead`——支持 Megatron 大规模并行训练。

引擎在 `initialize` 阶段自动依据环境配置注入 FlagGems 算子加速,对上层 RL 算法完全透明,业务代码无需感知底层硬件差异。

> 基于上述三层设计,VeRL-plugin-FL 已在 **NVIDIA、沐曦、天数智芯、寒武纪** 等多种硬件上完成验证,同一份 RL 训练代码无需修改即可跨芯片运行。

## 2.4 verl 0.8.1 版本新增内容

FlagOS 社区在 verl 0.8.1 中的主要贡献:

- **新增多芯片硬件验证支持**:本次新增**沐曦(MetaX)、天数智芯(Iluvatar)、寒武纪(Cambricon MLU)** 三类硬件的完整适配并**验证通过**,配合 FlagOS 统一软件栈,显著扩大 verl 在国产及异构芯片上的兼容性;
- **新增 FlagOS 多芯片引擎**:随 0.8.1 引入 FSDP 与 Megatron 的 FlagOS 专用引擎,覆盖策略模型与价值模型训练;
- **接入插件化平台抽象层**:配合 verl 核心的 Platform / Engine 注册机制([verl#6086](https://github.com/verl-project/verl/pull/6086)),硬件厂商可通过独立插件包接入 verl,无需修改主框架;
- **提供统一适配标准与文档**:随插件开源各硬件的用户指南与适配验收标准(GSM8K GRPO 基线、`critic/rewards/mean` 曲线与 NVIDIA 参考对齐等),为后续更多厂商接入提供可复用模板。
