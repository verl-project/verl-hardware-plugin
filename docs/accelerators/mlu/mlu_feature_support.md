# MLU backend feature support

本文将介绍MLU上对于verl的适配工作及后端特性支持供开发者进行参考

---

## 目录
- [1. 支持模型](#1-模型支持)
- [2. 推理后端](#2-推理后端)
- [3. 训练后端](#3-训练后端)
  - [3.1 FSDP](#31-fsdp)
  - [3.2 Megatron](#32-megatron)
- [4. 融合算子](#4-融合算子)
- [5. 其他特性](#5-其他特性)

## 1. 模型支持
MLU 上已支持并测试完成 Qwen 系列模型，包括dense模型、MoE模型、VL模型等。
| 模型| 支持 |
| --- | --- |
| Qwen2.5 系列 | ✅ |
| Qwen2.5-VL 系列 | ✅ |
| Qwen3 系列 | ✅ |
| Qwen3-VL 系列 | ✅ |
| Qwen3.5 系列 | ✅ |

## 2. 推理后端

当前 MLU 已支持 vLLM 推理后端, SGLang 后端正在适配中。

vLLM 已预安装在 cambricon_release docker image 中，用户无需额外操作可以直接使用。

注意，当使用cambricon_release docker image进行RL训练时，请确认处在`pytorch_infer`的环境中。

#### vLLM主要参数特性支持

| vllm参数| verl对应通用参数 | 简介 |
| --- | --- | --- |
| `gpu_memory_utilization` | `actor_rollout_ref.rollout.gpu_memory_utilization` |推理阶段vLLM预留的GPU memory占比，数值在0.0 到 1.0 之间。建议设置值为0.5-0.8|
| `enforce_eager`| `actor_rollout_ref.rollout.enforce_eager` |禁用图模式，verl默认为False|
| `enable_chunked_prefill`| `actor_rollout_ref.rollout.enable_chunked_prefill` |分块预填充允许将大预填充分块成更小的块，并将它们与解码请求一起批处理。|
| `free_cache_engine`| `actor_rollout_ref.rollout.free_cache_engine`  |在部署生成阶段之后卸载 KVCache，默认值为 True。|
| `enable_prefix_caching` | `actor_rollout_ref.rollout.enable_prefix_caching` |用于启用自动前缀缓存 |
| `max_model_len` | `actor_rollout_ref.rollout.max_model_len` |模型能够处理的最大序列长度。它限制了单个输入序列的最大长度 |
| `tp_size`|  `actor_rollout_ref.rollout.tensor_model_parallel_size * data_parallel_size`|TP并行度|
| `dp_size`| `actor_rollout_ref.rollout.data_parallel_size`|DP并行度|
| `ep_size`| `actor_rollout_ref.rollout.expert_parallel_size`|EP并行度|
| `max_num_seqs` | `actor_rollout_ref.rollout.max_num_seqs` |正在运行的请求的最大数量|
| `max_num_batched_tokens`| `actor_rollout_ref.rollout.max_num_batched_tokens` |在一次批处理（batch）中可以处理的最大总Token数|

vLLM-mlu 支持 `BATCH_INVARIANT` 模式，通过`VLLM_BATCH_INVARIANT=1`环境变量开启。

此外，vLLM-mlu支持部分算子使用小算子实现，以达到更好的训推一致性。目前支持的算子可通过在脚本中添加以下参数的方式开启：

```bash
ray job submit \
    --runtime-env=runtime_env.yaml \
    -- \
    python3 -m verl.trainer.main_ppo \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.additional_config.training_config.rms_norm_match_verl_residual_add=true \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.additional_config.training_config.use_torch_rope=true \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.additional_config.training_config.use_torch_topk_topp=true \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.additional_config.training_config.use_torch_random_sample=true \
```


## 3. 训练后端

### 3.1 FSDP

MLU 支持 FSDP1 和 FSDP2，主要参数如下：

#### FSDP 参数特性支持
| verl参数 | 简介|
| --- | --- |
| `actor_rollout_ref.actor.fsdp_config.param_offload` |是否卸载模型权重到CPU，默认值为False|
| `actor_rollout_ref.actor.fsdp_config.optimizer_offload` |是否卸载优化器状态到CPU，默认值为False|
| `actor_rollout_ref.actor.fsdp_config.reshard_after_forward` |控制前向计算后的参数行为，平衡内存与通信。默认值为True：前向后重新分片参数，反向时重新全收集|
| `actor_rollout_ref.actor.fsdp_config.fsdp_size` | 每个FSDP分片组中的MLU数量；默认值-1表示自动。|
| `actor_rollout_ref.actor.ulysses_sequence_parallel_size`|Ulysses序列并行大小|
| `actor_rollout_ref.actor.entropy_from_logits_with_chunking`|通过分块计算熵以减少显存峰值，默认值为False|
| `actor_rollout_ref.actor.fsdp_config.entropy_checkpointing`|在训练时对熵计算启用重计算,降低显存峰值，默认值为False|
| `actor_rollout_ref.actor.fsdp_config.forward_only` |是否只进行前向计算，默认值为False|

更多参数配置参考 [verl fsdp config](https://github.com/verl-project/verl/blob/main/verl/trainer/config/engine/fsdp.yaml)

### 3.2 Megatron

MLU 支持 Megatron 和 Megatron-Bridge，且已经预安装在 cambricon_release docker image 中，主要参数配置如下：

#### 参数特性支持
| verl参数 | 简介|
| --- | --- |
| `actor_rollout_ref.actor.megatron.optimizer_offload` |是否卸载模型优化器到CPU，默认值为False|
| `actor_rollout_ref.actor.megatron.param_offload` |是否卸载模型权重到CPU，默认值为False|
| `actor_rollout_ref.actor.megatron.tensor_model_parallel_size` | 张量并行大小；默认值为1。|
| `actor_rollout_ref.actor.megatron.pipeline_model_parallel_size`  |流水并行大小，默认值为1|
| `actor_rollout_ref.actor.megatron.expert_model_parallel_size` | 专家并行大小，默认值为1|
| `actor_rollout_ref.actor.megatron.expert_tensor_parallel_size`|TP拓展EP大小，默认值为null|
| `actor_rollout_ref.actor.megatron.context_parallel_size`|上下文并行大小，默认值为1|
| `actor_rollout_ref.actor.megatron.override_transformer_config.persist_layer_norm` |是否使用持久化 LayerNorm，默认值为False|
| `actor_rollout_ref.actor.megatron.override_transformer_config.moe_grouped_gemm` |是否使用Group GEMM，默认值为False|
| `actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_dtype` |用于路由和专家输出加权平均的数据类型。使用 fp32 或 fp64 可以提高稳定性，尤其是在专家数量较多时，默认值为fp32|
| `actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity` |重新计算激活的粒度，可选项为'full', 'selective' and 'none'。其中full代表重新计算整个transformer layer，selective代表只计算transformer layer中的核心注意力部分。默认为'none'。|
| `actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method` |该参数需将recompute_granularity设置为'full'才生效，可选项为'uniform', 'block'。默认为None。|
| `actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers` |该参数需将recompute_granularity设置为'full'才生效，默认为None。若recompute_method设置为uniform，该参数含义为每个均匀划分的重新计算单元的transformer layers数量。例如你可以指定为--recompute_granularity full --recompute_method uniform --recompute_num_layers 4。recompute_num_layers越大，显存占用越小，计算成本越大。注意：当前进程中的模型层数需能被recompute_num_layers整除。默认为None。|
| `actor_rollout_ref.actor.megatron.override_transformer_config.use_flash_attn` |是否使用Flash Attention，默认值为true|
| `actor_rollout_ref.actor.megatron.override_transformer_config.num_layers_in_first_pipeline_stage` |第一个pipeline stage 的层数，默认值为none|
| `actor_rollout_ref.actor.megatron.override_transformer_config.num_layers_in_last_pipeline_stage` |最后一个pipeline stage 的层数，默认值为none|

更多参数配置参考 [verl megatron config](https://github.com/verl-project/verl/blob/main/verl/trainer/config/engine/megatron.yaml)

## 4. 融合算子
| 算子 | 使用方式 | 说明 |
| ------- | --- | --- |
| linear_cross_entropy | `actor_rollout_ref.model.use_fused_kernels=True`<br>`actor_rollout_ref.model.fused_kernel_options.impl_backend=torch / triton` | `linear_cross_entropy` 融合算子，可选支持后端 `torch` 或 `triton`，MLU两种后端都支持。
| liger kernel | `actor_rollout_ref.model.use_liger=True` | 使用liger kernel中的融合算子，例如 `swiglu`、`rope`、`rms_norm` 等。训练的网络不同，适配的融合算子可能有所不同。注意，使用时需要安装liger kernel mlu版本。|
| megatron 中的融合算子 | `+actor_rollout_ref.actor.megatron.override_transformer_config.use_fused_rmsnorm=True`<br>`+actor_rollout_ref.actor.megatron.override_transformer_config.use_fused_rotary_pos_emb=True`<br>`+actor_rollout_ref.actor.megatron.override_transformer_config.use_fused_swiglu=True`<br>`+actor_rollout_ref.actor.megatron.override_transformer_config.use_fused_moe_token_permute_and_unpermute=True` | 使用Megatron中的融合算子，根据需求选择需要开启的算子。 |


## 5. 其他特性
这里列出部分MLU支持的高级特性，供开发者参考
| 特性 | 说明 |
| -------- | --- |
| SFT 训练 | 使用verl框架进行sft训练，运行脚本参考[SFT example](https://github.com/verl-project/verl/tree/main/examples/sft/gsm8k) |
| on policy distillation 训练 | OPD 在学生策略采样得到的状态上，将教师模型的知识蒸馏到学生模型。对比监督微调 (SFT) 或者标准知识蒸馏 (KD)，OPD 对齐训练阶段与推理阶段的状态，从而缓解暴露偏差。<br>运行及其详细说明见 [OPD example](https://github.com/verl-project/verl/tree/main/examples/on_policy_distillation_trainer)。|
| Router Replay | Router Replay 是 verl 框架中面向混合专家（MoE）模型的高级路由回放功能。它通过记录并重放路由决策来实现确定性训练，保证训练计算多次之间模型行为保持一致。 |





