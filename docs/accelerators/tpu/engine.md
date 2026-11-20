# Google TPU TorchTitan Training Engine

The plugin runs verl's actor and reference model training on Cloud TPU using TorchTitan (`TorchTitanTPUEngineWithLMHead` under `verl_hardware_plugin/accelerators/tpu/engines/torchtitan_tpu.py`).

## Configuration (`engine_kwargs.tpu`)

TPU options of the TorchTitan engine go in its config, under `engine_kwargs.tpu`, like verl's other
per-backend options (`rollout.engine_kwargs.vllm`), e.g.
`+actor_rollout_ref.actor.torchtitan.engine_kwargs.tpu.use_simple_fsdp=False`. Supported keys:
`use_splash_attention`, `use_simple_fsdp`, `use_fused_linear_cross_entropy`, `fuse_qkv`,
`tpu_eager_mode`, `optim_impl`, `seq_bucket_size`, `static_micro_batch_len`, `splash_max_block_size`,
`mixed_precision_reduce`, `moe_paired_grouped_mm`, `moe_native_weight_layout`, `trainer_mode`, `graph_trainer`, `model_kernels`,
`ep_capacity_factor`, `qwen3_pallas_experts`, `moe_paired_gemm`,
`context_parallel_strategy`, `context_parallel_gdn_strategy` and `context_parallel_gdn_reduce_dtype` (see
`TPUEngineOptions` in `options.py`); an unknown key is an error. The older `torchtitan.<option>` fields and `VERL_TPU_*` environment variables still work but are
deprecated and log a warning.

## Context Parallelism

`context_parallel_size > 1` follows torchtitan's TPU CP recipes: the full-attention layers run the TPU CP inner
attention of `context_parallel_strategy` (`all_gather`, the default, needs Splash; or `ulysses`). A hybrid model's
other sequence mixers carry state through the whole sequence, so they need their own CP transform too. The engine
looks them up by the layer types the model config contains (`engines/cp_layer_transforms.py`), so it holds no model
names, and it raises if the model needs a transform the installed torchtitan does not ship, rather than running those
layers on each rank's token shard (wrong log-probs and gradients, with no error).

| Model | Non-attention layers | CP support | Requires |
|---|---|---|---|
| Dense / MoE transformers (Qwen3, Qwen3 MoE, Llama) | none | `all_gather`, `ulysses` | torchtitan TPU CP (`experiments/tpu/cp_attention.py`) |
| Qwen3.5 (dense / MoE) | GatedDeltaNet (head-parallel under CP) | `all_gather`, `ulysses` | torchtitan with google-pytorch/torchtitan#201 (`experiments/tpu/qwen3_5/cp_gated_delta.py`, commit `36643b4c9` or later); older torchtitan raises |

Qwen3.5 under CP: `n_key_heads` and `n_value_heads` must be divisible by the CP degree, and tensor parallelism is not
supported (torchtitan's `validate_gated_delta_context_parallel` checks both). `gdn_row_slot_len` is rejected under
`all_gather` CP: the head-tail load balancer shards each packed micro-batch as one sequence, but the CP GatedDeltaNet
would read it as rows. `context_parallel_gdn_strategy` (`all_to_all` / `allgather_reducescatter`) and
`context_parallel_gdn_reduce_dtype` (`float32` / `bfloat16`) go to torchtitan's `GatedDeltaNetContextParallelTransform`
as `strategy` / `reduce_dtype`. They need a torchtitan whose transform has those fields; with an older one, setting them
is an error. Unset, they keep the transform's defaults. Another layer type registers its transform with
`cp_layer_transforms.register_cp_layer_transform`.

## MoE Support (Qwen3 MoE)

Qwen3 MoE models (HF `qwen3_moe`, e.g. Qwen3-30B-A3B) need no extra option. The engine turns off torchtitan's
expert-bias load balancing, so the trainer picks the same experts as vLLM, and runs the routed experts with torchtitan's
TPU token dispatcher and Pallas grouped matmul (required: torchtitan's default `torch._grouped_mm` path has a wrong
backward on TPU). This needs `expert_parallel_size=1` and a `seq_bucket_size` that times `num_experts_per_tok` is a
multiple of 256 (the default 256 is); otherwise the engine raises. `use_dynamic_bsz=True` works on TPU: the plugin runs
its int64 MAX / MIN all-reduces in int32, which `tpu_dist` supports.

For Qwen3-30B-A3B, actor and reference models also disable MoE and router auxiliary loss.
The legacy `qwen3_pallas_experts=False` is rejected because Pallas is required. For this model,
`moe_paired_gemm` is a fallback for `moe_paired_grouped_mm`; an explicitly set `moe_paired_grouped_mm`
takes precedence. With neither set, paired matmul stays off.

MoE models also require the fused LM-head log-prob (`use_fused_linear_cross_entropy`, on by default for them, with or
without `use_torch_compile`): Tokamax's fused linear + softmax + cross-entropy Pallas kernel computes the per-token
log-probs without writing the `[tokens, vocab]` logits (151,936 columns for Qwen3) to HBM, which is what lets a
micro-batch of tens of thousands of tokens per chip fit. Its backward runs in PyTorch, in chunks of 2048 tokens (with
tokamax 0.0.14, the Mosaic-TPU backward kernel returns a wrong LM-head weight gradient). Entropy, when a pass reports it without
differentiating it (old log-probs), is computed one chunk of tokens at a time. If the kernel cannot be used
(`tensor_parallel_size > 1`, tokamax missing) the engine raises instead of falling back to the logits; set
`use_fused_linear_cross_entropy=False` to run without it. The engine logs the log-prob path it uses
(`TPU log-prob path: ...`).

## Performance Options

Performance options for long packed micro-batches (the Qwen3-30B-A3B recipe sets them):

| Option | Effect |
|---|---|
| `splash_max_block_size: 1024` | Largest Splash attention block (torchtitan's default 512). Splash's grid grows as `(tokens / block)^2`; 1024 cut `update_actor` by 17% at 24k tokens per chip on v5p |
| `static_micro_batch_len: <tokens>` | Pads every packed micro-batch to at least this length (e.g. the `use_dynamic_bsz` token budget), so the trainer compiles one shape instead of recompiling when the packed length reaches a new bucket (`static_max_response_len` does the same for the response length) |
| `moe_paired_grouped_mm: true` | Qwen3 MoE: torchtitan's paired W1/W3 Pallas grouped matmul (shares the routed-token loads) |
| `moe_native_weight_layout: true` | Qwen3 MoE: kernel-native `(E, K, N)` expert weights; `state_dict()` / `load_state_dict()` keep the canonical layout and names, so the HF load and the tpu-sync expert export are unchanged |
| `mixed_precision_reduce: bfloat16` | bf16 FSDP gradient reduce-scatter instead of fp32 (half the traffic; a performance A/B option) |
| `rollout_expert_parallel: true` | Export stacked MoE experts (Qwen3.5 / Qwen3 MoE) in vllm-torchtpu's expert-parallel layout so the rollout receives them zero-copy into `w13_weight` / `w2_weight` without an HBM staging copy. When the rollout runs quantized MoE (`QUANTIZE_ON_LOAD_PREFIXES=experts` with `VLLM_RAY_EXTRA_ENV_VARS_TO_COPY=QUANTIZE_ON_LOAD_PREFIXES`), the trainer auto-discovers the rollout's `(quant_dtype, block_size)` format via `describe_vllm_params` and sends quantized experts + FP32 scales |
| `rollout_fp8_moe: true / false` | Explicit override for trainer-side MoE quantization on the zero-copy expert-parallel path (`true` forces trainer-side quantization and implies `rollout_expert_parallel=true`; `false` keeps bf16 export) |
| `gdn_row_slot_len: <max_model_len>` | Qwen3.5: every sequence of a packed micro-batch starts at a multiple of this length, so the GatedDeltaNet runs one sequence per row (torchtitan's `set_unflatten_seq_len`) instead of resetting at packed documents through Tokamax `segment_ids`. Needs `use_dynamic_bsz=False`, a fixed `ppo_micro_batch_size_per_gpu`, `static_micro_batch_len` = that size x `gdn_row_slot_len`, and the fused log-prob |
| `trainer_mode: graph_trainer` | Actor only (bypass mode, KL loss off, `use_simple_fsdp=True`, `static_micro_batch_len` set): traces the micro-batch forward, fused LM-head log-prob, dual-clip PPO loss and backward once with TorchTitan's `TpuGraphTrainer` (`aot_fx_trace`) and compiles it as one PJRT executable instead of per-block `torch.compile` |
| `graph_trainer: <dict>` | Optional compiler configuration passed straight to TorchTitan's `TpuGraphTrainerCompileConfig(**cfg)` (e.g. `memory_policy`, `save_regions`, `fsdp_barrier_policy`, `graph_split`); ignored when `trainer_mode` is not `graph_trainer` |

## Verify

On any host, without a TPU:

```bash
pytest tests/accelerators/tpu/test_tpu_engine.py tests/accelerators/tpu/test_tpu_graph_trainer_mode.py tests/accelerators/tpu/test_tpu_options.py -v
```

## Related Documentation

- [User Guide](./README.md)
- [Installation Guide](./install_guidance.md)
- [Rollout (vLLM & SGLang)](./rollout.md)
- [Weight Sync](./weight_sync.md)
