# Google TPU TorchTitan Training Engine

The plugin runs verl's actor and reference model training on Cloud TPU using TorchTitan (`TorchTitanTPUEngineWithLMHead` under `verl_hardware_plugin/accelerators/tpu/engines/torchtitan_tpu.py`).

## Configuration (`engine_kwargs.tpu`)

TPU options of the TorchTitan engine go in its config, under `engine_kwargs.tpu`, like verl's other
per-backend options (`rollout.engine_kwargs.vllm`), e.g.
`+actor_rollout_ref.actor.torchtitan.engine_kwargs.tpu.use_simple_fsdp=False`. Supported keys:
`use_splash_attention`, `use_simple_fsdp`, `use_fused_linear_cross_entropy`, `fuse_qkv`,
`tpu_eager_mode`, `optim_impl`, `seq_bucket_size`, `static_micro_batch_len`,
`static_log_prob_micro_batch_len`, `static_max_response_len`, `loss_row_bucketing`, `static_micro_batch_num_seqs`,
`splash_max_block_size`, `mixed_precision_reduce`, `moe_paired_grouped_mm`, `moe_native_weight_layout`,
`cp_full_positions`, `gdn_auto_row_slot_len`, `fuse_gdn_projections`, `sort_micro_batches_by_slot`,
`gdn_row_slot_len`, `trainer_mode`, `graph_trainer`, `precompile_train_step`, `model_kernels`,
`ep_capacity_factor`, `ep_transport`, `qwen3_pallas_experts`, `moe_paired_gemm`,
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
| Dense / MoE transformers (Qwen3, Qwen3 MoE, Llama) | none | `all_gather`, `ulysses` | torchtitan TPU CP (`experiments/tpu/distributed/cp_attention.py`) |
| Qwen3.5 (dense / MoE) | GatedDeltaNet (head-parallel under CP) | `all_gather`, `ulysses` | torchtitan with google-pytorch/torchtitan#201 (`experiments/tpu/qwen3_5/cp_gated_delta.py`, commit `36643b4c9` or later); older torchtitan raises |

Qwen3.5 under CP: `n_key_heads` and `n_value_heads` must be divisible by the CP degree, and tensor parallelism is not
supported (torchtitan's `validate_gated_delta_context_parallel` checks both). Static `gdn_row_slot_len` is rejected for
all `context_parallel_size > 1` (`all_gather` and `ulysses` alike): static row-slot permutation does not permute labels
before `cp_shard` or post-gather-unpermute outputs across CP shards. `context_parallel_gdn_strategy` (`all_to_all` /
`allgather_reducescatter`) and `context_parallel_gdn_reduce_dtype` (`float32` / `bfloat16`, controlling both the
forward `out_proj` reduce-scatter and the backward input-gradient reduce-scatter under
`context_parallel_gdn_strategy='allgather_reducescatter'`) go to torchtitan's `GatedDeltaNetContextParallelTransform` as
`strategy` / `reduce_dtype`. Setting them requires `context_parallel_size > 1`, a model with GatedDeltaNet layers, and a
torchtitan whose transform has those fields (`context_parallel_gdn_reduce_dtype` also requires
`context_parallel_gdn_strategy='allgather_reducescatter'`). Unset, they keep the transform's defaults. Another layer
type registers its transform with `cp_layer_transforms.register_cp_layer_transform`.

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
micro-batch of tens of thousands of tokens per chip fit. Its per-token backward runs in Tokamax's backward kernel too
(TorchTitan's `linear_softmax_cross_entropy_loss` with `reduction="none"`). Entropy, when a pass reports it without
differentiating it (old log-probs), is computed one chunk of tokens at a time. If the kernel cannot be used
(`tensor_parallel_size > 1`, tokamax missing) the engine raises instead of falling back to the logits; set
`use_fused_linear_cross_entropy=False` to run without it. The engine logs the log-prob path it uses
(`TPU log-prob path: ...`).

## Performance Options

Performance options for long packed micro-batches (the Qwen3-30B-A3B recipe sets them):

| Option | Effect |
|---|---|
| `splash_max_block_size: 1024` | Largest Splash attention block (torchtitan's default 512). Splash's grid grows as `(tokens / block)^2`; 1024 cut `update_actor` by 17% at 24k tokens per chip on v5p |
| `static_log_prob_micro_batch_len: <tokens>` | Static packed micro-batch length for forward-only (old/reference log-prob) passes, overriding `static_micro_batch_len` when `forward_only=True` (must be `>= gdn_auto_row_slot_len` and divisible by `gdn_auto_row_slot_len` when `gdn_auto_row_slot_len` is set) |
| `cp_full_positions: true` | Pass pre-`cp_shard` unsharded 1-D text `positions` to `model.forward(cp_full_positions=...)` (`google-pytorch/torchtitan#216`), removing per-layer `all_gather(positions)` under Ulysses and KV all-gather CP (fails loudly on multimodal / 3-D MRoPE batches) |
| `gdn_auto_row_slot_len: <tokens>` | Qwen3.5 (`use_dynamic_bsz=True` compatible, `google-pytorch/torchtitan#216`): dynamic per-micro-batch row-slot planner (e.g. `32768`) that permutes eligible single-doc sequences starting at position `0` into slots with `set_unflatten_seq_len(slot_len)` and `gated_delta_document_resets=False`, runs 1-doc micro-batches unsegmented at `padded_len`, and keeps multi-doc / continuation micro-batches segmented (cross-rank `BAND` agreement; works with Ulysses CP and non-CP; requires `use_fused_linear_cross_entropy=True`, `tensor_parallel_size=1`, and any set `static_micro_batch_len` / `static_log_prob_micro_batch_len` to be `>=` and divisible by `gdn_auto_row_slot_len`) |
| `fuse_gdn_projections: true` | Qwen3.5 (`google-pytorch/torchtitan#216`): set `fuse_input_projections=True` on GatedDeltaNet configs, fusing the six bias-free input projections into one matmul (requires `tensor_parallel_size=1`) |
| `sort_micro_batches_by_slot: true` | Qwen3.5 (`use_dynamic_bsz=True`): order micro-batches by `(slot_category, -total_tokens, idx)` so identical slot/segmentation shapes execute consecutively |
| `static_micro_batch_len: <tokens>` | Pads every packed micro-batch to at least this length (e.g. the `use_dynamic_bsz` token budget), so the trainer compiles one shape instead of recompiling when the packed length reaches a new bucket (`static_max_response_len` does the same for the response length; must be `>= gdn_auto_row_slot_len` and divisible by `gdn_auto_row_slot_len` when `gdn_auto_row_slot_len` is set) |
| `loss_row_bucketing: false` | Turns off PPO loss row bucketing (on by default). With it, the PPO loss inputs of every micro-batch are padded with fully masked rows to `tpu_padded_num_seqs`: the most sequences in any micro-batch of the mini-batch on any rank, rounded up to a power of two (the same sequence-count buckets as the Tokamax GatedDeltaNet document budget). The loss then does not recompile whenever `use_dynamic_bsz` changes the number of sequences per micro-batch. Only applied with verl's `ppo_loss` and a policy loss whose loss, gradients and metrics ignore fully masked rows (`vanilla`, `gspo`, `sapo`, `gpg`, `clip_cov`, `kl_cov`, `dro`, `cispo`, `dppo_tv`, `dppo_kl`, any `loss_agg_mode`); `geo_mean` (row mean), `bypass_mode` (its `rollout_corr/*` metrics average over rows) and custom `register_policy_loss` losses keep the exact row count |
| `static_micro_batch_num_seqs: <sequences>` | With row bucketing, pad the PPO loss inputs to at least this many rows (e.g. the most sequences a micro-batch can hold), so the loss keeps one row shape |
| `moe_paired_grouped_mm: true` | Qwen3 MoE: torchtitan's paired W1/W3 Pallas grouped matmul (shares the routed-token loads) |
| `moe_native_weight_layout: true` | Qwen3 MoE: kernel-native `(E, K, N)` expert weights; `state_dict()` / `load_state_dict()` keep the canonical layout and names, so the HF load and the tpu-sync expert export are unchanged |
| `mixed_precision_reduce: bfloat16` | bf16 FSDP gradient reduce-scatter instead of fp32 (half the traffic; a performance A/B option) |
| `rollout_expert_parallel: true` | Export stacked MoE experts (Qwen3.5 / Qwen3 MoE) in vllm-torchtpu's expert-parallel layout so the rollout receives them zero-copy into `w13_weight` / `w2_weight` without an HBM staging copy. When the rollout runs quantized MoE (`QUANTIZE_ON_LOAD_PREFIXES=experts` with `VLLM_RAY_EXTRA_ENV_VARS_TO_COPY=QUANTIZE_ON_LOAD_PREFIXES`), the trainer auto-discovers the rollout's `(quant_dtype, block_size)` format via `describe_vllm_params` and sends quantized experts + FP32 scales |
| `rollout_fp8_moe: true / false` | Explicit override for trainer-side MoE quantization on the zero-copy expert-parallel path (`true` forces trainer-side quantization and implies `rollout_expert_parallel=true`; `false` keeps bf16 export) |
| `gdn_row_slot_len: <max_model_len>` | Qwen3.5 (`context_parallel_size=1`, use `gdn_auto_row_slot_len` with `context_parallel_strategy='ulysses'` under CP): every sequence of a packed micro-batch starts at a multiple of this length, so the GatedDeltaNet runs one sequence per row (torchtitan's `set_unflatten_seq_len`) instead of resetting at packed documents through Tokamax `segment_ids`. Needs `use_dynamic_bsz=False`, a fixed `ppo_micro_batch_size_per_gpu`, `static_micro_batch_len` = that size x `gdn_row_slot_len`, and the fused log-prob |
| `trainer_mode: graph_trainer` | Actor only (bypass mode, KL loss off, `use_simple_fsdp=True`, `static_micro_batch_len` set): traces the micro-batch forward, fused LM-head log-prob, dual-clip PPO loss and backward once with TorchTitan's `TpuGraphTrainer` (`aot_fx_trace`) and compiles it as one PJRT executable instead of per-block `torch.compile`. `expert_parallel_size > 1` traces TorchTitan's `TpuAllToAllTokenDispatcher` and, with `graph_split`, needs `backward_partition_mode: by_layer` with one layer per executable |
| `graph_trainer: <dict>` | Optional compiler configuration passed straight to TorchTitan's `TpuGraphTrainerCompileConfig(**cfg)` (e.g. `memory_policy`, `save_regions`, `fsdp_barrier_policy`, `graph_split`); ignored when `trainer_mode` is not `graph_trainer` |
| `precompile_train_step: true` | `trainer_mode: graph_trainer` with standalone rollout replicas (`separate_async`, a `tpu-sync` or `ray-plasma-store` weight sync): right after the initial weight sync, every trainer worker runs the training step once on a synthetic micro-batch (gradients dropped, no optimizer step, module buffers and RNG states restored), so the step is traced and compiled while the first rollout batch is generated instead of in the first `update_actor`. If the first real step needs another graph (padded length or loss constants), it retraces as without the option. Workers that host a rollout engine skip it; Tokamax KDA needs `kda_document_budget`, and `graph_trainer.optimizer_compile: in_graph` is rejected |

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
