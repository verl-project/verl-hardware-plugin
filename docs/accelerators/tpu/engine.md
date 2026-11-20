# Google TPU TorchTitan Training Engine

The plugin runs verl's actor and reference model training on Cloud TPU using TorchTitan (`TorchTitanTPUEngineWithLMHead` under `verl_hardware_plugin/accelerators/tpu/engines/torchtitan_tpu.py`).

## Configuration (`engine_kwargs.tpu`)

TPU options of the TorchTitan engine go in its config, under `engine_kwargs.tpu`, like verl's other
per-backend options (`rollout.engine_kwargs.vllm`), e.g.
`+actor_rollout_ref.actor.torchtitan.engine_kwargs.tpu.use_simple_fsdp=False`. Supported keys:
`use_splash_attention`, `use_simple_fsdp`, `use_fused_linear_cross_entropy`, `fuse_qkv`,
`tpu_eager_mode`, `optim_impl`, `seq_bucket_size`, `static_micro_batch_len`, `static_max_response_len`,
`loss_row_bucketing`, `static_micro_batch_num_seqs`, `splash_max_block_size`,
`mixed_precision_reduce`, `moe_paired_grouped_mm`, `moe_native_weight_layout`, `trainer_mode`, `graph_trainer`, `model_kernels`,
`ep_capacity_factor`, `qwen3_pallas_experts`, `moe_paired_gemm` and
`context_parallel_strategy` (see `TPUEngineOptions` in `options.py`); an unknown key is an error. The older `torchtitan.<option>` fields and `VERL_TPU_*` environment variables still work but are
deprecated and log a warning.

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
| `loss_row_bucketing: false` | Turns off PPO loss row bucketing (on by default). With it, the PPO loss inputs of every micro-batch are padded with fully masked rows to `tpu_padded_num_seqs`: the most sequences in any micro-batch of the mini-batch on any rank, rounded up to a power of two (the same sequence-count buckets as the Tokamax GatedDeltaNet document budget). The loss then does not recompile whenever `use_dynamic_bsz` changes the number of sequences per micro-batch. Only applied with verl's `ppo_loss` and a policy loss whose loss, gradients and metrics ignore fully masked rows (`vanilla`, `gspo`, `sapo`, `gpg`, `clip_cov`, `kl_cov`, `dro`, `cispo`, `dppo_tv`, `dppo_kl`, any `loss_agg_mode`); `geo_mean` (row mean) and custom `register_policy_loss` losses keep the exact row count, and so does `bypass_mode` unless `static_micro_batch_num_seqs` is set (see [Loss rows in bypass mode](#loss-rows-in-bypass-mode)) |
| `static_micro_batch_num_seqs: <sequences>` | With row bucketing, pad the PPO loss inputs to at least this many rows (e.g. the most sequences a micro-batch can hold), so the loss keeps one row shape. With it, `bypass_mode` gets padded rows too (see [Loss rows in bypass mode](#loss-rows-in-bypass-mode)) |
| `moe_paired_grouped_mm: true` | Qwen3 MoE: torchtitan's paired W1/W3 Pallas grouped matmul (shares the routed-token loads) |
| `moe_native_weight_layout: true` | Qwen3 MoE: kernel-native `(E, K, N)` expert weights; `state_dict()` / `load_state_dict()` keep the canonical layout and names, so the HF load and the tpu-sync expert export are unchanged |
| `mixed_precision_reduce: bfloat16` | bf16 FSDP gradient reduce-scatter instead of fp32 (half the traffic; a performance A/B option) |
| `gdn_row_slot_len: <max_model_len>` | Qwen3.5: every sequence of a packed micro-batch starts at a multiple of this length, so the GatedDeltaNet runs one sequence per row (torchtitan's `set_unflatten_seq_len`) instead of resetting at packed documents through Tokamax `segment_ids`. Needs `use_dynamic_bsz=False`, a fixed `ppo_micro_batch_size_per_gpu`, `static_micro_batch_len` = that size x `gdn_row_slot_len`, and the fused log-prob |
| `trainer_mode: graph_trainer` | Actor only (bypass mode, KL loss off, `use_simple_fsdp=True`, `static_micro_batch_len` set): traces the micro-batch forward, fused LM-head log-prob, dual-clip PPO loss and backward once with TorchTitan's `TpuGraphTrainer` (`aot_fx_trace`) and compiles it as one PJRT executable instead of per-block `torch.compile` |
| `graph_trainer: <dict>` | Optional compiler configuration passed straight to TorchTitan's `TpuGraphTrainerCompileConfig(**cfg)` (e.g. `memory_policy`, `save_regions`, `fsdp_barrier_policy`, `graph_split`); ignored when `trainer_mode` is not `graph_trainer` |

### Loss rows in bypass mode

Without `static_micro_batch_num_seqs`, `bypass_mode` keeps the exact rows. With it, its PPO loss inputs get padding
rows (empty sequences) like the losses listed above. Its loss, its gradients and its masked metrics do not change.
Some of its per-sequence rollout-correction diagnostics reduce over every loss row, though, and verl 0.9.1 counts the
padding rows in them:

- the sequence-level IS statistics, `rollout_corr/rollout_is_seq_*`;
- with `rollout_is=sequence`, `rollout_corr/rollout_is_max`, `rollout_corr/rollout_is_min`,
  `rollout_corr/rollout_is_ratio_fraction_high` and `rollout_corr/rollout_is_ratio_fraction_low`. A padding row has
  the sequence ratio 1, so the ratios 2 and 4, for example, get a minimum of 1, and both fractions count the padding
  rows in their denominator;
- with rejection sampling, `rollout_corr/rollout_rs_seq_masked_fraction`;
- the off-policy perplexity and chi-square metrics, `rollout_corr/*ppl*` and `rollout_corr/chi2_seq`.

verl main leaves rows without valid tokens out of all of these (verl-project/verl#8628 for the IS and
rejection-sampling metrics, verl-project/verl#7785 for the perplexity and chi-square ones), so there the padding rows
change none of them. With verl 0.9.1 and rejection sampling, the rejection-sampling metrics also pick the sequences
with valid tokens through a boolean mask, so a few shapes inside the loss follow the data with or without padding rows.

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
