# Google TPU TorchTitan Training Engine

The plugin runs verl's actor and reference model training on Cloud TPU using TorchTitan (`TorchTitanTPUEngineWithLMHead` under `verl_hardware_plugin/accelerators/tpu/engines/torchtitan_tpu.py`).

## Configuration (`engine_kwargs.tpu`)

TPU options of the TorchTitan engine go in its config, under `engine_kwargs.tpu`, like verl's other
per-backend options (`rollout.engine_kwargs.vllm`), e.g.
`+actor_rollout_ref.actor.torchtitan.engine_kwargs.tpu.use_simple_fsdp=False`. Supported keys:
`use_splash_attention`, `use_simple_fsdp`, `use_fused_linear_cross_entropy`, `fuse_qkv`,
`tpu_eager_mode`, `optim_impl`, `seq_bucket_size`, `static_micro_batch_len`, `static_max_response_len`,
`static_micro_batch_seqs`, `splash_max_block_size`,
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
| `static_micro_batch_seqs: <sequences>` | Pads the PPO loss inputs of every micro-batch to at least this many rows, so the number of sequences that `use_dynamic_bsz` puts in a micro-batch no longer changes the shapes of the loss and its metrics. The extra rows are empty sequences (zero response mask), so the loss and its gradients do not change; with the `geo_mean` loss mode, which averages over every row, the loss inputs keep their exact rows. With Tokamax's GatedDeltaNet kernels it is also the floor of their document budget. See [Static micro-batches](#static-micro-batches) |
| `moe_paired_grouped_mm: true` | Qwen3 MoE: torchtitan's paired W1/W3 Pallas grouped matmul (shares the routed-token loads) |
| `moe_native_weight_layout: true` | Qwen3 MoE: kernel-native `(E, K, N)` expert weights; `state_dict()` / `load_state_dict()` keep the canonical layout and names, so the HF load and the tpu-sync expert export are unchanged |
| `mixed_precision_reduce: bfloat16` | bf16 FSDP gradient reduce-scatter instead of fp32 (half the traffic; a performance A/B option) |
| `gdn_row_slot_len: <max_model_len>` | Qwen3.5: every sequence of a packed micro-batch starts at a multiple of this length, so the GatedDeltaNet runs one sequence per row (torchtitan's `set_unflatten_seq_len`) instead of resetting at packed documents through Tokamax `segment_ids`. Needs `use_dynamic_bsz=False`, a fixed `ppo_micro_batch_size_per_gpu`, `static_micro_batch_len` = that size x `gdn_row_slot_len`, and the fused log-prob |
| `trainer_mode: graph_trainer` | Actor only (bypass mode, KL loss off, `use_simple_fsdp=True`, `static_micro_batch_len` set): traces the micro-batch forward, fused LM-head log-prob, dual-clip PPO loss and backward once with TorchTitan's `TpuGraphTrainer` (`aot_fx_trace`) and compiles it as one PJRT executable instead of per-block `torch.compile` |
| `graph_trainer: <dict>` | Optional compiler configuration passed straight to TorchTitan's `TpuGraphTrainerCompileConfig(**cfg)` (e.g. `memory_policy`, `save_regions`, `fsdp_barrier_policy`, `graph_split`); ignored when `trainer_mode` is not `graph_trainer` |

### Static micro-batches

With `use_dynamic_bsz=True`, the three `static_*` options give every micro-batch one packed length, one response width
and one row count of the PPO loss inputs, so the shapes of the trainer's forward, loss and backward programs no longer
depend on how many tokens or sequences a micro-batch holds:

```bash
actor_rollout_ref.actor.use_dynamic_bsz=True
actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192
+actor_rollout_ref.actor.torchtitan.engine_kwargs.tpu.static_micro_batch_len=8192
+actor_rollout_ref.actor.torchtitan.engine_kwargs.tpu.static_max_response_len=4096
+actor_rollout_ref.actor.torchtitan.engine_kwargs.tpu.static_micro_batch_seqs=64
```

Set `static_micro_batch_len` to the token budget, `static_max_response_len` to `data.max_response_length`, and
`static_micro_batch_seqs` to the most sequences a micro-batch holds; it never needs to exceed `ppo_mini_batch_size` x
`rollout.n` / data-parallel size, the sequences of one rank's mini-batch. The padding rows add work that grows with
`static_micro_batch_seqs` (verl slices the loss inputs one row at a time on the host), so a value close to the largest
count your run reaches costs less than a much larger one. A micro-batch longer, wider or with more sequences than these
sizes keeps its own size and compiles once for it; the engine logs every new sequence count above
`static_micro_batch_seqs`. With `use_dynamic_bsz=False` every micro-batch already has `ppo_micro_batch_size_per_gpu`
sequences, so `static_micro_batch_seqs` is not needed there.

Only verl's `ppo_loss` gets padding rows, and only with a policy loss mode whose loss ignores empty rows
(`ROW_PADDING_LOSS_MODES` in `losses/loss_inputs.py`): every mode of verl except `geo_mean`, which averages its
per-sequence losses over every row. With `geo_mean`, and with any other loss, the loss inputs keep their exact rows
(the engine logs this once). The loss, its gradients and the masked metrics do not change. Per-sequence diagnostics
that reduce over every loss row also count the padding rows, which are empty sequences. These bypass-mode metrics can
change:

- the sequence-level IS statistics, `rollout_corr/rollout_is_seq_*`;
- with `rollout_is=sequence`, the largest and smallest sequence IS ratio and the fractions of sequences above
  `rollout_is_threshold` and below its lower bound (by default 1 / `rollout_is_threshold`):
  `rollout_corr/rollout_is_max`, `rollout_corr/rollout_is_min`, `rollout_corr/rollout_is_ratio_fraction_high` and
  `rollout_corr/rollout_is_ratio_fraction_low`. A padding row has the sequence ratio 1, so the ratios 2 and 4, for
  example, get a minimum of 1, and both fractions count the padding rows in their denominator;
- with rejection sampling, the fraction of sequences with rejected tokens
  (`rollout_corr/rollout_rs_seq_masked_fraction`);
- with verl 0.9.1, the off-policy perplexity and chi-square metrics (`rollout_corr/*ppl*`, `rollout_corr/chi2_seq`).

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
