# MoRI checkpoint engine

Syncs weights from the trainer to the rollout workers with one-sided RDMA WRITE through
[MoRI-IO](https://github.com/ROCm/mori), on AMD GPUs with RDMA NICs, for disaggregated setups (separate trainer
and rollout GPUs, e.g. one-step-off or fully-async training). It implements verl's `CheckpointEngine` interface and
is registered as `mori`. Tested on MI355X with Pollara 400G NICs (one per GPU) and MoRI 1.2.4.

Code: `verl_hardware_plugin/accelerators/rocm/engines/mori_checkpoint_engine.py`, registered from
`verl_hardware_plugin/accelerators/rocm/registration.py`.

## How a sync works

- **Stripes over all NICs.** The model is split into stripes, one per trainer rank (up to `num_chains`), each sent
  by its own trainer GPU through its own NIC. All trainer ranks iterate the same weight generator and assign each
  tensor to the stripe with the fewest bytes so far, so they agree without talking.
- **XGMI fan-out inside each rollout node.** On every rollout node one GPU holds each stripe (a different GPU per
  stripe, so all NICs of the node are used): it receives the stripe from outside and copies it to the other GPUs
  of the node over XGMI. The holders of a stripe on different rollout nodes form a chain, so each rollout node
  receives one copy of the model, spread over its NICs.
- **Flow control.** Each worker registers one region of `num_slots * update_weights_bucket_megabytes` once and
  keeps it, the MoRI queue pairs and the ZMQ sockets across syncs; each stripe gets
  `update_weights_bucket_megabytes / num_chains` per slot. Credits per stripe, slot and hop allow `num_slots`
  buckets in flight. A receiver grants its slots only once it has entered the sync, so nothing is written into a
  worker before it guards its region.
- **No full-device synchronization.** A write waits for an event recorded after its packing copies; a receiver
  frees a slot once an event recorded after the consumer's reads has fired and the slot's forward writes are done.
- **Control messages** (bucket announcements, slot credits, aborts) travel as JSON over ZMQ, never pickle, so
  decoding a message cannot run code. Each worker listens on the address it publishes (its Ray node IP by default),
  not on every interface. Every message carries the random id of its topology and the sync's round; messages of
  other syncs, malformed messages and messages that do not fit the topology are dropped. The id is not
  authentication: whoever can read the control traffic can forge messages, so the control network must be trusted.

## Requirements

- MoRI-IO on every trainer and rollout node (`pip install amd-mori`); tested with 1.2.4.
- RDMA NICs reachable between the trainer and rollout nodes.
- cupy, to allocate the region outside PyTorch's caching allocator: MoRI cannot register PyTorch expandable-segment
  (VMM) memory. Without cupy the region is allocated with expandable segments switched off for that allocation.
- pyzmq (part of verl's ROCm images).
- verl 0.9.1 or newer.

## Configuration

```yaml
actor_rollout_ref:
  rollout:
    checkpoint_engine:
      backend: mori
      update_weights_bucket_megabytes: 2048
      engine_kwargs:
        mori:
          num_chains: 8   # max stripes (one per trainer rank)
          num_slots: 4    # buckets in flight per stripe and hop (default 2)
```

| `engine_kwargs.mori` | Default | Meaning |
|---|---|---|
| `num_chains` | 8 | Max stripes; the trainer world size caps it. |
| `num_slots` | 2 | Buckets in flight per stripe and hop (>= 2); costs `num_slots * bucket size` of device memory per worker. |
| `export` | `full` | `full`: stripes come from the trainer's full-tensor export (`get_per_tensor_param`). `shard`: see [Sharded export](#sharded-export). |
| `intra_node` | `xgmi` | `xgmi` copies inside a rollout node over XGMI; `rdma` keeps it on the NIC. |
| `qp_per_transfer`, `num_nics_per_transfer`, `enable_chunking` | 4, 1, True | MoRI RDMA settings. Chunking off halves the bandwidth of one transfer. |
| `host` | Ray node IP | Address for MoRI and the control socket. |
| `node_id` | Ray node id | Node identity used to group rollout workers. |
| `timeout_s` | 600 | Max wait for any single transfer or control message. |

## Sharded export

With `export: shard`, stripe r carries trainer rank r's own shards from the training engine's
`get_per_tensor_param_shard()` (bf16), so the trainer does not all-gather. Rollout workers put full tensors together
from the pieces of all stripes and split packed MoE experts into per-expert names, as the full export does; server
adapters see the same `(name, full tensor)` stream.

- Needs verl's `consumes_training_engine` hook: the trainer worker then hands its training engine to `send_weights`
  instead of a generator of full tensors. With a verl that lacks it, `export: shard` is refused.
- Only verl's FSDP training engines, whose shards are HF-named. VeOmni (fused expert stacks), Megatron (MCore-named
  TP slices) and TorchTitan are refused.
- Shards must be one flat range of the full tensor (FSDP2 `Shard(0)`); replicated tensors are sent once.
  `num_chains` must be >= the trainer world size.
- Also refused: LoRA, QAT and specs that need a to-HF conversion; use `export: full`. A refusal fails the sync on
  every worker.
- Rollout workers hold at most `num_slots * bucket size` plus the largest tensor of partly assembled tensors: a stripe
  starts a new tensor only if it fits that budget or is the next incomplete tensor in export order. A sync fails if
  any tensor is still incomplete when every stripe has ended.

```yaml
        mori:
          num_chains: 8   # >= trainer world size
          num_slots: 4
          export: shard
```

## Failures are fail-stop

MoRI 1.2.4 does not guarantee that a failed or abandoned transfer has stopped touching memory (a FAILED status can
leave work requests posted), and a GPU consumer may still read a slot when its sync is aborted. So when a worker's
engine sees its sync fail (a transfer or protocol error, a timeout, cancellation, or the receive stream closed before
it ended), it marks itself unusable before anything else and tells its peers, which then fail at once. It keeps its
MoRI engine, region and transfer statuses alive for the rest of the process and refuses every further call; recreate
the rollout and trainer workers to continue.

- A worker refuses `prepare`, `init_process_group` and a new sync while its previous sync is still running.
  `CheckpointEngineManager` raises on the first worker error without waiting for the others (`ray.get`, or
  `asyncio.gather` since verl#8542), so a retry could otherwise reach a worker that is still unwinding.
- A receiver's sync starts when `receive_weights` is called, not at its first iteration; closing the stream before
  the first tensor fails the sync too.
- A server adapter that raises while consuming the stream does not close it: verl's
  `CheckpointEngineWorker.update_weights` passes the error on and leaves the stream to garbage collection. The driver
  sees the error at once (the manager raises on the first worker error). The engine sees it only when an unfinished stream is
  finalized, and not at all if the adapter failed before its first tensor; until then that receiver stays busy and
  refuses new syncs, and its peers stop when `timeout_s` expires.

## Rollout workers that join, leave or are replaced

Each topology gets an id that tags its messages, so workers never act on another sync's messages. The engine
descriptors and sockets of peers that left are released at once. MoRI 1.2.4's `deregister_remote_engine` keeps
their RDMA endpoints, sessions and XGMI IPC mappings until the MoRI engine is destroyed, so once more than
max(16, 2 x current peers) peers have left, the next `prepare` replaces the worker's MoRI engine (same region,
re-registered).

## Results

Qwen3-30B-A3B, 8 trainer ranks (FSDP2) on one node and 8 rollout ranks on another; 8x MI355X per node, Pollara
400G, one NIC per GPU. Time of `CheckpointEngineManager.update_weights` with verl's test `MockServerAdapter` on the
rollout side, median of 9 syncs after the first:

| | seconds per sync |
|---|---|
| `nccl`, 2048 MB bucket | 1.323 |
| `mori`, 2048 MB bucket, 4 slots | **0.662** |
| `mori`, `export: shard`, 2048 MB bucket, 4 slots | **0.451** |
| trainer export alone (iterating `get_per_tensor_param()`, sending nothing) | 0.60 |
| trainer shard export alone (iterating `get_per_tensor_param_shard()`, sending nothing) | 0.30 |

The first two syncs of every configuration matched the HF checkpoint tensor by tensor on all 8 rollout workers. The
sharded export ships bf16 (61 GB for this model, what the rollout loads); the full export ships the trainer's fp32
parameters (122 GB). Earlier runs (16 runs, 5 node pairs) gave medians of 1.28-1.46 s for `nccl` and 0.65-0.83 s
for `mori`, 1.6-2.2x faster within each run. End to end (GRPO on GSM8K with a vLLM rollout, 2 nodes,
Qwen2.5-7B-Instruct, 20 steps), test accuracy went from 0.785 to 0.892 with `mori`, to 0.909 with `export: shard` and
to 0.901 with `nccl`.

## Tests

`tests/accelerators/rocm/test_mori_checkpoint_engine.py` runs on CPU with an in-process fake of MoRI-IO over real
ZMQ: a write reads its source and lands on a background thread after a random delay; RDMA writes notify the target
and same-node XGMI writes do not; as in MoRI 1.2.4, a write can report FAILED while its copy still runs, and a
deregistered peer's endpoint is kept until the engine is destroyed. The tests cover many topologies and stripe and
slot counts with tensors of every dtype (fp8 and e8m0 included) arriving bit-exact, the traffic pattern, every
failure path above, membership changes, and foreign messages on the control socket. For `export: shard`, a fake
trainer cuts each tensor along dim 0 (uneven, empty and 0-dim shards, replicated tensors, packed MoE experts) and
the tests check the memory bound and every refusal.

## MoRI 1.2.4 workarounds

- A bounded `WaitFor` on an XGMI transfer sleeps until its timeout
  ([ROCm/mori#873](https://github.com/ROCm/mori/issues/873)), so the engine polls `InProgress()` instead.
- XGMI writes raise no notification at the target, so the bucket message of an XGMI hop is sent only after the
  write completed.
