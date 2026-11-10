# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""tpu-sync weight-sync correctness on TPU: real TorchTitan trainer -> tpu-sync -> fake rollout workers (no vLLM).

A forward-only TorchTitan trainer on the first TPU slice sends its FSDP shards through the real
``TPUSyncCheckpointEngine``, tpu-sync controller and transport, driven by the plugin's ``update_tpu_sync_weights``.
Fake rollout replicas on the second slice run the real ``vLLMTPUSyncWorkerExtension.init_tpu_sync_on_worker``
(staging buffers, TP sharding specs, controller registration) and ``get_model_weights_stats`` in single-chip
processes, without vLLM. The trainer does not train, so after every sync each rollout rank's received tensors must
equal its TP slice of the HF checkpoint, bit for bit. The engine's own checks (``verify_parity=all``) run as well.

Not a pytest module (the plugin CI runs on CPU only). Submit it as a Ray job to a cluster with two TPU slices
(e.g. 2 x v6e-8), with verl and the plugin importable:

    python3 tpu_sync_parity_on_tpu.py

Environment: ``TPU_SYNC_TEST_MODEL_PATH`` (default ``/data/jialei/assets/hf/Qwen3-0.6B``).
Ported from jialei777/verl-upstream ``tests/special_tpu/raiden_sync_test.py`` (#84).
"""

import asyncio
import json
import logging
import os
import sys
import time
import traceback
from types import SimpleNamespace

import ray
import torch

import verl_hardware_plugin  # noqa: F401  (registers the TPU platform, engines and the tpu-sync backend)
from verl.checkpoint_engine import CheckpointEngineManager
from verl.single_controller.base.decorator import Dispatch, register
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.workers.config import (
    CheckpointEngineConfig,
    HFModelConfig,
    TorchtitanEngineConfig,
    TorchtitanOptimizerConfig,
    TrainingWorkerConfig,
)
from verl.workers.engine_workers import TrainingWorker
from verl_hardware_plugin.accelerators.tpu.weight_sync import tpu_sync_checkpoint_engine as tpu_sync_engine
from verl_hardware_plugin.accelerators.tpu.weight_sync.tpu_sync_parity import ExactParityCheck, report_exact_parity

MODEL_PATH = os.environ.get("TPU_SYNC_TEST_MODEL_PATH", "/data/jialei/assets/hf/Qwen3-0.6B")
NUM_SYNCS = 3
# (num_replicas, tp_size, release_buffers_after_sync)
CASES = [(1, 4, True), (2, 2, True), (1, 4, False)]
ROW_PARALLEL_SUFFIXES = (".o_proj.weight", ".down_proj.weight")
KV_PROJ_SUFFIXES = (".self_attn.k_proj.weight", ".self_attn.v_proj.weight")
TPU_SLICE_RESOURCE_PREFIX = "tpu-group-"


def log(msg: str) -> None:
    print(f"[tpu-sync test] {msg}", flush=True)


# ---------------------------------------------------------------------------------------------
# Trainer: forward-only TorchTitan engine + TPUSyncCheckpointEngine
# ---------------------------------------------------------------------------------------------


class TPUSyncTrainerWorker(TrainingWorker):
    """TrainingWorker that sends its weights with ``TPUSyncCheckpointEngine``, as ``ActorRolloutRefWorker`` does."""

    def __init__(self, config: TrainingWorkerConfig) -> None:
        super().__init__(config)
        self.checkpoint_engine = None

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def reset_tpu_sync(self, **engine_kwargs):
        """(Re)create the tpu-sync engine, so that every case registers with its own controller."""
        if self.checkpoint_engine is not None:
            self.checkpoint_engine.finalize()
        self.checkpoint_engine = tpu_sync_engine.TPUSyncCheckpointEngine(
            bucket_size=0, is_master=torch.distributed.get_rank() == 0, **engine_kwargs
        )

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    async def update_weights(self, global_steps: int = None, mode: str = "auto"):
        # Same calls as ActorRolloutRefWorker.update_weights for a disaggregated checkpoint engine.
        per_tensor_param, _ = self.engine.get_per_tensor_param()
        return await self.checkpoint_engine.send_weights(per_tensor_param, global_steps=global_steps) or {}

    @register(dispatch_mode=Dispatch.DP_COMPUTE, blocking=False)
    def execute_checkpoint_engine(self, method: str, *args, **kwargs):
        return getattr(self.checkpoint_engine, method)(*args, **kwargs)


# ---------------------------------------------------------------------------------------------
# Fake rollout worker: one single-chip process per TP rank, running the real tpu-sync rollout code
# ---------------------------------------------------------------------------------------------


def _pin_single_chip_env() -> None:
    """Restrict this process to the one TPU chip Ray assigned to it."""
    chip_ids = ray.get_runtime_context().get_accelerator_ids().get("TPU", [])
    assert len(chip_ids) == 1, f"expected exactly one assigned TPU chip, got {chip_ids}"
    port = 8471 + int(chip_ids[0])
    # torch_tpu starts in single-device mode without WORLD_SIZE (as a TP=1 vLLM worker does).
    for var in ("WORLD_SIZE", "RANK", "LOCAL_RANK", "LOCAL_WORLD_SIZE"):
        os.environ.pop(var, None)
    os.environ.update(
        {
            "TPU_VISIBLE_CHIPS": str(chip_ids[0]),
            "TPU_VISIBLE_DEVICES": str(chip_ids[0]),
            "TPU_CHIPS_PER_HOST_BOUNDS": "1,1,1",
            "TPU_CHIPS_PER_PROCESS_BOUNDS": "1,1,1",
            "TPU_HOST_BOUNDS": "1,1,1",
            "TPU_PROCESS_BOUNDS": "1,1,1",
            "TPU_WORKER_ID": "0",
            "TPU_WORKER_HOSTNAMES": "localhost",
            "CLOUD_TPU_TASK_ID": "0",
            "TORCH_TPU_TOPOLOGY": "1,1,1",
            "TORCH_TPU_SLICEBUILDER_ADDRESSES": f"localhost:{port}",
            "TPU_PROCESS_ADDRESSES": f"localhost:{port}",
            "TPU_PROCESS_PORT": str(port),
        }
    )


def expected_tp_slice(name: str, full: torch.Tensor, tp_size: int, tp_rank: int, num_kv_heads: int) -> torch.Tensor:
    """The slice of an HF tensor that vLLM's TP rank ``tp_rank`` holds, i.e. what tpu-sync must deliver to it."""
    if full.dim() != 2:
        return full
    if name.endswith(ROW_PARALLEL_SUFFIXES):
        return full.chunk(tp_size, dim=1)[tp_rank]
    if num_kv_heads < tp_size and name.endswith(KV_PROJ_SUFFIXES):
        return full.chunk(num_kv_heads, dim=0)[tp_rank // (tp_size // num_kv_heads)]
    if full.shape[0] % tp_size == 0:
        return full.chunk(tp_size, dim=0)[tp_rank]
    return full


@ray.remote(num_cpus=1)
class TPUSyncRolloutWorker:
    """Stands in for one vLLM TP worker.

    ``init_tpu_sync_on_worker`` and ``get_model_weights_stats`` run the real ``vLLMTPUSyncWorkerExtension`` code on
    this object. ``install_tpu_sync_weights`` replaces only the copy into vLLM's parameters: it runs the H2D (unless
    ``h2d_tpu_sync_weights`` already did) and compares the received staging tensors bit for bit with this rank's
    slice of the checkpoint.
    """

    def __init__(self, tp_rank: int, tp_size: int, model_path: str) -> None:
        _pin_single_chip_env()  # before anything initializes the TPU runtime
        import torch_tpu  # noqa: F401
        from transformers import AutoConfig

        hf_config = AutoConfig.from_pretrained(model_path)
        self.rank = tp_rank
        self.tp_size = tp_size
        self.model_path = model_path
        self.num_kv_heads = hf_config.num_key_value_heads
        self.tie_word_embeddings = bool(getattr(hf_config, "tie_word_embeddings", False))
        # What vLLMTPUSyncWorkerExtension reads from the vLLM worker. An empty model has no direct-install layout,
        # so the dense path's sharding rules apply.
        self.model_runner = SimpleNamespace(model=torch.nn.Module())
        self.vllm_config = SimpleNamespace(
            parallel_config=SimpleNamespace(tensor_parallel_size=tp_size),
            model_config=SimpleNamespace(get_total_num_kv_heads=lambda: self.num_kv_heads, hf_text_config=hf_config),
        )
        self._expected = None
        self.last_exact = None

    @staticmethod
    def _extension():
        from verl_hardware_plugin.accelerators.tpu.rollout.tpu_sync_worker_extension import (
            vLLMTPUSyncWorkerExtension,
        )

        return vLLMTPUSyncWorkerExtension

    def _get_vllm_model(self):
        return self._extension()._get_vllm_model(self)

    def init_tpu_sync_on_worker(self, **kwargs) -> bool:
        return self._extension().init_tpu_sync_on_worker(self, **kwargs)

    def get_model_weights_stats(self) -> dict:
        return self._extension().get_model_weights_stats(self)

    def _expected_tensors(self) -> dict:
        """This rank's slice of every checkpoint tensor (``lm_head`` falls back to the embedding when tied)."""
        if self._expected is None:
            from safetensors import safe_open

            index = os.path.join(self.model_path, "model.safetensors.index.json")
            if os.path.exists(index):
                with open(index) as f:
                    files = sorted(set(json.load(f)["weight_map"].values()))
            else:
                files = ["model.safetensors"]
            full = {}
            for file in files:
                with safe_open(os.path.join(self.model_path, file), framework="pt") as f:
                    for name in f.keys():
                        full[name] = f.get_tensor(name)
            if self.tie_word_embeddings and "lm_head.weight" not in full and "model.embed_tokens.weight" in full:
                full["lm_head.weight"] = full["model.embed_tokens.weight"]
            self._expected = {
                name: expected_tp_slice(name, t, self.tp_size, self.rank, self.num_kv_heads) for name, t in full.items()
            }
        return self._expected

    def _h2d(self) -> float:
        t0 = time.perf_counter()
        self._tpu_sync_ws.h2d()
        tpu_sync_engine.tpu_synchronize(strict=True)
        return time.perf_counter() - t0

    def h2d_tpu_sync_weights(self) -> float:
        return self._h2d()

    def install_tpu_sync_weights(self, exact_parity: bool = False, skip_h2d: bool = False) -> dict:
        t0 = time.perf_counter()
        t_h2d = 0.0 if skip_h2d else self._h2d()
        expected = self._expected_tensors()
        parity = ExactParityCheck()
        for name, received in self._tpu_sync_staging.items():
            if name in expected:
                parity.check(name, expected[name], received.cpu())
        received_names = set(self._tpu_sync_staging)
        # Every received tensor must exist in the checkpoint; a tied lm_head may or may not be sent.
        unresolved = sorted(received_names - set(expected))
        missing = set(expected) - received_names - ({"lm_head.weight"} if self.tie_word_embeddings else set())
        unresolved += sorted(missing)
        self.last_exact = parity.result(self.rank, unresolved)
        return {"total": time.perf_counter() - t0, "h2d": t_h2d, "exact_parity": self.last_exact}

    def last_exact_result(self) -> dict:
        return {"exact_parity": self.last_exact}


class _AsyncRemote:
    """``.remote(**kwargs)`` returning an awaitable, like a Ray actor method."""

    def __init__(self, fn):
        self._fn = fn

    def remote(self, *args, **kwargs):
        return self._fn(*args, **kwargs)


class FakeTPUSyncReplica:
    """The parts of a ``RolloutReplica`` that ``CheckpointEngineManager`` / ``update_tpu_sync_weights`` use."""

    def __init__(self, workers: list, replica_rank: int) -> None:
        self.workers = workers
        self.world_size = len(workers)
        self.replica_rank = replica_rank
        self.global_steps = None
        self.server_handle = SimpleNamespace(
            collective_rpc=_AsyncRemote(self._collective_rpc), set_global_steps=_AsyncRemote(self._set_global_steps)
        )

    async def _collective_rpc(self, method: str, timeout=None, args=(), kwargs: dict = None):
        return await asyncio.gather(*[getattr(w, method).remote(*args, **(kwargs or {})) for w in self.workers])

    async def _set_global_steps(self, global_steps):
        self.global_steps = global_steps

    async def abort_all_requests(self, reject_request: bool = False):
        return {"aborted_count": 0, "request_ids": []}

    async def resume_generation(self):
        pass


# ---------------------------------------------------------------------------------------------
# Test driver
# ---------------------------------------------------------------------------------------------


def discover_tpu_slices() -> list[SimpleNamespace]:
    """The cluster's TPU slices (KubeRay ``tpu-group-<n>`` resources): name, host node ids and chips per host."""
    slices: dict[str, SimpleNamespace] = {}
    for node in ray.nodes():
        if not node.get("Alive"):
            continue
        resources = node.get("Resources", {})
        names = [r for r in resources if r.startswith(TPU_SLICE_RESOURCE_PREFIX)]
        chips = int(resources.get("TPU", 0))
        if not names or not chips:
            continue
        s = slices.setdefault(names[0], SimpleNamespace(name=names[0], hosts=[], chips_per_host=chips))
        s.hosts.append(node["NodeID"])
    out = sorted(slices.values(), key=lambda s: int(s.name.removeprefix(TPU_SLICE_RESOURCE_PREFIX) or 0))
    for s in out:
        s.hosts.sort()
        s.chips = s.chips_per_host * len(s.hosts)
    return out


def create_trainer(trainer_slice) -> RayWorkerGroup:
    model_config = HFModelConfig(path=MODEL_PATH, use_remove_padding=True)
    engine_config = TorchtitanEngineConfig(
        forward_only=True,
        # fp32 weights, as a training actor holds: the export then sends fresh bf16 copies. A forward-only engine
        # defaults to bf16 weights, whose export would alias the parameters' own storage.
        dtype="float32",
        data_parallel_shard_size=trainer_slice.chips,
        use_torch_compile=False,
        attn_type="varlen",
        activation_checkpoint="none",
        engine_kwargs={"tpu": {"use_simple_fsdp": True}},  # as the TPU GRPO recipes
    )
    # forward_only builds no optimizer, but the engine still reads the optimizer fields.
    optimizer_config = TorchtitanOptimizerConfig(lr=1e-6, total_training_steps=1)
    config = TrainingWorkerConfig(
        model_type="language_model",
        model_config=model_config,
        engine_config=engine_config,
        optimizer_config=optimizer_config,
    )
    pool = RayResourcePool(
        process_on_nodes=[trainer_slice.chips_per_host] * len(trainer_slice.hosts),
        use_gpu=True,
        name_prefix="tpu_sync_test_trainer",
        accelerator_type=trainer_slice.name,
    )
    wg = RayWorkerGroup(
        resource_pool=pool,
        ray_cls_with_init=RayClassWithInitArgs(cls=ray.remote(TPUSyncTrainerWorker), config=config),
        device_name="tpu",
    )
    wg.reset()  # load the HF checkpoint into the TorchTitan engine
    return wg


def create_replicas(rollout_slice, num_replicas: int, tp_size: int) -> list:
    assert tp_size <= rollout_slice.chips_per_host, f"TP{tp_size} > {rollout_slice.chips_per_host} chips per host"
    assert num_replicas * tp_size <= rollout_slice.chips, f"{num_replicas}x TP{tp_size} > {rollout_slice.chips} chips"
    replicas = []
    for i in range(num_replicas):
        node_id = rollout_slice.hosts[i % len(rollout_slice.hosts)]
        strategy = ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(node_id=node_id, soft=False)
        workers = [
            TPUSyncRolloutWorker.options(resources={"TPU": 1}, scheduling_strategy=strategy).remote(
                r, tp_size, MODEL_PATH
            )
            for r in range(tp_size)
        ]
        replicas.append(FakeTPUSyncReplica(workers, replica_rank=i))
    return replicas


def run_case(trainer_wg, rollout_slice, num_replicas: int, tp_size: int, release_buffers: bool) -> None:
    engine_kwargs = {"verify_parity": "all", "release_buffers_after_sync": release_buffers}
    trainer_wg.reset_tpu_sync(**engine_kwargs)
    replicas = create_replicas(rollout_slice, num_replicas, tp_size)
    config = CheckpointEngineConfig(backend="tpu-sync", engine_kwargs={"tpu-sync": engine_kwargs})
    try:
        manager = CheckpointEngineManager(config=config, actor_wg=trainer_wg, replicas=replicas)
        for step in range(NUM_SYNCS):
            # Raises TPUSyncParityError on a mismatch: the engine's exact check at step 0, its norm check after.
            metrics = manager.update_weights(global_steps=step)
            assert metrics["timing_s/tpu-sync/total_sync"] > 0, metrics
            assert all(r.global_steps == step for r in replicas), [r.global_steps for r in replicas]
            # Every sync, every rollout rank of every replica: received tensors == checkpoint slice, bit for bit.
            results = [ray.get([w.last_exact_result.remote() for w in r.workers]) for r in replicas]
            labels = [""] if len(replicas) == 1 else [f" | replica {r.replica_rank}" for r in replicas]
            report_exact_parity(step, results, labels)
            log(
                f"{num_replicas}xTP{tp_size} release={release_buffers} step {step}: bit-exact, "
                f"total_sync={metrics['timing_s/tpu-sync/total_sync']:.3f}s "
                f"p2p={metrics['timing_s/tpu-sync/p2p_transfer']:.3f}s"
            )
    finally:
        for r in replicas:
            for w in r.workers:
                ray.kill(w)


def main() -> int:
    from verl.plugin.platform import get_platform

    # Show the engine's [TPU-SYNC PARITY ...] / [TPU-SYNC TELEMETRY ...] lines in the job log.
    logging.basicConfig(format="%(levelname)s:%(asctime)s:%(name)s: %(message)s")
    for name in (tpu_sync_engine.__name__, "verl_hardware_plugin.accelerators.tpu.weight_sync.tpu_sync_parity"):
        logging.getLogger(name).setLevel(logging.INFO)
    ray.init(**get_platform().get_ray_init_kwargs())
    slices = discover_tpu_slices()
    log(f"TPU slices: {[(s.name, len(s.hosts), s.chips_per_host) for s in slices]}")
    if len(slices) < 2:
        log(f"FAILED: needs two TPU slices (trainer + rollout), found {len(slices)}")
        return 1
    trainer_slice, rollout_slice = slices[0], slices[1]
    t0 = time.perf_counter()
    trainer_wg = create_trainer(trainer_slice)
    log(f"trainer ready on {trainer_slice.name} ({trainer_slice.chips} chips) in {time.perf_counter() - t0:.1f}s")

    failed = []
    for num_replicas, tp_size, release_buffers in CASES:
        name = f"{num_replicas}xTP{tp_size}-release={release_buffers}"
        try:
            run_case(trainer_wg, rollout_slice, num_replicas, tp_size, release_buffers)
            log(f"case {name}: PASSED")
        except Exception:
            traceback.print_exc()
            log(f"case {name}: FAILED")
            failed.append(name)
    if failed:
        log(f"FAILED: {len(failed)}/{len(CASES)} cases: {failed}")
        return 1
    log(f"PASSED: {len(CASES)}/{len(CASES)} cases, {NUM_SYNCS} syncs each")
    return 0


if __name__ == "__main__":
    sys.exit(main())
