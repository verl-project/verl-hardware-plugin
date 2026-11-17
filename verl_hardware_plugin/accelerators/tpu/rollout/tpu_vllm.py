# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""vLLM rollout replica and HTTP server for Google TPU.

Upstream verl builds the vLLM rollout through ``RolloutReplicaRegistry``. On TPU the plugin
registers a ``vllm`` loader that returns :class:`TPUvLLMReplica`. It keeps upstream's HTTP server
and its multi-node layout, and changes only what TPU needs:

* One server actor per host of the replica, as upstream does on GPU: node 0 runs the engine and
  the HTTP API, the other hosts run vLLM headless. Each server holds no chips; vLLM's ``mp``
  executor starts one worker per local chip (or TensorCore on TPU 7x), and vllm-torchtpu's ``mp``
  multi-host backend (``TPU_MULTIHOST_BACKEND=mp``) joins the hosts of the replica into one TPU
  mesh through a TCPStore rendezvous at the replica's master address.
* ``enable_sleep_mode=False``: vllm-torchtpu has no sleep mode.
* With ``checkpoint_engine.backend=tpu-sync``, vLLM's workers load the tpu-sync worker extension
  (``tpu_sync_worker_extension.py``), which receives the trainer's weights over the network.

The same logic previously lived behind ``get_resource_name() == "TPU"`` branches in verl's
``vllm_async_server.py`` and ``replica.py``.
"""

import logging

import ray

from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer, vLLMReplica
from verl_hardware_plugin.accelerators.tpu.platform_tpu import ray_tpu_accelerator_name, single_host_tpu_topology
from verl_hardware_plugin.accelerators.tpu.profilers.tpu_vllm_timeline import vLLMServerTimelineMixin
from verl_hardware_plugin.accelerators.tpu.rollout.tpu_vllm_patches import patch_vllm_for_tpu
from verl_hardware_plugin.accelerators.tpu.weight_sync.weight_sync_backends import TPU_SYNC_BACKEND, canonical_backend

logger = logging.getLogger(__name__)

# vLLM imports the worker extension by name in its worker processes.
TPU_SYNC_WORKER_EXTENSION_CLS = (
    "verl_hardware_plugin.accelerators.tpu.rollout.tpu_sync_worker_extension.vLLMTPUSyncWorkerExtension"
)


class TPUvLLMHttpServer(vLLMServerTimelineMixin, vLLMHttpServer):
    """``vLLMHttpServer`` with the TPU engine arguments."""

    def __init__(self, *args, **kwargs):
        # This actor builds the vLLM engine and starts its EngineCore process, so patch vLLM before
        # anything else. The patches carry themselves into the EngineCore and vLLM's workers.
        patch_vllm_for_tpu()
        super().__init__(*args, **kwargs)

    def _preprocess_engine_kwargs(self, engine_kwargs: dict) -> None:
        super()._preprocess_engine_kwargs(engine_kwargs)
        # engine_kwargs is merged last into the vLLM CLI args, so this wins over the default.
        engine_kwargs["enable_sleep_mode"] = False

    def _get_worker_extension_cls(self) -> str:
        # The tpu-sync checkpoint engine transfers weights straight into vLLM's workers, which need the
        # methods of the tpu-sync worker extension. Other backends keep upstream's extension.
        checkpoint_engine = getattr(getattr(self, "config", None), "checkpoint_engine", None)
        if canonical_backend(getattr(checkpoint_engine, "backend", None)) == TPU_SYNC_BACKEND:
            return TPU_SYNC_WORKER_EXTENSION_CLS
        return super()._get_worker_extension_cls()


class TPUvLLMReplica(vLLMReplica):
    """vLLM rollout replica for TPU: one server actor per host, none of which holds a chip."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.server_class = ray.remote(TPUvLLMHttpServer)

    def _get_server_env_vars(self, node_ids: list[str]) -> dict[str, str]:
        env_vars = {
            **super()._get_server_env_vars(node_ids),
            # vllm-torchtpu's default; set so that a job-level value cannot switch the replica to the Ray backend.
            "TPU_MULTIHOST_BACKEND": "mp",
        }
        if self.nnodes == 1:
            # A replica on one host (e.g. one of several replicas on a multi-host rollout slice) is its own mesh; it
            # must not inherit the slice's TORCH_TPU_TOPOLOGY. Multi-host replicas get theirs from vllm-torchtpu's
            # rendezvous. The TPU type comes from the replica's host, not from this process's environment, which
            # may be a CPU Ray head without TPU_ACCELERATOR_TYPE. Sizes without a known mesh (e.g. TP=2 / TP=4 on TPU
            # 7x) keep the inherited environment, as before.
            chips_per_host = self.gpus_per_replica_node
            pod_type = ray_tpu_accelerator_name(node_ids=list(node_ids))
            topology = single_host_tpu_topology(chips_per_host, pod_type=pod_type)
            if topology:
                env_vars["TORCH_TPU_TOPOLOGY"] = topology
            else:
                logger.warning(
                    "No single-host TPU mesh of %d chips (TPU type %r); the rollout replica keeps the inherited "
                    "TORCH_TPU_TOPOLOGY",
                    chips_per_host,
                    pod_type,
                )
        return env_vars

    async def launch_servers(self):
        if self.config.data_parallel_size > 1:
            raise NotImplementedError(
                "actor_rollout_ref.rollout.data_parallel_size="
                f"{self.config.data_parallel_size} is not supported on TPU. The TPU "
                "distributed runtime requires each compiled program to span the whole "
                "TPU mesh, but a single DP group only spans "
                f"tensor_model_parallel_size={self.config.tensor_model_parallel_size} "
                "chips. Set actor_rollout_ref.rollout.data_parallel_size=1 instead: verl "
                "then creates one rollout replica per tensor_model_parallel_size chips, "
                "which is equivalent to engine-internal data parallelism."
            )
        await super().launch_servers()
