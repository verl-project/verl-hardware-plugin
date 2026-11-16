# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""NIXL-only checkpoint bridge for native TorchTitan and native vLLM-Neuron."""

import json
import logging
import os
import socket
import time
from datetime import timedelta

import torch

from verl.checkpoint_engine.base import CheckpointEngine, CheckpointEngineRegistry

from ..runtime import physical_core
from .nixl_transport import NixlEndpoint, NixlTopology, control_store, synchronize
from .streaming import iter_weight_buckets, validate_box_coverage
from .weight_layout import pack_box, plan_box, plan_buckets, slot_numel_for

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


@CheckpointEngineRegistry.register("neuron_nixl")
class NeuronNixlCheckpointEngine(CheckpointEngine):
    def __init__(self, bucket_size, is_master=False, timeout_s=300, verify_weights=True, **kwargs):
        from ..engines.torchtitan_neuron import actor_engine

        self.engine = actor_engine()
        if self.engine.trainer.device.type != "neuron":
            raise ValueError("neuron_nixl requires real Trainium trainer tensors, not a CPU/Gloo fallback")
        self.bucket_size = int(bucket_size)
        self.timeout_s = float(timeout_s)
        self.verify_weights = bool(verify_weights)
        self.connected = False
        self.schedule = self.engine.weight_sync_metadata()
        self.rank = torch.distributed.get_rank()
        self.world_size = torch.distributed.get_world_size()
        self.device = self.engine.trainer.device
        self.core = physical_core(os.environ["NEURON_RT_VISIBLE_CORES"], int(os.environ.get("LOCAL_RANK", "0")))
        self.endpoints, self.inventories = {}, {}
        self.export_group = torch.distributed.new_group(backend="gloo", timeout=timedelta(seconds=self.timeout_s))

    def prepare(self):
        import ray

        metadata = {"role": "actor", "schedule": self.schedule, "bucket_size": self.bucket_size}
        checksums = {}
        if self.verify_weights:
            from torch.distributed.tensor import DTensor

            parameters = self.engine.hf_parameters()
            for name in (
                "model.norm.weight",
                "model.layers.0.input_layernorm.weight",
                "model.layers.0.post_attention_layernorm.weight",
            ):
                parameter = parameters[name]
                if isinstance(parameter, DTensor):
                    parameter = parameter.full_tensor()
                host = parameter.detach().to(torch.bfloat16).to("cpu").to(torch.float64)
                checksums[name] = [host.sum().item(), host.abs().sum().item()]
        metadata["checksums"] = checksums
        if self.rank == 0:
            if not hasattr(self, "port"):
                with socket.socket() as listener:
                    listener.bind(("127.0.0.1", 0))
                    self.port = listener.getsockname()[1]
            metadata.update(host=ray.util.get_node_ip_address(), port=self.port, timeout_s=self.timeout_s)
        return metadata

    @classmethod
    def build_topology(cls, actor_wg_world_size, rollout_world_size, metadata):
        actors, replicas = metadata[:actor_wg_world_size], metadata[actor_wg_world_size:]
        master = actors[0]
        if len(replicas) != rollout_world_size or not replicas:
            raise ValueError("NIXL requires at least one rollout replica")
        if any(item["schedule"] != master["schedule"] for item in actors):
            raise ValueError("Trainer ranks disagree on weight metadata")
        if any(item["bucket_size"] != master["bucket_size"] for item in actors):
            raise ValueError("Trainer ranks disagree on the NIXL bucket size")
        degrees = {item["rollout_tp"] for item in replicas}
        if len(degrees) != 1:
            raise ValueError("All rollout replicas must use the same TP degree")
        spec = NixlTopology(
            store_host=master["host"],
            store_port=master["port"],
            rollout_tp=degrees.pop(),
            timeout_s=master["timeout_s"],
            replicas=rollout_world_size,
        )
        actor_args = {"spec": [spec] * actor_wg_world_size}
        rollout_args = {
            "spec": [spec] * rollout_world_size,
            "schedule": [master["schedule"]] * rollout_world_size,
            "bucket_bytes": [master["bucket_size"]] * rollout_world_size,
            "replica_index": list(range(rollout_world_size)),
            "checksums": [master["checksums"]] * rollout_world_size,
        }
        return actor_args, rollout_args

    def init_process_group(self, spec):
        if self.connected:
            if self.spec != spec:
                raise ValueError("NIXL topology cannot change during this run")
            return
        self.spec = spec
        self.store = control_store(spec, master=self.rank == 0)
        for replica in range(spec.replicas):
            for rank in range(spec.rollout_tp):
                if (replica * spec.rollout_tp + rank) % self.world_size != self.rank:
                    continue
                prefix = f"replica-{replica}-tp-{rank}"
                endpoint = NixlEndpoint(
                    self.store, prefix, self.device, self.core, sender=True, timeout_s=spec.timeout_s
                )
                self.endpoints[replica, rank] = endpoint
                self.inventories[replica, rank] = {
                    name: tuple(shape) for name, shape in json.loads(endpoint.store.get("inventory")).items()
                }
        self.connected = True

    async def send_weights(self, weights, global_steps=None):
        if not self.connected:
            raise RuntimeError("Connect both NIXL pools before sending weights")
        started = time.monotonic()
        buckets = plan_buckets(self.schedule, bucket_bytes=self.bucket_size)
        shapes = {name: shape for name, shape, dtype in self.schedule}
        plans = {}
        for key, endpoint in self.endpoints.items():
            plans[key] = [
                plan_box(bucket, shapes, self.inventories[key], key[1], self.spec.rollout_tp) for bucket in buckets
            ]
            validate_box_coverage(plans[key], self.inventories[key])
            endpoint.register(slot_numel_for(plans[key]))
        torch.distributed.barrier(group=self.export_group)
        for index, buffer in enumerate(iter_weight_buckets(weights, buckets, self.device)):
            if self.rank == 0:
                logger.info("neuron_nixl completing export bucket %d/%d", index + 1, len(buckets))
            synchronize()
            torch.distributed.barrier(group=self.export_group)
            for key, endpoint in self.endpoints.items():
                plan = plans[key][index]
                if plan.numel:
                    pack_box(endpoint.buffer, buffer, buckets[index], plan)
                    endpoint.push(plan.numel)
            torch.distributed.barrier(group=self.export_group)
            if self.rank == 0:
                logger.info(
                    "neuron_nixl delivered bucket %d/%d to all %d receivers",
                    index + 1,
                    len(buckets),
                    self.spec.replicas * self.spec.rollout_tp,
                )
        logger.info(
            "neuron_nixl rank %d exported %d HF buckets at policy step %s for %d TP receivers",
            self.rank,
            len(buckets),
            global_steps,
            len(self.endpoints),
        )
        return {"timing_s/neuron_nixl/send": time.monotonic() - started} if self.rank == 0 else {}

    async def receive_weights(self, global_steps=None):
        raise RuntimeError("NIXL receivers run inside native vLLM workers")

    def finalize(self):
        return None
