# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Native vLLM rollout using verl's HTTP server and generation lifecycle."""

import asyncio
import logging
import os
import time

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy, PlacementGroupSchedulingStrategy

from verl.single_controller.ray.base import RayResourcePool
from verl.workers.rollout.replica import RolloutMode
from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer, vLLMReplica

from ..runtime import CoreLease

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def serving_environment():
    environment = {
        "VERL_PLATFORM": "aws",
        "VERL_NEURON_PROCESS_ROLE": "rollout",
        "TORCH_DEVICE_BACKEND_AUTOLOAD": "0",
        "VLLM_PLUGINS": "neuron",
        "VLLM_NEURON_BACKEND": "neuron_native",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "VLLM_USE_V2_MODEL_RUNNER": "0",
        "NEURON_RT_MAP_HBM": "1",
        "NEURON_RT_ENABLE_P2P": "0",
        "FI_EFA_ENABLE_SHM_TRANSFER": "0",
        "NEURON_ENFORCE_NKI_LNC_CONSISTENCY": "0",
        "NEURON_CC_FLAGS": "",
    }
    for name in ("LD_LIBRARY_PATH", "NEURON_LOGICAL_NC_CONFIG", "VLLM_CACHE_ROOT", "HF_HOME", "HF_HUB_OFFLINE"):
        if name in os.environ:
            environment[name] = os.environ[name]
    return environment


class NeuronVLLMHttpServer(vLLMHttpServer):
    async def collective_rpc(self, method, timeout=None, args=(), kwargs=None):
        # verl 0.9.1 awaits this RPC but drops the TP workers' acknowledgements.
        return await self.engine.collective_rpc(method=method, timeout=timeout, args=args, kwargs=kwargs)

    async def generate(
        self,
        prompt_ids,
        sampling_params,
        request_id,
        image_data=None,
        video_data=None,
        audio_data=None,
        mm_processor_kwargs=None,
        priority=0,
        kv_transfer_params=None,
    ):
        # 0.9.1 has no preprocessing/postprocessing hooks. Resolve the same
        # defaults as its generate() before validating the actual sampling policy.
        params = dict(sampling_params)
        params.setdefault("repetition_penalty", self.config.get("repetition_penalty", 1.0))
        if self.config.full_determinism:
            params.setdefault("seed", self.replica_rank + self.config.seed)
        self._preprocess_sampling_params(params)
        wants_logprobs = bool(params.get("logprobs", False))
        if not params.get("stop"):
            params.setdefault("detokenize", False)
        output = await super().generate(
            prompt_ids,
            params,
            request_id,
            image_data=image_data,
            video_data=video_data,
            audio_data=audio_data,
            mm_processor_kwargs=mm_processor_kwargs,
            priority=priority,
            kv_transfer_params=kv_transfer_params,
        )
        self._validate_output(output, wants_logprobs)
        return output

    def _preprocess_engine_kwargs(self, engine_kwargs):
        super()._preprocess_engine_kwargs(engine_kwargs)
        visible = os.environ.pop("NEURON_RT_VISIBLE_CORES", None)
        if visible:
            os.environ["NEURON_VISIBLE_DEVICES"] = visible
        engine_kwargs.update(
            worker_cls="verl_hardware_plugin.accelerators.neuron.rollout.worker.NeuronRLWorker",
            distributed_executor_backend="mp",
            enable_sleep_mode=False,
            enable_prefix_caching=False,
            async_scheduling=False,
            compilation_config={"cudagraph_mode": "NONE"},
        )
        engine_kwargs.setdefault("additional_config", {}).setdefault("neuron_config", {})[
            "enable_structured_outputs"
        ] = True

    def _get_worker_extension_cls(self):
        return "verl_hardware_plugin.accelerators.neuron.rollout.worker_extension.NeuronWorkerExtension"

    def _preprocess_sampling_params(self, sampling_params):
        # The pinned native sampler consumes only temperature, top-k and top-p;
        # these vLLM options would otherwise affect scoring but not generation.
        defaults = {
            "repetition_penalty": 1.0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "min_p": 0.0,
            "min_tokens": 0,
        }
        unsupported = [name for name, default in defaults.items() if sampling_params.get(name, default) != default]
        unsupported.extend(
            name for name in ("logit_bias", "bad_words", "logits_processors") if sampling_params.get(name)
        )
        unsupported.extend(
            name
            for name in ("allowed_token_ids", "thinking_token_budget", "logprob_token_ids", "seed")
            if sampling_params.get(name) is not None
        )
        if unsupported:
            raise ValueError(f"Native Neuron sampling does not support {', '.join(unsupported)}")

    def _postprocess_output(self, output, final_res, sampling_params):
        self._validate_output(output, sampling_params.logprobs is not None)
        return output

    @staticmethod
    def _validate_output(output, wants_logprobs):
        if wants_logprobs and output.token_ids:
            if output.log_probs is None or len(output.log_probs) != len(output.token_ids):
                raise RuntimeError("Native rollout must return one log-probability for each sampled token")

    async def execute_checkpoint_engine(self, method, **kwargs):
        if method == "prepare":
            return {"role": "rollout", "rollout_tp": self.config.tensor_model_parallel_size}
        if method == "finalize":
            return None
        if method != "init_process_group":
            raise ValueError(f"Unsupported checkpoint operation {method}")
        spec, schedule = kwargs["spec"], kwargs["schedule"]
        self._sync_checksums = kwargs["checksums"]
        if getattr(self, "_sync_connected", False):
            if spec != self._sync_spec or schedule != self._sync_schedule:
                raise ValueError("NIXL topology and weight metadata must remain stable")
            return
        self._sync_spec = spec
        self._sync_schedule = schedule
        self._sync_bucket_bytes = kwargs["bucket_bytes"]
        rpc_spec = spec.receiver(kwargs["replica_index"])
        results = await self.collective_rpc("weight_sync_connect", timeout=spec.timeout_s, args=(rpc_spec,))
        expected = set(range(self.config.tensor_model_parallel_size))
        if {item["tp_rank"] for item in results} != expected or any(item["device"] == "cpu" for item in results):
            raise RuntimeError("Every rollout TP worker must connect a real Trainium NIXL endpoint")
        self._sync_connected = True

    async def update_weights(self, global_steps=None):
        if not self._submission_paused:
            raise RuntimeError("Generation must be paused before a NIXL update")
        if not getattr(self, "_sync_connected", False):
            raise RuntimeError("Connect NIXL before receiving weights")
        started = time.monotonic()
        results = await self.collective_rpc(
            "weight_sync_recv",
            timeout=self._sync_spec.timeout_s,
            args=(self._sync_schedule, self._sync_bucket_bytes),
        )
        if len(results) != self.config.tensor_model_parallel_size or any(count <= 0 for count in results):
            raise RuntimeError("At least one TP worker did not apply the NIXL update")
        if self._sync_checksums:
            checksums = await self.collective_rpc(
                "weight_sync_checksum", timeout=self._sync_spec.timeout_s, args=(list(self._sync_checksums),)
            )
            if len(checksums) != self.config.tensor_model_parallel_size or any(
                actual != self._sync_checksums for actual in checksums
            ):
                raise RuntimeError("NIXL replicated-parameter checksum mismatch; generation remains paused")
        await self.engine.reset_prefix_cache()
        self.global_steps = global_steps
        logger.info(
            "neuron_nixl applied policy step %s on %d TP workers; verified %d replicated parameters",
            global_steps,
            len(results),
            len(self._sync_checksums),
        )
        return {
            "timing_s/neuron_nixl/receive": time.monotonic() - started,
            "neuron_nixl/verified_replicated_parameters": len(self._sync_checksums),
        }

    async def close_weight_sync(self):
        if getattr(self, "_sync_connected", False):
            await self.collective_rpc("weight_sync_close")
            self._sync_connected = False


class NeuronVLLMReplica(vLLMReplica):
    async def init_hybrid(self, worker_group):
        raise ValueError("Trainium v1 requires trainer.v1.trainer_mode=separate_async")

    async def init_hybrid_colocated(self, worker_group, resource_pool):
        await self.init_hybrid(worker_group)

    async def init_colocated(self, resource_pool):
        await self.init_hybrid(None)

    async def init_standalone(self, resource_pool=None):
        if self.nnodes != 1 or self.config.data_parallel_size != 1 or self.config.pipeline_model_parallel_size != 1:
            raise ValueError("Each Trainium rollout replica must be a single-node TP-only engine")
        if self.config.checkpoint_engine.backend != "neuron_nixl":
            raise ValueError("Trainium rollout requires checkpoint_engine.backend=neuron_nixl")
        if self.config.enable_sleep_mode or self.config.free_cache_engine or self.config.enable_prefix_caching:
            raise ValueError("Disable rollout sleep, free_cache_engine and prefix caching on Trainium")
        self.rollout_mode = RolloutMode.STANDALONE
        self.resource_pool = resource_pool or RayResourcePool(
            process_on_nodes=[self.world_size], name_prefix=f"neuron_rollout_{self.replica_rank}{self.name_suffix}"
        )
        await self.launch_servers()

    async def launch_servers(self):
        groups = self.resource_pool.get_placement_groups(device_name="neuron")
        start = int(getattr(self.resource_pool, "start_bundle_index", 0))
        available = [(group, index) for group in groups for index in range(len(group.bundle_specs))]
        selected = available[start : start + self.world_size]
        if len(selected) != self.world_size:
            raise ValueError("Rollout resource pool does not contain enough NeuronCore bundles")
        environment = serving_environment()
        lease_class = ray.remote(CoreLease)
        self._leases = []
        self.servers = []
        try:
            for group, index in selected:
                lease = lease_class.options(
                    num_cpus=0,
                    resources={"neuron_cores": 1},
                    runtime_env={"env_vars": {**environment, "VLLM_PLUGINS": ""}},
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=group, placement_group_bundle_index=index
                    ),
                ).remote()
                self._leases.append(lease)
            allocations = await asyncio.gather(*[lease.allocation.remote() for lease in self._leases])
            nodes = {node for node, cores in allocations}
            cores = sorted(int(core) for node, core_ids in allocations for core in core_ids)
            if len(nodes) != 1 or len(set(cores)) != self.world_size:
                raise ValueError("A rollout replica requires distinct NeuronCores on one node")
            visible = ",".join(str(core) for core in cores)
            environment.update(
                NEURON_RT_VISIBLE_CORES=visible,
                NEURON_VISIBLE_DEVICES=visible,
                RAY_EXPERIMENTAL_NOSET_NEURON_RT_VISIBLE_CORES="1",
            )
            server = (
                ray.remote(NeuronVLLMHttpServer)
                .options(
                    num_cpus=1,
                    runtime_env={"env_vars": environment},
                    scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=nodes.pop(), soft=False),
                    name=f"neuron_vllm_server_{self.replica_rank}{self.name_suffix}",
                    max_concurrency=self.config.get("ray_actor_max_concurrency", 1024),
                )
                .remote(
                    config=self.config,
                    model_config=self.model_config,
                    rollout_mode=self.rollout_mode,
                    workers=[],
                    replica_rank=self.replica_rank,
                    node_rank=0,
                    gpus_per_node=self.world_size,
                    nnodes=1,
                    cuda_visible_devices=visible,
                )
            )
            self.servers = self.workers = [server]
            await server.launch_server.remote()
            host, port = await server.get_server_address.remote()
            self._server_handle = server
            self._server_address = f"[{host}]:{port}" if ":" in host else f"{host}:{port}"
        except BaseException:
            for actor in self.servers + self._leases:
                ray.kill(actor)
            raise
