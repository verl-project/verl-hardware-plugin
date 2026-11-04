# Copyright 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Real-hardware check of the mori checkpoint engine.

``test_mori_checkpoint_engine.py`` runs the engine against a fake of MoRI-IO. This one moves weights between the
GPUs of this node with real MoRI-IO transfers, 2 trainer workers to the others, driven the way
``CheckpointEngineManager`` drives them, for several syncs in a row; every tensor must arrive bit-exact.

- ``xgmi``: one node, so every hop is a same-node XGMI copy.
- ``rdma``: MoRI's same-node XGMI shortcut is switched off (``MORI_DISABLE_AUTO_XGMI=1``), the trainers get a node id of
  their own and the rollout workers are split into two "nodes", so every hop goes through the NICs, with the
  topology of a multi-node job: stripes from the trainers, chains across rollout nodes, fan-out inside each node.

Skipped unless MoRI, Ray and at least 4 ROCm GPUs are present, so the CPU CI run is unaffected; the ``rdma`` case also
needs RDMA NICs. With 4 GPUs each "rollout node" of the ``rdma`` case has one GPU, so only 8 GPUs cover the fan-out
inside a node.
"""

import pytest

torch = pytest.importorskip("torch")
ray = pytest.importorskip("ray")
pytest.importorskip("mori", reason="needs MoRI-IO (pip install amd-mori)")

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.version.hip and torch.cuda.device_count() >= 4),
    reason="needs at least 4 ROCm GPUs",
)


@pytest.mark.parametrize(
    "num_chains, num_slots, intra_node",
    [(1, 2, "xgmi"), (8, 4, "xgmi"), (8, 2, "rdma")],
    ids=["xgmi-one-stripe", "xgmi-stripe-per-trainer", "rdma-two-nodes"],
)
def test_every_rollout_gets_every_tensor(num_chains, num_slots, intra_node):
    # Defined inside the test so Ray ships them by value; workers cannot import this test module by name.
    def make_weights(seed):
        g = torch.Generator().manual_seed(seed)
        ws = [("embed", torch.randn(4096, 1024, generator=g).to(torch.bfloat16))]
        dtypes = [torch.bfloat16, torch.float32, torch.float16, torch.int64]
        for i in range(64):
            n, dt = int(torch.randint(1, 1 << 20, (1,), generator=g)), dtypes[i % 4]
            t = torch.randint(-1000, 1000, (n,), generator=g) if dt == torch.int64 else torch.randn(n, generator=g)
            ws.append((f"layers.{i}.weight", t.to(dt)))
        ws.append(("mlp.fp8.weight", torch.randn(2048, 1024, generator=g).to(torch.float8_e4m3fn)))
        ws.append(("big", torch.randn(48 << 20, generator=g).to(torch.bfloat16)))  # 96 MB: spans several buckets
        return ws

    @ray.remote(num_gpus=1)
    class Worker:
        def __init__(self, node_id):
            from verl_hardware_plugin.accelerators.rocm.engines.mori_checkpoint_engine import MoriCheckpointEngine

            self.engine = MoriCheckpointEngine(
                bucket_size=64 << 20,
                num_chains=num_chains,
                num_slots=num_slots,
                intra_node=intra_node,
                node_id=node_id,
                timeout_s=120,
            )

        def prepare(self):
            return self.engine.prepare()

        def init_process_group(self, kwargs):
            self.engine.init_process_group(**kwargs)

        def finalize(self):
            self.engine.finalize()

        async def send(self, seed):
            weights = [(name, t.cuda()) for name, t in make_weights(seed)]
            await self.engine.send_weights(iter(weights), global_steps=seed)

        async def receive(self, seed):
            expected = {name: t.cuda() for name, t in make_weights(seed)}
            wrong, seen = [], set()
            async for name, t in self.engine.receive_weights(global_steps=seed):
                want = expected[name]
                same = t.dtype == want.dtype and t.shape == want.shape
                if not (same and torch.equal(t.reshape(-1).view(torch.uint8), want.reshape(-1).view(torch.uint8))):
                    wrong.append(name)
                seen.add(name)
            return wrong, sorted(set(expected) - seen)

    n_trainers = 2
    n_rollouts = torch.cuda.device_count() - n_trainers
    if intra_node == "rdma":
        env = {"MORI_DISABLE_AUTO_XGMI": "1"}
        trainer_nodes = ["trainer"] * n_trainers
        rollout_nodes = [f"rollout{i * 2 // n_rollouts}" for i in range(n_rollouts)]
    else:  # the real node id: MoRI copies between GPUs of one node over XGMI, which raises no notification
        env, trainer_nodes, rollout_nodes = {}, [None] * n_trainers, [None] * n_rollouts
    ray.init(ignore_reinit_error=True)
    try:
        options = {"runtime_env": {"env_vars": env}}
        trainers = [Worker.options(**options).remote(node) for node in trainer_nodes]
        rollouts = [Worker.options(**options).remote(node) for node in rollout_nodes]
        engine_cls = __import__(
            "verl_hardware_plugin.accelerators.rocm.engines.mori_checkpoint_engine", fromlist=["MoriCheckpointEngine"]
        ).MoriCheckpointEngine
        for seed in range(3):  # the region, queue pairs and sockets are kept across syncs
            meta = ray.get([w.prepare.remote() for w in trainers + rollouts], timeout=300)
            # the engine falls back to RDMA if it cannot create the XGMI backend: make sure each case runs its path
            assert [m["xgmi"] for m in meta] == [intra_node == "xgmi"] * len(meta), [m["xgmi"] for m in meta]
            trainer_kwargs, rollout_kwargs = engine_cls.build_topology(n_trainers, n_rollouts, meta)
            ray.get(
                [
                    w.init_process_group.remote({k: v[i] for k, v in trainer_kwargs.items()})
                    for i, w in enumerate(trainers)
                ]
                + [
                    w.init_process_group.remote({k: v[i] for k, v in rollout_kwargs.items()})
                    for i, w in enumerate(rollouts)
                ],
                timeout=300,
            )
            results = ray.get(
                [w.send.remote(seed) for w in trainers] + [w.receive.remote(seed) for w in rollouts], timeout=600
            )
            for rank, (wrong, missing) in enumerate(results[n_trainers:]):
                assert not wrong and not missing, (seed, rank, wrong[:5], missing[:5])
            ray.get([w.finalize.remote() for w in trainers + rollouts], timeout=300)
    finally:
        ray.shutdown()
