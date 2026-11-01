# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Real-hardware check for the XCCL collective shim.

``tests/test_plugin_registration.py`` only checks the shim's shape against mocks. This drives it the way
verl core does -- ``create_nccl_communicator_in_ray`` across two Ray workers, each on its own XPU -- and
then runs a real collective, so a broken port hand-off, backend string or device pinning shows up here.

Skipped unless at least two XPUs and Ray are present, so the CPU CI run is unaffected.
"""

import pytest

torch = pytest.importorskip("torch")
ray = pytest.importorskip("ray")

pytestmark = pytest.mark.skipif(
    not (hasattr(torch, "xpu") and torch.xpu.is_available() and torch.xpu.device_count() >= 2),
    reason="needs at least 2 XPUs",
)


def test_shim_builds_communicator_and_all_reduces_across_two_xpus():
    # Defined inside the test so Ray ships them by value; workers cannot import this test module by name.
    @ray.remote(num_gpus=1)
    class Worker:
        def __init__(self, rank, world_size):
            self.rank = rank
            self.world_size = world_size

        def run(self):
            import torch.distributed as dist

            from verl.utils.rendezvous.ray_backend import create_nccl_communicator_in_ray

            comm = create_nccl_communicator_in_ray(self.rank, self.world_size, "xccl_shim_hw_test")
            device_tensor = torch.ones(4, device="xpu") * (self.rank + 1)
            dist.all_reduce(device_tensor)
            cpu_tensor = torch.ones(4) * (self.rank + 1)
            dist.all_reduce(cpu_tensor)
            result = (comm.rank_id(), device_tensor.cpu().tolist(), cpu_tensor.tolist())
            dist.destroy_process_group()
            return result

    world_size = 2
    ray.init(ignore_reinit_error=True)
    try:
        workers = [Worker.remote(rank, world_size) for rank in range(world_size)]
        results = ray.get([w.run.remote() for w in workers], timeout=300)
    finally:
        ray.shutdown()

    assert [r[0] for r in results] == [0, 1]
    for _, device_sum, cpu_sum in results:
        assert device_sum == [3.0] * 4
        assert cpu_sum == [3.0] * 4
