# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from datetime import timedelta

import torch
import torch.distributed as dist
import torch_mlu  # noqa: F401 -- registers ProcessGroupCNCL on torch.distributed


class CNCLCommunicator:
    """Independent CNCL group; callers control broadcast completion."""

    def __init__(
        self,
        master_address: str,
        master_port: int,
        rank: int,
        world_size: int,
        device: int,
        timeout: timedelta = timedelta(seconds=300),
    ) -> None:
        self.device = torch.device("mlu", device)
        torch.mlu.set_device(self.device)
        self._store = dist.TCPStore(
            master_address,
            master_port,
            world_size,
            rank == 0,
            timeout=timeout,
            use_libuv=False,
        )
        self._backend = dist.ProcessGroupCNCL(self._store, rank, world_size, timeout)

    def broadcast(self, tensor: torch.Tensor, src: int = 0) -> None:
        """Enqueue a broadcast without synchronizing."""
        options = dist.BroadcastOptions()
        options.rootRank = src
        options.rootTensor = 0
        self._backend.broadcast([tensor], options)

    def barrier(self) -> None:
        options = dist.BarrierOptions()
        options.device_ids = [self.device.index]
        self._backend.barrier(options).wait()

    def close(self) -> None:
        """Synchronize the device and release the communicator."""
        if self._backend is not None:
            torch.mlu.synchronize(self.device)
            self._backend.shutdown()
            self._backend = None
            self._store = None
