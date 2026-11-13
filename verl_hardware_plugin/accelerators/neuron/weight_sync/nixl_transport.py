# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Direct NIXL/LIBFABRIC HBM transport; TCPStore carries only control metadata."""

import ctypes
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path

import torch


@dataclass(frozen=True)
class NixlTopology:
    store_host: str
    store_port: int
    rollout_tp: int
    replicas: int
    timeout_s: float

    def receiver(self, replica):
        if not 0 <= replica < self.replicas:
            raise ValueError("Invalid rollout replica index")
        return {**asdict(self), "replica": replica}


def control_store(spec, *, master=False):
    return torch.distributed.TCPStore(
        spec.store_host,
        spec.store_port,
        world_size=None,
        is_master=master,
        timeout=timedelta(seconds=spec.timeout_s),
        wait_for_workers=False,
    )


def import_nixl(*, lite=False):
    """Load NIXL only after the native runtime, preserving protobuf symbol ordering."""
    paths = [
        Path(sys.prefix) / "lib/nixl-cuda-stubs/libcuda.so.1",
        Path(sys.prefix) / "lib/nixl-cuda-stubs/libcudart.so.12",
        Path("/opt/amazon/efa/lib/libfabric.so.1"),
    ]
    for path in paths:
        if path.exists():
            ctypes.CDLL(str(path))
    if lite:
        if "torch_neuronx" in sys.modules:
            raise RuntimeError("A lite rollout worker must not load the training runtime")
    else:
        import torch_neuronx._C as native

        ctypes.CDLL(native.__file__, mode=ctypes.RTLD_GLOBAL)
    from nixl._api import nixl_agent, nixl_agent_config

    return nixl_agent, nixl_agent_config


def synchronize(*, lite=False):
    """Drain the assigned native device or the isolated lite worker's queue."""
    if lite:
        from libtorch_neuronx_lite._compiler import _C

        _C._neuron_synchronize(-1)
    else:
        torch.neuron.synchronize(torch.neuron.current_device())


def efa_interface(core):
    product = Path("/sys/devices/virtual/dmi/id/product_name").read_text().strip()
    if not product.startswith("trn2"):
        raise RuntimeError("Automatic NIXL NIC selection requires Trainium2; set FI_EFA_IFACE explicitly")
    addresses = (
        "c9",
        "b4",
        "b3",
        "ca",
        "6c",
        "57",
        "56",
        "6d",
        "98",
        "83",
        "82",
        "99",
        "f5",
        "e0",
        "df",
        "f6",
    )
    device = core * int(os.environ.get("NEURON_LOGICAL_NC_CONFIG", "2")) // 8
    node = Path(f"/dev/neuron{device}")
    if not node.is_char_device():
        raise RuntimeError(f"No Neuron character device for core {core}")
    host_device = os.minor(node.stat().st_rdev)
    if not 0 <= host_device < len(addresses):
        raise RuntimeError(f"No Trainium2 EFA mapping for device {host_device}")
    interfaces = sorted(Path(f"/sys/bus/pci/devices/0000:{addresses[host_device]}:00.0/infiniband").iterdir())
    if not interfaces:
        raise RuntimeError(f"No EFA interface for core {core}")
    return interfaces[0].name


def slot_address(buffer, *, lite=False):
    """Reject suballocations: registration must refer to this buffer's whole HBM region."""
    if buffer.device.type != "neuron" or buffer.storage_offset() != 0:
        raise ValueError("NIXL slots must own a Neuron allocation, not CPU storage or a view")
    synchronize(lite=lite)
    expected = buffer.numel() * buffer.element_size()
    if lite:
        address, size = buffer.data_ptr(), buffer.untyped_storage().nbytes()
    else:
        from torch_neuronx import _C

        capsule = _C._get_nrt_tensor_capsule(buffer)
        get_pointer = ctypes.pythonapi.PyCapsule_GetPointer
        get_pointer.restype = ctypes.c_void_p
        get_pointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
        handle = get_pointer(capsule, b"nrt_tensor_t")
        runtime = ctypes.CDLL("libnrt.so.1")
        runtime.nrt_tensor_get_va.restype = ctypes.c_void_p
        runtime.nrt_tensor_get_va.argtypes = [ctypes.c_void_p]
        runtime.nrt_tensor_get_size.restype = ctypes.c_size_t
        runtime.nrt_tensor_get_size.argtypes = [ctypes.c_void_p]
        address, size = runtime.nrt_tensor_get_va(handle), runtime.nrt_tensor_get_size(handle)
    if not address or size != expected:
        raise RuntimeError(f"NIXL slot does not own its entire HBM allocation: expected {expected}, got {size}")
    return int(address), expected


class NixlEndpoint:
    def __init__(self, store, prefix, device, core, *, sender, lite=False, timeout_s=300):
        self.device = torch.device(device)
        if self.device.type != "neuron":
            raise ValueError("NIXL requires real Trainium HBM; CPU transport is not supported")
        self.store = torch.distributed.PrefixStore(prefix, store)
        self.sender, self.lite, self.timeout_s = sender, lite, timeout_s
        self.sequence = 0
        self.buffer = self.registration = self.remote = None
        self.address = self.remote_address = None
        self.transfers = {}
        torch.zeros(1, device=self.device).cpu()
        synchronize(lite=lite)
        if "FI_EFA_IFACE" not in os.environ:
            os.environ["FI_EFA_IFACE"] = efa_interface(core)
        agent_type, config_type = import_nixl(lite=lite)
        self.agent = agent_type(
            f"verl-{os.getpid()}-{prefix}",
            config_type(enable_prog_thread=True, enable_listen_thread=False, listen_port=0, backends=["LIBFABRIC"]),
        )
        if "LIBFABRIC" not in self.agent.backends:
            raise RuntimeError("The NIXL LIBFABRIC backend is unavailable; no transport fallback is allowed")

    def register(self, numel):
        if self.buffer is not None:
            if self.buffer.numel() != numel:
                raise ValueError("NIXL slot size cannot change after registration")
            return
        self.buffer = torch.zeros(numel, dtype=torch.bfloat16, device=self.device)
        address, size = slot_address(self.buffer, lite=self.lite)
        self.address = address
        descriptors = self.agent.get_reg_descs([(address, size, 0, "")], "VRAM")
        self.registration = self.agent.register_memory(descriptors, backends=["LIBFABRIC"])
        if self.sender:
            remote_address, remote_size = json.loads(self.store.get("region"))
            if remote_size != size:
                raise ValueError("Sender and receiver disagree on the NIXL slot size")
            self.remote = self.agent.add_remote_agent(self.store.get("agent"))
            if isinstance(self.remote, bytes):
                self.remote = self.remote.decode()
            self.remote_address = remote_address
        else:
            self.store.set("agent", self.agent.get_agent_metadata())
            self.store.set("region", json.dumps([address, size]))

    def push(self, numel):
        """Write only the packed prefix of a fully registered, reusable HBM slot."""
        if not self.sender or self.remote is None or self.buffer is None:
            raise RuntimeError("Connect and register a sender before pushing weights")
        if not 0 < numel <= self.buffer.numel():
            raise ValueError("NIXL payload must be nonempty and fit in the registered slot")
        size = numel * self.buffer.element_size()
        if size not in self.transfers:
            local = self.agent.get_xfer_descs([(self.address, size, 0)], "VRAM")
            remote = self.agent.get_xfer_descs([(self.remote_address, size, 0)], "VRAM")
            self.transfers[size] = self.agent.initialize_xfer("WRITE", local, remote, self.remote, b"")
        transfer = self.transfers[size]
        synchronize(lite=self.lite)
        state = self.agent.transfer(transfer)
        deadline = time.monotonic() + self.timeout_s
        while state == "PROC":
            if time.monotonic() >= deadline:
                raise TimeoutError("NIXL HBM write timed out")
            time.sleep(0.0001)
            state = self.agent.check_xfer_state(transfer)
        if state != "DONE":
            raise RuntimeError(f"NIXL HBM write failed: {state}")
        self.store.set(f"written/{self.sequence}", b"ok")
        acknowledgement = self.store.get(f"applied/{self.sequence}")
        self.store.delete_key(f"applied/{self.sequence}")
        if acknowledgement != b"ok":
            raise RuntimeError(f"NIXL receiver rejected weights: {acknowledgement.decode()}")
        self.sequence += 1

    def receive(self, apply):
        if self.sender or self.buffer is None:
            raise RuntimeError("Connect and register a receiver before receiving weights")
        self.store.get(f"written/{self.sequence}")
        self.store.delete_key(f"written/{self.sequence}")
        try:
            result = apply(self.buffer)
            synchronize(lite=self.lite)
        except Exception as error:
            self.store.set(f"applied/{self.sequence}", b"error: " + str(error).encode())
            raise
        self.store.set(f"applied/{self.sequence}", b"ok")
        self.sequence += 1
        return result

    def close(self):
        for transfer in self.transfers.values():
            self.agent.release_xfer_handle(transfer)
        self.transfers.clear()
        if self.remote is not None:
            self.agent.remove_remote_agent(self.remote)
            self.remote = None
        if self.registration is not None:
            self.agent.deregister_memory(self.registration)
            self.registration = None
        self.buffer = None
        self.address = self.remote_address = None
