# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Public vLLM-Neuron worker with the synchronization needed by policy updates."""

from vllm_neuron.vllm.worker.neuron_worker import NeuronWorker

from ..weight_sync.nixl_transport import synchronize


class NeuronRLWorker(NeuronWorker):
    def synchronize_device(self):
        """Pause only completes after the assigned lite device drains its work."""
        synchronize(lite=True)
