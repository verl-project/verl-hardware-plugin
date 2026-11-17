# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Opt-in native engine acceptance test, launched under torchrun on idle cores."""

import os
from dataclasses import replace
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_NEURON_HARDWARE_TESTS") != "1", reason="requires allocated Trainium cores and an HF checkpoint"
)


def test_native_forward_backward_and_checkpoint(tmp_path, monkeypatch, request):
    import torch
    from tensordict import TensorDict

    from verl.trainer.config import CheckpointConfig
    from verl.utils.distributed import initialize_global_process_group
    from verl.utils.metric import Metric
    from verl.utils.tensordict_utils import assign_non_tensor
    from verl.workers.config import HFModelConfig, TorchtitanOptimizerConfig
    from verl.workers.config.actor import ActorConfig
    from verl.workers.engine_workers import TrainingWorker
    from verl.workers.utils.losses import ppo_loss
    from verl.workers.utils.padding import no_padding_2_padding
    from verl_hardware_plugin.accelerators.neuron.engines.config import NeuronTorchtitanEngineConfig
    from verl_hardware_plugin.accelerators.neuron.engines.torchtitan_neuron import TorchTitanNeuronEngineWithLMHead

    monkeypatch.chdir(tmp_path)
    initialize_global_process_group()
    request.addfinalizer(torch.distributed.destroy_process_group)
    world_size = torch.distributed.get_world_size()
    tensor_parallel_size = int(os.environ.get("NEURON_TEST_TP", "1"))
    data_parallel_size = world_size // tensor_parallel_size
    micro_batch_size = int(os.environ.get("NEURON_TEST_MICRO_BATCH", "1"))
    engine = TorchTitanNeuronEngineWithLMHead(
        HFModelConfig(path=os.environ["MODEL_PATH"], use_remove_padding=False, use_fused_kernels=False),
        NeuronTorchtitanEngineConfig(
            max_seq_len=int(os.environ.get("NEURON_TEST_SEQ_LEN", "512")),
            tensor_parallel_size=tensor_parallel_size,
            data_parallel_size=data_parallel_size,
            data_parallel_shard_size=data_parallel_size,
            activation_checkpoint=os.environ.get("NEURON_TEST_AC", "none"),
        ),
        TorchtitanOptimizerConfig(lr=1e-6, total_training_steps=2),
        CheckpointConfig(),
    )
    request.addfinalizer(engine.trainer.close)
    assert engine.trainer.device.type == "neuron"
    engine.initialize()
    original_forward = engine.model_forward_step

    def checked_forward(**kwargs):
        from torch.distributed.tensor import DTensor

        predictions = original_forward(**kwargs)
        if tensor_parallel_size > 1:
            assert isinstance(predictions, DTensor)
            assert predictions.to_local().shape[-1] == engine.model_config.hf_config.vocab_size // tensor_parallel_size
        return predictions

    monkeypatch.setattr(engine, "model_forward_step", checked_forward)
    from verl_hardware_plugin.accelerators.neuron.engines.snapshots import restore_model_parts, save_model_parts

    snapshot = save_model_parts(engine.module)
    with torch.no_grad():
        for parameter in engine.module[0].parameters():
            parameter.to_local().zero_()
    restore_model_parts(engine.module, snapshot)
    for name, parameter in engine.module[0].named_parameters():
        torch.testing.assert_close(parameter.to_local().detach().cpu(), snapshot[0][name][0], atol=0, rtol=0)
    del snapshot
    from verl_hardware_plugin.accelerators.neuron.weight_sync.nixl_transport import import_nixl, synchronize

    import_nixl()
    exported_name, exported_weight = next(engine.get_per_tensor_param()[0])
    synchronize()
    assert exported_name == "lm_head.weight"
    assert exported_weight.device.type == "neuron"
    assert exported_weight.dtype == torch.bfloat16
    assert torch.isfinite(exported_weight.flatten()[0]).item()
    del exported_weight
    input_offset = 32 * engine.get_data_parallel_rank()
    rows = [torch.arange(16 + input_offset, 32 + input_offset) for _ in range(micro_batch_size)]
    inputs = torch.nested.nested_tensor(rows, layout=torch.jagged)
    batch = TensorDict(
        {
            "input_ids": inputs,
            "prompts": torch.nested.nested_tensor([row[:8] for row in rows], layout=torch.jagged),
            "responses": torch.nested.nested_tensor([row[8:] for row in rows], layout=torch.jagged),
            "temperature": torch.ones(micro_batch_size),
            "loss_mask": torch.nested.nested_tensor(
                [torch.cat([torch.zeros(8), torch.ones(8)]) for _ in rows], layout=torch.jagged
            ),
            "response_mask": torch.ones(micro_batch_size, 8),
            "advantages": torch.ones(micro_batch_size, 8),
        },
        batch_size=[micro_batch_size],
    )
    assign_non_tensor(
        batch,
        use_dynamic_bsz=False,
        micro_batch_size_per_gpu=micro_batch_size,
        global_batch_size=data_parallel_size * micro_batch_size,
        calculate_entropy=True,
    )
    if os.environ.get("NEURON_TEST_REFERENCE") == "1":
        reference_config = replace(engine.engine_config, forward_only=True, activation_checkpoint="none")
        reference = TorchTitanNeuronEngineWithLMHead(
            engine.model_config, reference_config, TorchtitanOptimizerConfig(), CheckpointConfig()
        )
        request.addfinalizer(reference.trainer.close)
        reference.initialize()
        reference_output = reference.infer_batch(batch)
        assert torch.isfinite(reference_output["model_output"]["log_probs"].values()).all()
        del reference_output
    loss_config = ActorConfig(
        strategy="torchtitan",
        rollout_n=2,
        ppo_micro_batch_size_per_gpu=micro_batch_size,
        use_kl_loss=True,
        kl_loss_type="low_var_kl",
    )

    def loss_function(model_output, data, dp_group):
        baseline = no_padding_2_padding(model_output["log_probs"], data).detach()
        data["old_log_probs"] = baseline
        data["ref_log_prob"] = baseline + 0.01
        return ppo_loss(loss_config, model_output, data, dp_group)

    norm_before = engine.hf_parameters()["model.norm.weight"].to_local().detach().cpu().clone()
    output = engine.train_batch(batch, loss_function)
    if engine.is_mp_src_rank_with_outputs():
        assert output["metrics"]["grad_norm"] > 0
    norm_after = engine.hf_parameters()["model.norm.weight"].to_local().detach().cpu()
    assert not torch.equal(norm_before, norm_after)
    worker = SimpleNamespace(device_name="neuron", engine=engine, flops_counter=None)
    processed = TrainingWorker._postprocess_output(
        worker, output, global_token_num=None, delta_time=1, forward_only=False, images_seqlens=None
    )
    assert torch.isfinite(torch.tensor(processed["metrics"]["loss"]))
    assert Metric.aggregate_dp(processed["metrics"]["kl_loss"]) > 0
    engine.lr_scheduler_step()
    checkpoint = os.environ["NEURON_TEST_CHECKPOINT_DIR"]
    engine.save_checkpoint(checkpoint, global_step=1)
    parameter = next(engine.module[0].parameters()).to_local()
    original = parameter.detach().cpu().clone()
    state = next(iter(engine.optimizer.optimizers[0].state.values()))
    optimizer_step = state["step"].detach().cpu().clone()
    scheduler_step = engine.lr_scheduler.schedulers[0].last_epoch
    with torch.no_grad():
        parameter.zero_()
        state["step"].zero_()
    engine.trainer.step = 999
    engine.lr_scheduler.schedulers[0].last_epoch = 999
    engine.load_checkpoint(checkpoint)
    torch.testing.assert_close(parameter.detach().cpu(), original, rtol=0, atol=0)
    restored = next(iter(engine.optimizer.optimizers[0].state.values()))
    torch.testing.assert_close(restored["step"].detach().cpu(), optimizer_step, rtol=0, atol=0)
    assert engine.trainer.step == 1
    assert engine.lr_scheduler.schedulers[0].last_epoch == scheduler_step
