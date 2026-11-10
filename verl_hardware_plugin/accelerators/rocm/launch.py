# Copyright 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Launch verl PPO/GRPO from a Primus experiment YAML.

    python -m verl_hardware_plugin.accelerators.rocm.launch <primus.yaml> [--primus-module NAME] [verl overrides ...]

The module's ``framework`` picks the trainer: ``megatron`` -> ``model_engine=primus``,
``maxtext`` -> ``model_engine=maxtext``. A Megatron module's ``hf_path`` becomes the model path
unless an override sets one; MaxText runs set ``actor_rollout_ref.model.path``. Remaining
arguments are passed to ``verl.trainer.main_ppo`` unchanged.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ENGINE_NODES = {
    "megatron": ("primus", "actor_rollout_ref.actor.megatron"),
    "maxtext": ("maxtext", "actor_rollout_ref.actor.maxtext"),
}


def build_overrides(primus_config: str, module: str, user_overrides: list[str]) -> list[str]:
    from .primus.yaml_config import load_module_params, module_framework

    path = str(Path(primus_config).expanduser().resolve())
    framework = module_framework(path, module)
    if framework not in ENGINE_NODES:
        raise ValueError(f"Primus module {module!r} uses framework={framework!r}; supported: {sorted(ENGINE_NODES)}")
    model_engine, node = ENGINE_NODES[framework]
    params = load_module_params(path, module=module, framework=framework)

    overrides = [f"model_engine={model_engine}", f"{node}.primus_config_path={path}", f"{node}.primus_module={module}"]
    user_keys = {item.split("=", 1)[0].lstrip("+~") for item in user_overrides}
    # Megatron's hf_path is the model; MaxText's is its Hugging Face dataset.
    if framework == "megatron" and params.get("hf_path") and "actor_rollout_ref.model.path" not in user_keys:
        overrides.append(f"actor_rollout_ref.model.path={params['hf_path']}")
    if params.get("use_packed_attention") and "actor_rollout_ref.model.use_remove_padding" not in user_keys:
        overrides.append("actor_rollout_ref.model.use_remove_padding=True")
    return overrides + list(user_overrides)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("primus_config", help="Primus experiment YAML")
    parser.add_argument("--primus-module", default="pre_trainer", help="module of the experiment to train")
    args, rest = parser.parse_known_args(argv)
    command = [
        sys.executable,
        "-m",
        "verl.trainer.main_ppo",
        *build_overrides(args.primus_config, args.primus_module, rest),
    ]
    print("+ " + " ".join(command), flush=True)
    os.execv(sys.executable, command)


if __name__ == "__main__":
    main()
