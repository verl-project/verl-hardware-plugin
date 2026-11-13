# Run verl on AWS Trainium

This experimental backend runs **verl 0.9.1** with native Neuron TorchTitan
training and vLLM-Neuron rollout. This guide covers environment setup and a small
GRPO example using Qwen3-0.6B and GSM8K on one Trainium2 host.

## 1. Set up the environment

### Prepare the host

Start with an idle **Ubuntu 26.04 Trainium2 host** with EFA enabled. The pinned
TorchNeuron binary requires GLIBC 2.43. Run these commands on the host to install
the system dependencies:

```bash
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
  ca-certificates curl git gnupg libnuma-dev "linux-headers-$(uname -r)"

NEURON_SETUP_DIR=$(mktemp -d)

# Install the Neuron driver, runtime, collectives, and command-line tools.
curl -fsSL https://apt.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB \
  -o "$NEURON_SETUP_DIR/neuron.asc"
sudo gpg --dearmor --yes -o /usr/share/keyrings/aws-neuron.gpg \
  "$NEURON_SETUP_DIR/neuron.asc"
echo "deb [signed-by=/usr/share/keyrings/aws-neuron.gpg] https://apt.repos.neuron.amazonaws.com resolute main" \
  | sudo tee /etc/apt/sources.list.d/neuron.list
sudo apt-get update
sudo apt-get install -y \
  aws-neuronx-dkms=2.31.7.0 \
  aws-neuronx-runtime-lib=2.35.30.0-d2725ccb8 \
  aws-neuronx-collectives=2.35.43.0-354ad3064 \
  aws-neuronx-tools

# EFA 1.51.0 supplies libfabric 2.7.0amzn2.0 and the EFA kernel driver.
curl -fsSL https://efa-installer.amazonaws.com/aws-efa-installer-1.51.0.tar.gz \
  -o "$NEURON_SETUP_DIR/efa.tar.gz"
tar -xzf "$NEURON_SETUP_DIR/efa.tar.gz" -C "$NEURON_SETUP_DIR"
(
  cd "$NEURON_SETUP_DIR/aws-efa-installer"
  sudo bash efa_installer.sh --yes --skip-mpi --skip-plugin
)

# Install the CUDA 12 driver stub needed by NIXL, using NVIDIA's Ubuntu 24.04 repo.
curl -fsSL \
  https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb \
  -o "$NEURON_SETUP_DIR/cuda-keyring.deb"
sudo dpkg -i "$NEURON_SETUP_DIR/cuda-keyring.deb"
sudo apt-get update
sudo apt-get install -y --no-install-recommends cuda-cudart-dev-12-6
sudo install -d /opt/aws/neuron/cuda-stubs
sudo ln -sfn /usr/local/cuda-12.6/targets/x86_64-linux/lib/stubs/libcuda.so \
  /opt/aws/neuron/cuda-stubs/libcuda.so.1

sudo modprobe neuron
cat /sys/module/neuron/version
/opt/amazon/efa/bin/fi_info --version
/opt/amazon/efa/bin/fi_info -p efa
```

The version checks should report Neuron driver `2.31.7.0` and libfabric
`2.7.0amzn2.0`, and `fi_info -p efa` should list an EFA provider. If an older
Neuron driver is still loaded after installation, reboot the host before
continuing.

### Install the Python dependencies

The [setup script](../../../scripts/accelerators/neuron/setup_neuron_public.sh)
creates one virtual environment with Python 3.12.13, PyTorch 2.13.0 (CPU wheel),
verl 0.9.1, Neuron TorchTitan, and vLLM-Neuron. All workers use this environment.

Run the following from the plugin checkout. Set `PUBLIC_ENV_ROOT` to an absolute,
writable path with room for the environment, model weights, and compiler caches.
Activate it once and keep using the same shell for the example below.

```bash
curl -fsSL https://astral.sh/uv/0.11.14/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

export PLUGIN_DIR="$PWD"
export PUBLIC_ENV_ROOT=/absolute/path/to/neuron-env
bash "$PLUGIN_DIR/scripts/accelerators/neuron/setup_neuron_public.sh" "$PUBLIC_ENV_ROOT"
source "$PUBLIC_ENV_ROOT/venv/bin/activate"

export VERL_NEURON_PYTHON="$VIRTUAL_ENV/bin/python"
export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib:/opt/aws/neuron/lib:/opt/aws/neuron/cuda-stubs:/opt/amazon/efa/lib:/opt/amazon/efa/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# Use controller settings for data preparation; the launcher sets each worker's role.
export VERL_PLATFORM=aws
export VERL_NEURON_PROCESS_ROLE=controller
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export VLLM_PLUGINS=""

python "$PLUGIN_DIR/scripts/accelerators/neuron/check_neuron_environment.py" --hardware
```

Resolve any check failures before training. `VERL_NEURON_PYTHON` selects the same
interpreter for every worker. Caches use their defaults, with NKI binaries stored
in `$PUBLIC_ENV_ROOT/nki-cache`.

## 2. Run a minimal GRPO example

This example uses **four logical NeuronCores**: two for the actor and frozen
reference, and two for one rollout replica. Both use tensor parallelism of two.
On Trainium2, the launcher defaults to `NEURON_LOGICAL_NC_CONFIG=2`.
The verl fields named `n_gpus_per_node` count NeuronCores in this backend.

The run uses four prompts per step, two responses per prompt, and two training
steps. It is a smoke run for setup, generation, training, and weight
synchronization; it is too short to assess model quality.

### Prepare the model and dataset

Download Qwen3-0.6B and use the GSM8K preprocessing script from the pinned verl
release. The script creates the prompt and answer fields used by verl's built-in
GSM8K reward function.

```bash
export NEURON_EXAMPLE_DIR="$PUBLIC_ENV_ROOT/examples/gsm8k"
export MODEL_DIR="$NEURON_EXAMPLE_DIR/Qwen3-0.6B"
export DATA_DIR="$NEURON_EXAMPLE_DIR/data"
export CONFIG_PATH="$NEURON_EXAMPLE_DIR/config"
mkdir -p "$DATA_DIR" "$CONFIG_PATH"

hf download Qwen/Qwen3-0.6B --local-dir "$MODEL_DIR"
curl -fsSL \
  https://raw.githubusercontent.com/verl-project/verl/1876b06d0a3e4e71e06230be10af14492ca8a75b/examples/data_preprocess/gsm8k.py \
  -o "$NEURON_EXAMPLE_DIR/prepare_gsm8k.py"
python "$NEURON_EXAMPLE_DIR/prepare_gsm8k.py" \
  --local_save_dir "$DATA_DIR"
```

### Create the training configuration

This configuration extends verl's `ppo_trainer` defaults. The actor and reference
share the Neuron engine settings through a YAML anchor.

```bash
cat > "$CONFIG_PATH/neuron_gsm8k.yaml" <<'YAML'
defaults:
  - ppo_trainer
  - override model_engine: torchtitan
  - _self_

hydra:
  searchpath:
    - pkg://verl.trainer.config

data:
  train_files: ${oc.env:DATA_DIR}/train.parquet
  val_files: ${oc.env:DATA_DIR}/test.parquet
  train_max_samples: 32
  val_max_samples: 8
  train_batch_size: 4
  max_prompt_length: 512
  max_response_length: 512
  filter_overlong_prompts: true
  dataloader_num_workers: 0
  apply_chat_template_kwargs:
    enable_thinking: false

algorithm:
  adv_estimator: grpo

actor_rollout_ref:
  hybrid_engine: false
  model:
    path: ${oc.env:MODEL_DIR}
    use_remove_padding: false
    use_fused_kernels: false
  actor:
    ppo_mini_batch_size: 4
    ppo_micro_batch_size_per_gpu: 1
    use_dynamic_bsz: false
    use_kl_loss: true
    kl_loss_coef: 0.001
    optim:
      lr: 1.0e-6
    torchtitan: &neuron_engine
      _target_: verl_hardware_plugin.accelerators.neuron.engines.config.NeuronTorchtitanEngineConfig
      attn_type: sdpa
      spmd_backend: default
      dtype: bfloat16
      use_remove_padding: false
      activation_checkpoint: none
      max_seq_len: 1024
      tensor_parallel_size: 2
      data_parallel_size: 1
      data_parallel_replicate_size: 1
      data_parallel_shard_size: 1
      param_offload: false
      optimizer_offload: false
  ref:
    log_prob_micro_batch_size_per_gpu: 1
    log_prob_use_dynamic_bsz: false
    torchtitan:
      <<: *neuron_engine
      forward_only: true
  rollout:
    name: vllm
    nnodes: 1
    n_gpus_per_node: 2
    tensor_model_parallel_size: 2
    dtype: bfloat16
    load_format: auto
    n: 2
    temperature: 1.0
    top_k: 256
    top_p: 1.0
    max_model_len: 1024
    max_num_batched_tokens: 1024
    max_num_seqs: 4
    calculate_log_probs: true
    log_prob_micro_batch_size_per_gpu: 1
    log_prob_use_dynamic_bsz: false
    enable_sleep_mode: false
    free_cache_engine: false
    enable_prefix_caching: false
    enable_chunked_prefill: false
    agent:
      num_workers: 1
    checkpoint_engine:
      backend: neuron_nixl
      custom_backend_module: verl_hardware_plugin.accelerators.neuron.weight_sync.checkpoint_engine
      update_weights_bucket_megabytes: 64

reward:
  num_workers: 1

trainer:
  device: neuron
  use_v1: true
  nnodes: 1
  n_gpus_per_node: 2
  total_epochs: 1
  total_training_steps: 2
  project_name: neuron_gsm8k
  experiment_name: qwen3_0_6b_smoke
  logger: [console]
  val_before_train: false
  test_freq: -1
  save_freq: -1
  resume_mode: disable
  v1:
    trainer_mode: separate_async
    separate_async:
      parameter_sync_step: 1
YAML
```

The training sequence length is `512 + 512 = 1024`. Keep
`torchtitan.max_seq_len` equal to the prompt plus response limit and divisible by
512 when changing these values. This backend uses fixed micro-batches and padded
inputs. CP, EP, and PP remain one; packed inputs, offloading, MoE, quantization,
learned critics, multimodal inputs, colocated rollout, and multi-node jobs are
outside the supported configuration.

### Launch and inspect the run

```bash
export CONFIG_NAME=neuron_gsm8k
export RUN_DIR="$NEURON_EXAMPLE_DIR/runs/$(date +%Y%m%d-%H%M%S)"
bash "$PLUGIN_DIR/scripts/accelerators/neuron/run_neuron.sh"
```

The launcher starts verl and writes logs to `$RUN_DIR/train.log`. It refuses an
existing Ray cluster. Run the environment check in the setup section before
launching. Cold native compilation can take more than 15 minutes before training
starts. Preserve the NKI and vLLM caches at their original paths: compiled graphs
contain absolute paths to NKI binaries.

Look for two completed training steps and their console metrics. This example
disables evaluation and disk checkpoints; actor weights still synchronize to
rollout through `neuron_nixl`. For a longer run with evaluation, pass Hydra
overrides to the same launcher:

```bash
RUN_DIR="$NEURON_EXAMPLE_DIR/runs/longer-$(date +%Y%m%d-%H%M%S)" \
  bash "$PLUGIN_DIR/scripts/accelerators/neuron/run_neuron.sh" \
  data.train_max_samples=-1 data.val_max_samples=32 \
  trainer.total_training_steps=20 trainer.test_freq=5
```
