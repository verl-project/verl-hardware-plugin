#!/usr/bin/env bash
# Run this ON an Intel GPU machine, inside the Python environment you installed the plugin into. Plain PASS/FAIL, no expertise needed.
#   scripts/accelerators/xpu/smoke_xpu.sh
FAILS=0
check() { # name, python-code, what to do if it fails
  out=$(python - 2>&1 <<PY
import warnings; warnings.filterwarnings("ignore")
$2
PY
); rc=$?
  if [ $rc = 0 ]; then printf 'PASS  %s   [%s]\n' "$1" "$(echo "$out" | tail -1)"
  else printf 'FAIL  %s\n      -> %s\n      | %s\n' "$1" "$3" "$(echo "$out" | tail -2 | tr '\n' ' ' | cut -c1-200)"; FAILS=$((FAILS+1)); fi
}
echo "Python: $(python --version 2>&1)  from: $(command -v python)"
check "torch is the Intel (+xpu) build, not CUDA" 'import torch; v=torch.__version__; assert "+xpu" in v, v; print(v)' "reinstall with the three --extra-index-url flags from the install guide"
check "torch can see an Intel GPU and run a kernel on it" 'import torch; assert torch.xpu.is_available(); x=torch.ones(1<<20,device="xpu"); print(torch.xpu.device_count(),"GPU(s), sum =",float((x*2).sum()))' "check the GPU driver (compute-runtime >= 26.18) and that /dev/dri is visible"
# Informational: Docker and from-source installs use --no-deps on purpose (verl pins an older transformers than vLLM needs), so
# `pip check` reports a conflict there even though the stack works. A pip-extras install should be clean.
pc=$(python -m pip check 2>&1); if [ $? = 0 ]; then printf 'PASS  installed packages do not contradict each other\n'
else printf 'WARN  pip check reports conflicts (expected for Docker or --no-deps installs; not expected for a pip-extras install)\n      | %s\n' "$(echo "$pc" | head -2 | tr '\n' ' ' | cut -c1-200)"; fi
check "the plugin is registered with verl (entry point)" 'from importlib.metadata import entry_points as e; x=[p.value for p in e(group="verl.plugins")]; assert "verl_hardware_plugin" in x, x; print(x)' "install the plugin: pip install -e . (or the wheel) into THIS environment"
check "verl picks the Intel platform by itself" 'from verl.plugin.platform import get_platform as g; n=g().device_name; assert n=="xpu", n; print(n)' "check VERL_USE_EXTERNAL_PLUGINS is not set to none, and that torch.xpu is available"
check "the kernels package is in the range the plugin needs" 'import importlib.metadata as m; v=m.version("kernels"); a=tuple(map(int,v.split(".")[:2])); assert (0,16)<=a<(0,17), v; print(v)' "pip install \"kernels>=0.16.0,<0.17.0\""
# Skip vLLM only when it is not installed. A vLLM that is installed but cannot be imported (missing shared library,
# incompatible dependency) must FAIL, so the import runs inside the check.
if python -c "import importlib.metadata as m; m.version('vllm')" 2>/dev/null; then
  check "vLLM imports and is the Intel (+xpu) build" 'import vllm, importlib.metadata as m; v=m.version("vllm"); assert "+xpu" in v, v; print(v)' "reinstall vllm from the wheels.vllm.ai xpu index and check its shared libraries"
fi
echo; [ $FAILS = 0 ] && echo "RESULT: ALL PASS - the install is sound. Next: a 2-step GRPO run (see the install guide)." || echo "RESULT: $FAILS FAILED - fix the first FAIL above, then re-run."; exit $((FAILS>0))
