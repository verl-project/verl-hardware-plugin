#!/usr/bin/env bash
# Plain-English check of the per-vendor pip extras. Needs only python3 and internet for pip. No GPU.
#   scripts/verify_extras.sh            ~10 s
#   scripts/verify_extras.sh --resolve  ~3 min: also asks the real package servers whether the XPU pins exist (installs nothing)
#   scripts/verify_extras.sh -v         also print the raw evidence behind each line
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd); T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
V=0; R=0; for a in "$@"; do [ "$a" = -v ] && V=1; [ "$a" = --resolve ] && R=1; done
FAILS=0
pass() { printf 'PASS  %s\n' "$1"; }
fail() { printf 'FAIL  %s\n      -> %s\n' "$1" "$2"; FAILS=$((FAILS+1)); }
raw()  { [ "$V" = 1 ] && sed 's/^/        | /' || true; }
python3 -m venv "$T/venv" && . "$T/venv/bin/activate" && pip -q install build pytest packaging >/dev/null 2>&1

if python -I -m build --wheel --outdir "$T/dist" "$ROOT" >"$T/build.log" 2>&1; then pass "the package builds into a wheel"
else fail "the package builds into a wheel" "see: $T/build.log (re-run with -v)"; cat "$T/build.log" | raw; fi
W=$(ls "$T"/dist/*.whl 2>/dev/null | head -1); M=$(unzip -p "$W" '*/METADATA' 2>/dev/null)

missing=""; for d in "$ROOT"/verl_hardware_plugin/accelerators/*/; do b=$(basename "$d"); [ -f "$d/__init__.py" ] || continue
  echo "$M" | grep -q "^Provides-Extra: $b$" || missing="$missing $b"; done
[ -z "$missing" ] && pass "every hardware backend folder has its own pip extra" || fail "every backend has an extra" "missing:$missing"
echo "$M" | grep '^Provides-Extra' | raw

echo "$M" | grep -q 'torch==2.15.0+xpu; extra == "xpu"' && pass "extra [xpu] pins the Intel build of torch (2.15.0+xpu)" || fail "extra [xpu] pins torch +xpu" "check accelerators/xpu/requirements.txt"
echo "$M" | grep -q 'vllm==0.34.0+xpu; extra == "vllm-xpu"' && pass "extra [vllm-xpu] pins the Intel build of vLLM" || fail "extra [vllm-xpu] pins vllm" "check accelerators/xpu/requirements-vllm.txt"
echo "$M" | grep 'extra == "xpu"' | raw

cp -r "$ROOT" "$T/copy"; rm -rf "$T/copy/.git"
echo '--extra-index-url https://example.invalid/simple' >> "$T/copy/verl_hardware_plugin/accelerators/xpu/requirements.txt"
if (cd "$T/copy" && python -m pytest -q tests/test_vendor_extras.py >"$T/neg.log" 2>&1); then fail "the safety test rejects a bad requirements file" "it PASSED a bad file: the test is broken"
else pass "the safety test rejects a bad requirements file (we planted one on purpose)"; fi; tail -3 "$T/neg.log" | raw
(cd "$ROOT" && python -m pytest -q tests/test_vendor_extras.py >"$T/pos.log" 2>&1) && pass "the safety test accepts the real files" || fail "the safety test accepts the real files" "see -v"; tail -2 "$T/pos.log" | raw

if [ "$R" = 1 ]; then
  SHA=${VERL_CORE_SHA:-8612d0292c38dbe7b104a8c783594f5c57119fa7}
  F="--extra-index-url https://download.pytorch.org/whl/xpu --extra-index-url https://wheels.vllm.ai/0.34.0/xpu --extra-index-url https://wheels.vllm.ai/xpu --extra-index-url https://vllm-project.github.io/vllm-xpu-kernels/whl"
  if pip install --dry-run --ignore-installed "verl[verl-core] @ git+https://github.com/verl-project/verl.git@${SHA}" "$ROOT[xpu,vllm-xpu]" $F >"$T/res.log" 2>&1; then
    pass "the pinned Intel torch, vLLM and kernels all exist on the real package servers today"; grep '^Would install' "$T/res.log" | tr ' ' '\n' | grep -E '^(torch|vllm|kernels)-' | raw
  else fail "the pins resolve on the real servers" "a pinned version is gone or two pins conflict (see -v)"; tail -5 "$T/res.log" | raw; fi
  if pip install --dry-run --ignore-installed "$ROOT[xpu]" >"$T/nores.log" 2>&1; then fail "without the Intel index pip refuses (so it cannot grab CUDA torch)" "pip succeeded without the flag: it could install the wrong torch"
  else pass "without the Intel index pip refuses to continue (so it cannot silently install CUDA torch)"; fi
fi
echo; [ "$FAILS" = 0 ] && echo "RESULT: ALL PASS" || echo "RESULT: $FAILS FAILED"; exit $((FAILS>0))
