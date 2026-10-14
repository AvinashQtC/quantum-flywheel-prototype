#!/usr/bin/env bash
# End-to-end prototype run on a CUDA GPU (tested target: NVIDIA L40S, sm_89).
# Usage:  bash run_all.sh            (set BLUEQUBIT_API_TOKEN to include the BlueQubit cross-check)
set -euo pipefail
cd "$(dirname "$0")"

echo "== 1/5 smoke test: torch (GPU) vs numpy (CPU) on a tiny problem =="
python qdrift_proto.py --backend torch --out smoke_gpu --n 5 --N 256 --M 20 --skip-grouped
python qdrift_proto.py --backend numpy --out smoke_cpu --n 5 --N 256 --M 20 --skip-grouped
python - <<'EOF'
import numpy as np
a = np.load("smoke_gpu/precision_obs.npz"); b = np.load("smoke_cpu/precision_obs.npz")
d = np.abs(a["fp64"] - b["fp64"]).max()
print(f"max |torch fp64 - numpy fp64| = {d:.2e}")
assert d < 1e-10, "torch and numpy FP64 paths disagree"
for t in ["fp32", "tf32", "bf16"]:
    print(f"{t:5s} torch-vs-fp64 {np.abs(a[t]-a['fp64']).max():.1e}   numpy-emulated-vs-fp64 {np.abs(b[t]-b['fp64']).max():.1e}")
EOF

echo "== 2/5 precision study + grouped-GEMM timing (GPU) =="
python qdrift_proto.py --backend torch --out results_gpu --n 10 --N 65536 --M 200 \
    --bench-n 10 --bench-J 19 --bench-N 1024 8192 65536 --bench-tiers fp32 tf32 --reps 20

echo "== 3/5 BlueQubit cross-check =="
if [ -n "${BLUEQUBIT_API_TOKEN:-}" ]; then
    python bluequbit_check.py --results results_gpu --K 8
else
    echo "BLUEQUBIT_API_TOKEN not set: skipping"
fi

echo "== 4/5 figures + macros =="
python make_figures.py results_gpu/results_torch.json report

echo "== 5/5 PDF =="
if command -v pdflatex >/dev/null; then
    (cd report && pdflatex -interaction=nonstopmode report.tex >/dev/null && pdflatex -interaction=nonstopmode report.tex >/dev/null && echo "report/report.pdf")
else
    echo "pdflatex not found: send the results_gpu/ folder back and the PDF will be built from it"
fi
