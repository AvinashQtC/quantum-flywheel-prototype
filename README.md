# Is Tensor Core precision free?

Prototype for a [Quantum Flywheel](https://www.bluequbit.io/quantum-flywheel) Track 2 proposal (breaking quantum advantage claims): error-budgeted, grouped-GEMM GPU simulation of randomized (qDRIFT) quantum trajectories.

**Report:** [`report/report.pdf`](report/report.pdf) (3 pages, NVIDIA L40S results) · **Proposal text:** [`proposal/`](proposal/)

## Results (NVIDIA L40S, 10 qubits, 65,536 trajectories, 200 qDRIFT steps)

The same sampled trajectories were run in each arithmetic tier and compared with FP64. N\* = (σ/|bias|)² is the ensemble size at which a tier's arithmetic bias reaches the statistical error σ/√N.

| Tier | Bias vs FP64 | Free up to N\* | Time (s) |
|---|---|---|---|
| FP64 (reference) | — | — | 109.2 |
| FP32 | 1.8×10⁻⁹ | 5×10¹⁵ | 4.4 |
| 3xTF32 | 1.7×10⁻⁸ | 6×10¹³ | 6.1 |
| TF32 | 9.3×10⁻⁷ | 2×10¹⁰ | 2.7 |
| BF16 | 6.3×10⁻⁵ | 4×10⁶ | 2.4 |

- **TF32 is accurate at every ensemble size a real estimate uses.** It ran 40× faster than FP64, and its bias is ~138× smaller than its per-trajectory error, because rounding errors cancel across trajectories.
- **BF16 is only accurate for small ensembles** (up to ~4×10⁶ trajectories).
- **Standard ways of running the skewed per-term GEMMs leave throughput unused.** At 1,024 trajectories, a per-group loop and a padded batched GEMM both reach about 10% of a single dense TF32 GEMM (91.4 TFLOP/s). At 65,536 trajectories they reach 6–61%. This is the gap a persistent grouped-GEMM kernel targets.

The N\* values above 65,536 are extrapolations from the measured bias.

**First target claim:** Floquet Ising prethermal dynamics on IBM Heron r3, up to 74 qubits ([Leviatan et al., arXiv:2607.24937](https://arxiv.org/abs/2607.24937)). It is in the same transverse-field Ising family as this prototype.

## Run

```bash
pip install numpy matplotlib torch          # + bluequbit qiskit for the optional cross-check
bash run_all.sh                              # GPU: smoke test, sweep, timing, figures, PDF
```

`BLUEQUBIT_API_TOKEN` is optional. When it is set, `run_all.sh` also evaluates sampled trajectories on BlueQubit's GPU statevector, MPS and Pauli-path simulators.

CPU-only run, with TF32/BF16 emulated by bit-exact operand rounding:

```bash
python qdrift_proto.py --backend numpy --out results_cpu --n 8 --N 16384 --bench-n 7 --bench-J 15 --bench-N 1024 8192 --bench-tiers fp32 --reps 5
python make_figures.py results_cpu/results_numpy.json report
```

## Files

| File | Purpose |
|---|---|
| `qdrift_proto.py` | Simulator: precision study and grouped-GEMM timing; NumPy (CPU) and PyTorch (CUDA) backends |
| `bluequbit_check.py` | Optional cross-check of sampled trajectories on BlueQubit simulators |
| `make_figures.py` | Figures, `report/macros.tex` and tables from the result files |
| `run_all.sh` | End-to-end GPU pipeline |
| `report/` | LaTeX source, L40S figures and macros, compiled PDF |
| `results_cpu/` | CPU-emulated run (8 qubits, 16,384 trajectories) |
| `proposal/` | Proposal text, related work and compute requests |

Related: [SGEMM from scratch on the L40S](https://github.com/AvinashQtC/cuda-performance-engineering/tree/main/gemm/sgemm-from-scratch).
