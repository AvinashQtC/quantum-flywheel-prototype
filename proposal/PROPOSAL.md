# Tensor Cores vs. Qubits: Quantum Flywheel Track 2 proposal

Avinash Singh, Fujitsu Research (Bangalore) · October 2026

## What I propose and why it matters

Recent classical refutations of quantum-advantage claims have come from faster contractions, not new theory. GPU tensor networks with U(1)×SU(2) symmetry reached χ≈62,000 and cut Q-CTRL's claimed 3000× Fermi–Hubbard advantage to ~36×, using general-purpose libraries. I propose to attack the layer beneath those libraries. I also extend the attack from closed-system dynamics to thermal and open-system protocols, which have had little adversarial scrutiny.

Both workloads reduce to many independent GEMMs of skewed size. In symmetric tensor networks, the groups are charge sectors. In randomized trajectory simulation (qDRIFT-style compilation, unravelled KMS-detailed-balance Lindbladians), each trajectory samples a term with probability p_j ∝ λ_j. Grouping trajectories by sampled term turns each step into one GEMM per term, with multinomially skewed group sizes. Per-group cuBLAS calls lose throughput to launch overhead, low occupancy and load imbalance.

Against cuTENSOR/cuTensorNet baselines, I will build:

- **K1:** a persistent grouped-GEMM kernel with size-aware scheduling, cp.async pipelining and MMA Tensor Core inner loops, serving both sectors and trajectory groups.
- **K2:** fused permute-GEMM via swizzled shared-memory layouts.
- **K3:** precision-tiered paths (FP64 DMMA on Hopper; 3xTF32/TF32 on Ada).
- **K4:** a batched trajectory engine with on-GPU sampling, segmented sort, and gate fusion into Tensor-Core-sized blocks.

A Pauli-path engine is a stretch goal, as an independent cross-check.

K3 tests an error-budget hypothesis. Tensor-network results carry truncation error, and trajectory estimates carry Hoeffding error ~‖O‖√(2ln(2/δ)/N) plus discretization bias. If floating-point error sits well below these, low-precision Tensor Core arithmetic is physically free. The prototype in this repository confirms this for trajectory estimators: on an L40S, TF32 stays below the statistical error up to ~2×10¹⁰ trajectories (extrapolated from 65,536 measured) and runs 40× faster than FP64.

**Targets:** the Floquet Ising prethermal dynamics claim on IBM Heron r3, up to 74 qubits ([Leviatan et al., arXiv:2607.24937](https://arxiv.org/abs/2607.24937)), where tensor networks did not converge and sparse Pauli-path simulation on Fugaku remained truncation-dependent; and a published thermalization or open-system experiment. IBM QPU runs at contested points, plus small-system runs of the randomized protocol itself, bound classical fidelity where divergence is predicted.

**Why it matters:** advantage factors only mean something against the best classical implementation. Kernel efficiency is the largest unmeasured variable, so I report roofline fraction beside every factor. Mapping where claims survive tells experimentalists which regimes are genuine.

**Deliverables:** open-source kernels, a reproducible benchmark suite, and a preprint.

## Related work and how this goes beyond it

**My prior work**

- [SGEMM from scratch on the NVIDIA L40S](https://github.com/AvinashQtC/cuda-performance-engineering/tree/main/gemm/sgemm-from-scratch): hand-written FP32 kernels up to cp.async double buffering; best kernel 24.4 TFLOP/s (57% of cuBLAS at 4096³). It measures FP32/DRAM ceilings and includes full Nsight Compute analysis. An arithmetic-intensity model predicted 17.18 GB of L2 traffic, and Nsight measured 17.19–17.28 GB. cuBLAS TF32 runs 2.2× faster than FP32 cuBLAS.
- [vLLM PR #61015](https://github.com/vllm-project/vllm/pull/61015): a performance warning for attention backends that fall off Tensor Core paths on pre-SM80 GPUs.
- Tensor Core and fused-kernel work: WMMA/MMA, XOR swizzling for ldmatrix, CuTe layout algebra, CUTLASS 3.x, and FlashAttention in Triton.
- Preprints (co-author): [arXiv:2507.16373](https://arxiv.org/abs/2507.16373) (meta-learning Gibbs states), [arXiv:2503.16361](https://arxiv.org/abs/2503.16361) (classical/quantum workload balancing), [arXiv:2509.16186](https://arxiv.org/abs/2509.16186) (quantum generative adversarial autoencoders).
- Ongoing at Fujitsu Research: error analysis of randomized trajectory estimators for Gibbs-state preparation, building on qDRIFT and Chen–Li–Lu–Ying ([arXiv:2407.06594](https://arxiv.org/abs/2407.06594)).

**Beyond existing results**

1. **Kernel engineering is the missing variable.** Published GPU refutations use general-purpose contraction libraries and report no roofline efficiency. This proposal treats skewed grouped contraction as a dedicated kernel problem and reports efficiency beside every advantage factor.
2. **Thermal and open-system claims.** Adversarial simulation has targeted closed-system dynamics. A batched trajectory engine extends it to thermal and open-system protocols, and one kernel serves both trajectory groups and symmetry sectors.
3. **Precision has never been measured against the error budget.** The proposal tests whether truncation and sampling error make low-precision Tensor Core arithmetic physically free; the prototype confirms this for trajectories.
4. **QPU data maps the divergence boundary.** IBM QPU runs show where classical simulation diverges, instead of assuming where it does.

## Compute requests

| Resource | Request | Basis |
|---|---|---|
| AI tokens | ~10M output tokens | ~350 agentic coding/analysis sessions across K1–K4, profiling, physics pipeline, write-up, plus 40% for retries |
| IBM QPU | 1M–10M shots (~8M), ~100 QPU minutes | Contested-point dynamics 1.8M, divergence sweep 1.2M, trajectory runs 4.0M, noise learning 0.5M, pilots 0.5M |
| GPU | 100–1,000 hours (~800) | Kernel development 200, baseline reproduction 80, large-χ runs 300, precision sweeps 120, trajectories 60, contingency 40 |
