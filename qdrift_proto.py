#!/usr/bin/env python3
"""
Quantum Flywheel prototype (Track 2).

Two experiments on batched randomized-trajectory (qDRIFT) simulation of the 1D
transverse-field Ising chain:

  precision : run the SAME sampled trajectories in several arithmetic tiers
              (fp64 reference, fp32, 3xTF32, TF32, BF16, BF16 without per-step
              renormalization) and compare the floating-point error of the Monte
              Carlo estimator with its statistical error sigma/sqrt(N).

  grouped   : time one trajectory step when the N trajectories are split into
              J multinomially skewed groups (one propagator per sampled term):
              per-group GEMM loop vs. zero-padded batched GEMM vs. a single
              dense GEMM of the same useful size (the target a persistent
              grouped-GEMM kernel should reach).

Backends: --backend torch (CUDA, real Tensor Cores) or --backend numpy (CPU,
TF32/BF16 emulated by rounding operands to 10/7 mantissa bits, fp32 accumulate).

Term propagators exp(-i theta P_j) are applied as dense real 2D x 2D matrices
(D = 2^n), a deliberately GEMM-shaped stand-in for fused gate blocks / MPS-site
contractions. The study is about grouping and precision, not about the most
efficient way to apply a single Pauli rotation.
"""
import argparse
import json
import math
import os
import platform
import time

import numpy as np

TIERS_ALL = ["fp64", "fp32", "3xtf32", "tf32", "bf16", "bf16_norenorm"]


# --------------------------------------------------------------------------- physics
def tfim_terms(n, J=1.0, h=1.0):
    """H = -J sum Z_i Z_{i+1} - h sum X_i (open chain). Returns [(kind, qubits, coeff)]."""
    terms = [("ZZ", (i, i + 1), -J) for i in range(n - 1)]
    terms += [("X", (i,), -h) for i in range(n)]
    return terms


def pauli_dense(n, kind, qs):
    """Dense real D x D matrix of a Pauli term (little-endian: qubit i <-> bit i)."""
    D = 1 << n
    idx = np.arange(D)
    if kind == "X":
        P = np.zeros((D, D))
        P[idx ^ (1 << qs[0]), idx] = 1.0
        return P
    i, j = qs
    z = (1 - 2 * ((idx >> i) & 1)) * (1 - 2 * ((idx >> j) & 1))
    return np.diag(z.astype(np.float64))


def propagators_T(n, terms, tau):
    """Stack of R_j^T acting on row vectors x = [Re psi, Im psi] for U_j = exp(-i theta_j P_j),
    theta_j = tau * sign(h_j).  U psi = (c a + s P b) + i (c b - s P a)."""
    D = 1 << n
    I = np.eye(D)
    out = []
    for kind, qs, coeff in terms:
        th = tau * math.copysign(1.0, coeff)
        c, s = math.cos(th), math.sin(th)
        P = pauli_dense(n, kind, qs)  # symmetric
        RT = np.block([[c * I, -s * P], [s * P, c * I]])
        out.append(RT)
    return np.stack(out)  # (J, 2D, 2D) float64


def zbar_vector(n):
    idx = np.arange(1 << n)
    return np.mean([(1 - 2 * ((idx >> i) & 1)) for i in range(n)], axis=0)


def sample_sequences(N, M, probs, seed):
    rng = np.random.default_rng(seed)
    return rng.choice(len(probs), size=(N, M), p=probs).astype(np.int64)


# --------------------------------------------------------------------------- numpy backend
def _np_round(x, add, mask):
    b = x.astype(np.float32).view(np.uint32)
    return ((b + np.uint32(add)) & np.uint32(mask)).view(np.float32)


def np_tf32(x):
    return _np_round(x, 0x1000, 0xFFFFE000)


def np_bf16(x):
    return _np_round(x, 0x8000, 0xFFFF0000)


class NumpyOps:
    name = "numpy"

    def __init__(self):
        self.device_name = f"CPU ({platform.processor() or platform.machine()}, {os.cpu_count()} threads)"

    def prep(self, RT64, tier):
        if tier == "fp64":
            return RT64
        R = RT64.astype(np.float32)
        if tier == "3xtf32":
            hi = np_tf32(R)
            return (hi, np_tf32(R - hi))
        if tier == "tf32":
            return np_tf32(R)
        if tier.startswith("bf16"):
            return np_bf16(R)
        return R

    def init_state(self, N, D, tier):
        x = np.zeros((N, 2 * D), dtype=np.float64 if tier == "fp64" else np.float32)
        x[:, 0] = 1.0
        return x

    def mm(self, x, R, tier):
        if tier in ("fp64", "fp32"):
            return x @ R
        if tier == "tf32":
            return np_tf32(x) @ R
        if tier == "3xtf32":
            Rh, Rl = R
            xh = np_tf32(x)
            xl = np_tf32(x - xh)
            return (xl @ Rh + xh @ Rl) + xh @ Rh
        return np_bf16(np_bf16(x) @ R)  # bf16 operands, fp32 accumulate, bf16 storage

    def renorm(self, x, tier):
        if tier == "bf16_norenorm":
            return x
        nrm = np.sqrt(np.sum(x.astype(np.float32 if tier != "fp64" else np.float64) ** 2, axis=1, keepdims=True))
        y = x / nrm
        return np_bf16(y) if tier.startswith("bf16") else y.astype(x.dtype)

    def observable(self, x, zbar):
        x = np.asarray(x, dtype=np.float64)
        D = x.shape[1] // 2
        p = x[:, :D] ** 2 + x[:, D:] ** 2
        return (p @ zbar) / p.sum(axis=1)

    def take(self, a, idx):
        return a[idx]

    def put(self, a, idx, v):
        a[idx] = v

    def sync(self):
        pass

    def now(self):
        return time.perf_counter()


# --------------------------------------------------------------------------- torch backend
class TorchOps:
    name = "torch"

    def __init__(self):
        import torch

        self.t = torch
        self.dev = torch.device("cuda")
        self.device_name = torch.cuda.get_device_name(0)
        self._set_tf32(False)

    def _tf32(self, x):
        b = x.contiguous().view(self.t.int32)
        return ((b + 0x1000) & -0x2000).view(self.t.float32)  # -0x2000 == 0xFFFFE000

    def _set_tf32(self, on):
        # "high" = TF32 Tensor Cores for fp32 matmul, "highest" = true FP32 (stable API across versions)
        self.t.set_float32_matmul_precision("high" if on else "highest")

    def prep(self, RT64, tier):
        t = self.t
        R = t.from_numpy(RT64).to(self.dev)
        if tier == "fp64":
            return R
        R = R.float()
        if tier == "3xtf32":
            hi = self._tf32(R)
            return (hi, self._tf32(R - hi))
        if tier.startswith("bf16"):
            return R.bfloat16()
        return R

    def init_state(self, N, D, tier):
        t = self.t
        dt = {"fp64": t.float64}.get(tier, t.bfloat16 if tier.startswith("bf16") else t.float32)
        x = t.zeros((N, 2 * D), dtype=dt, device=self.dev)
        x[:, 0] = 1.0
        return x

    def mm(self, x, R, tier):
        if tier in ("fp64", "fp32"):
            self._set_tf32(False)
            return x @ R
        if tier == "tf32":
            self._set_tf32(True)
            return x @ R
        if tier == "3xtf32":
            self._set_tf32(True)
            Rh, Rl = R
            xh = self._tf32(x)
            xl = x - xh
            return (xl @ Rh + xh @ Rl) + xh @ Rh
        return x @ R  # bf16 x bf16 -> bf16 (fp32 accumulate on Tensor Cores)

    def renorm(self, x, tier):
        if tier == "bf16_norenorm":
            return x
        y = x.float() if tier != "fp64" else x
        y = y / y.norm(dim=1, keepdim=True)
        return y.to(x.dtype)

    def observable(self, x, zbar):
        x = x.double()
        D = x.shape[1] // 2
        p = x[:, :D] ** 2 + x[:, D:] ** 2
        z = self.t.as_tensor(zbar, device=self.dev)
        return ((p @ z) / p.sum(dim=1)).cpu().numpy()

    def take(self, a, idx):
        return a[idx]

    def put(self, a, idx, v):
        a[idx] = v

    def sync(self):
        self.t.cuda.synchronize()

    def now(self):
        self.sync()
        return time.perf_counter()


# --------------------------------------------------------------------------- experiment 1
def run_precision(ops, args, out):
    n, N, M = args.n, args.N, args.M
    D = 1 << n
    terms = tfim_terms(n)
    lam = sum(abs(c) for _, _, c in terms)
    probs = np.array([abs(c) for _, _, c in terms]) / lam
    tau = lam * args.t / M
    RT64 = propagators_T(n, terms, tau)
    zbar = zbar_vector(n)
    seqs = sample_sequences(N, M, probs, args.seed)
    checkpoints = sorted(set(np.linspace(0, M, args.checkpoints + 1).astype(int)[1:]))

    # group indices per step, shared by all tiers (same sampled trajectories)
    groups = []
    for k in range(M):
        jk = seqs[:, k]
        order = np.argsort(jk, kind="stable")
        counts = np.bincount(jk, minlength=len(terms))
        groups.append([(j, g) for j, g in enumerate(np.split(order, np.cumsum(counts)[:-1])) if len(g)])
    if ops.name == "torch":
        groups = [[(j, ops.t.as_tensor(g, device=ops.dev)) for j, g in gk] for gk in groups]

    obs, times = {}, {}
    for tier in args.tiers:
        R = ops.prep(RT64, tier)
        x = ops.init_state(N, D, tier)
        rec = []
        t0 = ops.now()
        for k in range(M):
            y = x.clone() if ops.name == "torch" else x.copy()
            for j, g in groups[k]:
                Rj = tuple(r[j] for r in R) if isinstance(R, tuple) else R[j]
                ops.put(y, g, ops.mm(ops.take(x, g), Rj, tier))
            x = ops.renorm(y, tier)
            if k + 1 in checkpoints:
                rec.append(ops.observable(x, zbar))
        times[tier] = ops.now() - t0
        obs[tier] = np.stack(rec, axis=1)  # (N, C)
        print(f"  {tier:14s} {times[tier]:8.2f} s   <Zbar>(t)={obs[tier][:, -1].mean():+.5f}", flush=True)

    ref = obs["fp64"]
    sigma = ref.std(axis=0, ddof=1)
    Ns = [2 ** k for k in range(4, int(math.log2(N)) + 1)]
    summary = {"n": n, "N": N, "M": M, "t": args.t, "tau": tau, "lambda": lam, "J_terms": len(terms),
               "checkpoints": [int(c) for c in checkpoints], "sigma": sigma.tolist(),
               "estimate_fp64": ref.mean(axis=0).tolist(), "tiers": {}, "times_s": times,
               "stat_err_vs_N": {str(m): (sigma[-1] / math.sqrt(m)) for m in Ns}}
    for tier in args.tiers:
        if tier == "fp64":
            continue
        d = obs[tier] - ref  # paired per-trajectory differences (N, C)
        bias = d.mean(axis=0)
        se = d.std(axis=0, ddof=1) / math.sqrt(N)
        rms = np.sqrt((d ** 2).mean(axis=0))
        resolved = abs(bias[-1]) > 2 * se[-1]
        b_eff = abs(bias[-1]) if resolved else 2 * se[-1]
        nstar = (sigma[-1] / b_eff) ** 2 if b_eff > 0 else float("inf")
        summary["tiers"][tier] = {
            "bias": bias.tolist(), "bias_se": se.tolist(), "rms": rms.tolist(),
            "max_abs": float(np.abs(d[:, -1]).max()),
            "bias_resolved": bool(resolved), "N_star": float(nstar),
            "N_star_is_lower_bound": bool(not resolved),
            "bias_vs_N": {str(m): float(abs(d[:m, -1].mean())) for m in Ns},
        }
    np.savez_compressed(os.path.join(out, "precision_obs.npz"), seqs=seqs[:64], **obs)
    return summary


# --------------------------------------------------------------------------- experiment 2
def run_grouped(ops, args):
    rng = np.random.default_rng(args.seed + 1)
    n = args.bench_n
    D2 = 2 << n
    J = args.bench_J
    t = ops.t if ops.name == "torch" else None
    res = []
    for tier in args.bench_tiers:
        if ops.name == "torch":
            ops._set_tf32(tier == "tf32")
            R = t.randn(J, D2, D2, device=ops.dev) / math.sqrt(D2)
        else:
            R = (rng.standard_normal((J, D2, D2)) / math.sqrt(D2)).astype(np.float32)
        for N in args.bench_N:
            x = t.randn(N, D2, device=ops.dev) if t else rng.standard_normal((N, D2)).astype(np.float32)
            for alpha in args.alphas:
                p = np.arange(1, J + 1, dtype=float) ** (-alpha)
                p /= p.sum()
                jb = rng.choice(J, size=N, p=p)
                counts = np.bincount(jb, minlength=J)
                order = np.argsort(jb, kind="stable")
                segs = [(j, g) for j, g in enumerate(np.split(order, np.cumsum(counts)[:-1])) if len(g)]
                mmax = int(counts.max())
                if t:
                    segs = [(j, t.as_tensor(g, device=ops.dev)) for j, g in segs]
                    pad_idx = t.zeros(J, mmax, dtype=t.long, device=ops.dev)
                    pad_mask = t.zeros(J, mmax, dtype=t.bool, device=ops.dev)
                    for j, g in segs:
                        pad_idx[j, :len(g)] = g
                        pad_mask[j, :len(g)] = True
                else:
                    pad_idx = np.zeros((J, mmax), dtype=np.int64)
                    pad_mask = np.zeros((J, mmax), dtype=bool)
                    for j, g in segs:
                        pad_idx[j, :len(g)] = g
                        pad_mask[j, :len(g)] = True

                def loop():
                    y = t.empty_like(x) if t else np.empty_like(x)
                    for j, g in segs:
                        y[g] = x[g] @ R[j]
                    return y

                def padded():
                    xp = x[pad_idx] * (pad_mask[..., None] if t else pad_mask[..., None])
                    yp = xp @ R  # batched GEMM (J, mmax, D2)
                    y = t.empty_like(x) if t else np.empty_like(x)
                    y[pad_idx[pad_mask]] = yp[pad_mask]
                    return y

                def single():
                    return x @ R[0]

                row = {"tier": tier, "N": N, "alpha": alpha, "J": J, "D2": D2,
                       "max_group": mmax, "nonempty_groups": len(segs),
                       "pad_waste": float(J * mmax / N)}
                flops = 2.0 * N * D2 * D2
                for name, fn in (("single_gemm", single), ("loop", loop), ("padded_bmm", padded)):
                    for _ in range(2):
                        fn()
                    ops.sync()
                    reps = args.reps
                    t0 = ops.now()
                    for _ in range(reps):
                        fn()
                    dt = (ops.now() - t0) / reps
                    row[name + "_ms"] = dt * 1e3
                    row[name + "_tflops"] = flops / dt / 1e12
                row["loop_eff"] = row["single_gemm_ms"] / row["loop_ms"]
                row["padded_eff"] = row["single_gemm_ms"] / row["padded_bmm_ms"]
                res.append(row)
                print(f"  {tier:5s} N={N:6d} alpha={alpha:3.1f} groups={len(segs):3d} max={mmax:6d} | "
                      f"single {row['single_gemm_tflops']:7.2f}  loop {row['loop_tflops']:7.2f} ({row['loop_eff']:.0%})  "
                      f"padded {row['padded_bmm_tflops']:7.2f} ({row['padded_eff']:.0%}) TFLOP/s", flush=True)
    return res


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["torch", "numpy"], default="torch")
    ap.add_argument("--out", default="results")
    ap.add_argument("--n", type=int, default=10, help="qubits for the precision study")
    ap.add_argument("--N", type=int, default=8192, help="trajectories")
    ap.add_argument("--M", type=int, default=200, help="qDRIFT steps")
    ap.add_argument("--t", type=float, default=2.0, help="evolution time")
    ap.add_argument("--checkpoints", type=int, default=10)
    ap.add_argument("--tiers", nargs="+", default=TIERS_ALL)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--skip-precision", action="store_true")
    ap.add_argument("--skip-grouped", action="store_true")
    ap.add_argument("--bench-n", type=int, default=10, help="qubits for the timing study (2D = 2^(n+1))")
    ap.add_argument("--bench-J", type=int, default=19)
    ap.add_argument("--bench-N", type=int, nargs="+", default=[1024, 8192, 65536])
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.0, 0.5, 1.0, 1.5, 2.0])
    ap.add_argument("--bench-tiers", nargs="+", default=["fp32", "tf32"])
    ap.add_argument("--reps", type=int, default=20)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    ops = TorchOps() if args.backend == "torch" else NumpyOps()
    meta = {"backend": ops.name, "device": ops.device_name, "args": vars(args),
            "date": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    if ops.name == "torch":
        meta["torch"] = ops.t.__version__
        meta["cuda"] = ops.t.version.cuda
    print(f"device: {ops.device_name}")

    if not args.skip_precision:
        print(f"[precision] n={args.n} N={args.N} M={args.M}")
        meta["precision"] = run_precision(ops, args, args.out)
    if not args.skip_grouped:
        print(f"[grouped] n={args.bench_n} J={args.bench_J}")
        meta["grouped"] = run_grouped(ops, args)
    path = os.path.join(args.out, f"results_{ops.name}.json")
    with open(path, "w") as f:
        json.dump(meta, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print("wrote", path)


if __name__ == "__main__":
    main()
