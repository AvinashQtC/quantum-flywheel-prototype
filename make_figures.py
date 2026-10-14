#!/usr/bin/env python3
"""Figures + LaTeX macros for the prototype report, from results_<backend>.json."""
import json
import math
import sys
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SER = {"fp32": "#2a78d6", "3xtf32": "#eb6834", "tf32": "#1baf7a", "bf16": "#6250d6", "bf16_norenorm": "#e87ba4"}
LBL = {"fp32": "FP32", "3xtf32": "3xTF32", "tf32": "TF32", "bf16": "BF16", "bf16_norenorm": "BF16, no renorm"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "lines.linewidth": 2.0,
    "legend.frameon": False, "savefig.dpi": 300, "savefig.bbox": "tight",
})


def sci(x, digits=1):
    if x == 0 or not math.isfinite(x):
        return str(x)
    e = int(math.floor(math.log10(abs(x))))
    m = x / 10 ** e
    return f"{m:.{digits}f}\\times10^{{{e}}}"


def main(path, outdir):
    r = json.load(open(path))
    os.makedirs(outdir, exist_ok=True)
    P = r.get("precision")
    G = r.get("grouped")
    mac = {"Device": r["device"].replace("_", "\\_"), "Backend": r["backend"],
           "BackendNote": ("TF32 and BF16 are emulated on this CPU run by bit-exact rounding of operands to 10 and 7 mantissa bits with FP32 accumulation, matching Tensor Core semantics; the same script runs on real Tensor Cores with --backend torch."
                           if r["backend"] == "numpy" else
                           f"All tiers run on real Tensor Cores via cuBLAS (PyTorch {r.get('torch', '')}, CUDA {r.get('cuda', '')}).")}

    if P:
        sig = P["sigma"][-1]
        Nmax = P["N"]
        tiers = [t for t in ["fp32", "3xtf32", "tf32", "bf16"] if t in P["tiers"]]
        # ---- Fig 1: estimator error budget vs N
        fig, ax = plt.subplots(figsize=(6.4, 3.3))
        xs = [2 ** k for k in range(4, 31)]
        ax.plot(xs, [sig / math.sqrt(x) for x in xs], color=INK, lw=2, zorder=3)
        ax.text(xs[2], sig / math.sqrt(xs[2]) * 1.5, r"statistical error $\sigma/\sqrt{N}$", color=INK, fontsize=8)
        labels = []
        for t in tiers:
            d = P["tiers"][t]
            b = abs(d["bias"][-1]) if d["bias_resolved"] else 2 * d["bias_se"][-1]
            ns = sorted(int(k) for k in d["bias_vs_N"])
            ax.plot(ns, [max(d["bias_vs_N"][str(k)], 1e-13) for k in ns], "o", ms=3.5, color=SER[t], alpha=0.55,
                    mec="white", mew=0.6)
            nstar = d["N_star"]
            xe = min(nstar, xs[-1])
            ax.plot([ns[0], Nmax], [b, b], color=SER[t], lw=2)
            ax.plot([Nmax, xe], [b, b], color=SER[t], lw=1.4, ls=(0, (3, 2)))
            if nstar <= xs[-1]:
                ax.plot([nstar], [b], "o", ms=7, color=SER[t], mec="white", mew=1.5, zorder=4)
            labels.append([b, LBL[t]])
        labels.sort()
        ys = []
        for b, txt in labels:
            y = b if not ys or b / ys[-1] > 4 else ys[-1] * 4
            ys.append(y)
            ax.text(xs[-1] * 1.25, y, txt, color=INK, va="center", fontsize=8)
        ax.axvline(Nmax, color=INK2, lw=0.8, ls=":")
        ax.text(Nmax * 1.15, ax.get_ylim()[0] if False else 3e-12, f"measured\nup to N={Nmax:,}", color=INK2, fontsize=7)
        ax.set_xscale("log", base=10)
        ax.set_yscale("log")
        ax.set_xlim(10, xs[-1])
        ax.set_ylim(1e-12, 1)
        ax.set_xlabel("number of trajectories N")
        ax.set_ylabel(r"error in $\langle \bar Z\rangle$ estimate")
        fig.savefig(os.path.join(outdir, "fig_budget.pdf"))
        plt.close(fig)

        # ---- Fig 2: per-trajectory rms error vs depth
        fig, ax = plt.subplots(figsize=(3.3, 2.6))
        steps = P["checkpoints"]
        for t in [t for t in ["fp32", "3xtf32", "tf32", "bf16", "bf16_norenorm"] if t in P["tiers"]]:
            ax.plot(steps, P["tiers"][t]["rms"], color=SER[t], marker="o", ms=3, label=LBL[t])
        ax.set_yscale("log")
        ax.set_xlabel("qDRIFT step")
        ax.set_ylabel("per-trajectory rms error")
        ax.legend(fontsize=6.5, loc="center right", bbox_to_anchor=(1.0, 0.42))
        fig.savefig(os.path.join(outdir, "fig_depth.pdf"))
        plt.close(fig)

        mac.update({"Nq": P["n"], "Ntraj": f"{Nmax:,}".replace(",", "{,}"), "Msteps": P["M"],
                    "Jterms": P["J_terms"], "Tevo": P["t"], "SigmaVal": f"{sig:.3f}",
                    "StatErr": sci(sig / math.sqrt(Nmax)), "Est": f"{P['estimate_fp64'][-1]:+.4f}"})
        for t, d in P["tiers"].items():
            k = {"fp32": "Fp", "3xtf32": "Three", "tf32": "Tf", "bf16": "Bf", "bf16_norenorm": "Bfn"}[t]
            mac[f"Bias{k}"] = sci(abs(d["bias"][-1]))
            mac[f"Rms{k}"] = sci(d["rms"][-1])
            mac[f"Max{k}"] = sci(d["max_abs"])
            mac[f"Nstar{k}"] = sci(d["N_star"], 0) if math.isfinite(d["N_star"]) else "\\infty"
            mac[f"Time{k}"] = f"{P['times_s'][t]:.1f}"
        tf = P["tiers"].get("tf32")
        if tf:
            mac["RatioTf"] = f"{tf['rms'][-1] / max(abs(tf['bias'][-1]), 1e-300):.0f}"
        mac["TimeRef"] = f"{P['times_s']['fp64']:.1f}"

    if G:
        tiers = sorted(set(g["tier"] for g in G), key=lambda t: ["fp32", "tf32"].index(t))
        Ns = sorted(set(g["N"] for g in G))
        show = [Ns[0], Ns[-1]] if len(Ns) > 1 else Ns
        fig, axes = plt.subplots(1, len(show), figsize=(6.4, 2.5), sharey=True)
        axes = axes if len(show) > 1 else [axes]
        tier = tiers[-1]
        for ax, N in zip(axes, show):
            rows = [g for g in G if g["N"] == N and g["tier"] == tier]
            a = [g["alpha"] for g in rows]
            ax.plot(a, [100 * g["loop_eff"] for g in rows], color=SER["fp32"], marker="o", ms=4)
            ax.plot(a, [100 * g["padded_eff"] for g in rows], color=SER["3xtf32"], marker="o", ms=4)
            yl, yp = 100 * rows[-1]["loop_eff"], 100 * rows[-1]["padded_eff"]
            if abs(yl - yp) < 14:  # keep the two direct labels apart
                mid = (yl + yp) / 2
                yl, yp = (mid + 7, mid - 7) if yl >= yp else (mid - 7, mid + 7)
            ax.text(a[-1] + 0.08, yl, "per-group\nGEMM loop", fontsize=7, va="center", color=INK)
            ax.text(a[-1] + 0.08, yp, "padded\nbatched GEMM", fontsize=7, va="center", color=INK)
            ax.axhline(100, color=INK2, lw=0.8, ls=":")
            ax.set_title(f"N = {N:,} trajectories, {rows[0]['J']} terms ({tier.upper()})", fontsize=8, color=INK)
            ax.set_xlabel(r"group-size skew $\alpha$  ($p_j\propto j^{-\alpha}$)")
            ax.set_xlim(-0.1, a[-1] + 0.95)
            ax.set_ylim(0, 110)
        axes[0].set_ylabel("% of single-GEMM throughput")
        fig.savefig(os.path.join(outdir, "fig_grouped.pdf"))
        plt.close(fig)
        big = [g for g in G if g["N"] == Ns[-1]]
        for t in tiers:
            rows = [g for g in big if g["tier"] == t]
            k = {"fp32": "Fp", "tf32": "Tf"}[t]
            mac[f"Single{k}"] = f"{max(g['single_gemm_tflops'] for g in rows):.2f}"
            mac[f"LoopMin{k}"] = f"{100 * min(g['loop_eff'] for g in rows):.0f}"
            mac[f"PadMin{k}"] = f"{100 * min(g['padded_eff'] for g in rows):.0f}"
            mac[f"PadMax{k}"] = f"{100 * max(g['padded_eff'] for g in rows):.0f}"
        effs = [e for g in big for e in (g["loop_eff"], g["padded_eff"])]
        mac["GapMin"] = f"{100 * min(effs):.0f}"
        mac["GapMax"] = f"{100 * max(effs):.0f}"
        mac["BenchDtwo"] = G[0]["D2"]
        mac["BenchJ"] = G[0]["J"]
        mac["BenchNmax"] = f"{Ns[-1]:,}".replace(",", "{,}")

    bqp = os.path.join(os.path.dirname(path), "results_bluequbit.json")
    bqt = os.path.join(outdir, "bq.tex")
    if os.path.exists(bqp):
        B = json.load(open(bqp))
        names = {"gpu": "GPU statevector", "mps.gpu": "GPU MPS", "pauli-path": "Pauli-path", "cpu": "CPU statevector"}
        rows = "\\\\\n".join(f"{names.get(d, d)} & ${sci(v['max_abs_diff'])}$ " for d, v in B["devices"].items())
        with open(bqt, "w") as f:
            f.write("\\subsection*{Cross-check on the BlueQubit platform}\n"
                    f"The first {B['K']} sampled trajectories ({B['n']} qubits, {B['depth_gates']} rotations each) were rebuilt "
                    "as Qiskit circuits and evaluated on three BlueQubit simulators, using the promo credits. "
                    "Table~\\ref{tab:bq} lists the largest deviation from the local FP64 values.\n"
                    "\\begin{table}[h]\\centering\\small\\caption{Max $|\\langle\\bar Z\\rangle_{\\rm BlueQubit}-\\langle\\bar Z\\rangle_{\\rm local\\ FP64}|$ "
                    f"over {B['K']} trajectories.}}\\label{{tab:bq}}\n"
                    "\\begin{tabular}{lc}\\toprule Backend & max abs.\\ difference\\\\\\midrule\n"
                    + rows + "\\\\\\bottomrule\\end{tabular}\\end{table}\n")
    elif os.path.exists(bqt):
        os.remove(bqt)
    url = os.environ.get("CODE_URL", "")
    mac["CodeLine"] = f"\\textbf{{Code and raw data:}} \\url{{{url}}}" if url else ""

    with open(os.path.join(outdir, "macros.tex"), "w") as f:
        for k, v in mac.items():
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    print("figures + macros ->", outdir)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "report")
