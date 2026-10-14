#!/usr/bin/env python3
"""
Cross-check sampled qDRIFT trajectories on the BlueQubit platform.

Rebuilds the first K sampled trajectories of the precision run as Qiskit circuits
(RZZ / RX rotations) and evaluates <Zbar> = (1/n) sum_i <Z_i> on three BlueQubit
backends (GPU statevector, GPU MPS, Pauli-path), comparing each with the fp64
value from qdrift_proto.py for the same trajectory.

  export BLUEQUBIT_API_TOKEN=...      # Account -> API token
  pip install bluequbit qiskit
  python bluequbit_check.py --results results_gpu --K 8
"""
import argparse
import json
import math
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results_gpu")
    ap.add_argument("--backend-json", default=None, help="results_<backend>.json (auto)")
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--devices", nargs="+", default=["gpu", "mps.gpu", "pauli-path"])
    ap.add_argument("--mps-bond", type=int, default=64)
    args = ap.parse_args()

    import bluequbit
    from qiskit import QuantumCircuit

    jpath = args.backend_json or next(
        os.path.join(args.results, f) for f in os.listdir(args.results) if f.startswith("results_") and f.endswith(".json"))
    P = json.load(open(jpath))["precision"]
    data = np.load(os.path.join(args.results, "precision_obs.npz"))
    seqs, ref = data["seqs"], data["fp64"][:, -1]
    n, tau = P["n"], P["tau"]

    terms = [("ZZ", (i, i + 1), -1.0) for i in range(n - 1)] + [("X", (i,), -1.0) for i in range(n)]
    pauli_sum = [("I" * (n - 1 - i) + "Z" + "I" * i, 1.0 / n) for i in range(n)]

    circuits = []
    for b in range(args.K):
        qc = QuantumCircuit(n)
        for j in seqs[b]:
            kind, qs, c = terms[int(j)]
            phi = 2 * tau * math.copysign(1.0, c)
            if kind == "ZZ":
                qc.rzz(phi, qs[0], qs[1])
            else:
                qc.rx(phi, qs[0])
        circuits.append(qc)

    bq = bluequbit.init(os.environ.get("BLUEQUBIT_API_TOKEN"))
    out = {"n": n, "K": args.K, "depth_gates": int(seqs.shape[1]), "fp64": ref[:args.K].tolist(), "devices": {}}
    for dev in args.devices:
        opts = {}
        if dev.startswith("mps"):
            opts["mps_bond_dimension"] = args.mps_bond
        if dev == "pauli-path":
            opts["pauli_path_truncation_threshold"] = None if n <= 13 else 1e-5
        res = bq.run(circuits, device=dev, pauli_sum=pauli_sum, options=opts or None, job_name=f"flywheel-qdrift-{dev}")
        res = res if isinstance(res, list) else [res]
        vals = [float(np.real(r.expectation_value)) for r in res]
        err = np.abs(np.array(vals) - ref[:args.K])
        out["devices"][dev] = {"values": vals, "max_abs_diff": float(err.max()),
                               "run_time_ms": [r.run_time_ms for r in res], "cost": [r.cost for r in res]}
        print(f"{dev:12s} max |BlueQubit - fp64 local| = {err.max():.2e}")
    with open(os.path.join(args.results, "results_bluequbit.json"), "w") as f:
        json.dump(out, f, indent=1, default=str)
    print("wrote", os.path.join(args.results, "results_bluequbit.json"))


if __name__ == "__main__":
    main()
