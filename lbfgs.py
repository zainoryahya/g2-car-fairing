#!/usr/bin/env python3
"""Deterministic gradient-based baseline: scipy L-BFGS-B (finite-difference gradients, box |c| <= delta)
on exactly the same penalised objective and evaluation budget as the metaheuristics.
Usage: python3 lbfgs.py [results_dir] [budget] [deltas_mm e.g. 2,5,10] [outfile]   -> appends to results_dir/outfile (default runs.jsonl)"""
import json, subprocess, sys, os, time
import numpy as np
from scipy.optimize import minimize

OUT = sys.argv[1] if len(sys.argv) > 1 else "results"
BUDGET = int(sys.argv[2]) if len(sys.argv) > 2 else 6000
DELTAS = [float(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else None
OUTFILE = sys.argv[4] if len(sys.argv) > 4 else "runs.jsonl"
cfg = json.load(open(os.path.join(OUT, "config.json")))
here = os.path.dirname(os.path.abspath(__file__))

class Server:
    def __init__(self):
        self.p = subprocess.Popen(["node", os.path.join(here, "src", "evalserver.js")], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
    def call(self, **m):
        self.p.stdin.write(json.dumps(m) + "\n"); self.p.stdin.flush()
        return json.loads(self.p.stdout.readline())
    def close(self): self.p.stdin.close(); self.p.wait()

class Budget(Exception): pass

srv = Server()
K = cfg["checkpoints"]
with open(os.path.join(OUT, OUTFILE), "a") as fout:
    for sid in cfg["surfaces"]:
        for dmm in (DELTAS or cfg["deltas"]):
            for space in cfg["spaces"]:
                Dm = srv.call(cmd="setup", sid=sid, delta_mm=dmm, space=space)["Dm"]
                d = dmm / 1000
                st = {"n": 0, "best": None, "bestF": np.inf, "hist": []}
                def F(c):
                    if st["n"] >= BUDGET: raise Budget()
                    v = srv.call(cmd="F", c=c.tolist())["F"]; st["n"] += 1
                    if v < st["bestF"]: st["bestF"], st["best"] = v, c.copy()
                    st["hist"].append((st["n"], st["bestF"]))
                    return v
                t0 = time.time()
                try:
                    minimize(F, np.zeros(Dm), method="L-BFGS-B", bounds=[(-d, d)] * Dm,
                             options={"maxfun": BUDGET * 10, "maxiter": 100000, "ftol": 1e-14, "gtol": 1e-12, "eps": 1e-7})
                except Budget:
                    pass
                sec = time.time() - t0
                curve, h, cur = [], 0, st["hist"][0][1]
                for k in range(K + 1):
                    e = BUDGET * k / K
                    while h < len(st["hist"]) and st["hist"][h][0] <= e: cur = st["hist"][h][1]; h += 1
                    curve.append(round(cur, 6))
                met = srv.call(cmd="metrics", c=st["best"].tolist())
                rec = {"sid": sid, "delta_mm": dmm, "algo": "lbfgsb", "space": space, "seed": 0, "D": Dm, "evals": st["n"],
                       "sec": round(sec, 3), "F_pen": st["bestF"], **met, "curve": curve}
                fout.write(json.dumps(rec) + "\n"); fout.flush()
                print(sid, dmm, space, "evals", st["n"], "J_val %.3f J_up %.4f rough %.2f" % (met["J_val"], met["J_up"], met["rough_mm"]), "%.1fs" % sec, flush=True)
srv.close()
