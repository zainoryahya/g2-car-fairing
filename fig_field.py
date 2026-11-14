#!/usr/bin/env python3
"""Spatial map of local degradation max(0, c_q(t) - c_q(0)) over the parameter domain, full vs smooth (L-BFGS-B).
Usage: python3 fig_field.py results SID DELTA_MM [paper_dir]"""
import json, sys, os, subprocess
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT, SID, DMM = sys.argv[1], sys.argv[2], int(sys.argv[3]); PAPER = sys.argv[4] if len(sys.argv) > 4 else None
rows = [json.loads(l) for l in open(os.path.join(OUT, "runs.jsonl"))]
p = subprocess.Popen(["node", os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "evalserver.js")], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
def call(**m): p.stdin.write(json.dumps(m) + "\n"); p.stdin.flush(); return json.loads(p.stdout.readline())
call(cmd="setup", sid=SID, delta_mm=DMM, space="full")
maps = {}
for sp in ["full", "reduced"]:
    r = [x for x in rows if x["sid"] == SID and x["delta_mm"] == DMM and x["algo"] == "lbfgsb" and x["space"] == sp][0]
    f = call(cmd="field", t=r["t"]); NU, NV = f["NU"], f["NV"]
    C, C0 = np.array(f["C"]).reshape(NU, NV), np.array(f["C0"]).reshape(NU, NV)
    # density per unit parameter area: integrates over the sampled domain to J_up (contributions carry du*dv in J)
    maps[sp] = (np.maximum(0, C - C0) / f["J0"], np.maximum(0, C0 - C) / f["J0"], r)
    dudv = (0.86 / (NU - 1)) * (0.86 / (NV - 1))
    print(sp, "check: integral", float(maps[sp][0].sum() * dudv), "vs J_up", r["J_up"])
p.stdin.close()
vmax = max(np.percentile(m[0], 99.7) for m in maps.values())
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 8.5})
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.6), sharey=True)
for ax, (sp, lab) in zip(axs, [("full", "full net"), ("reduced", "smooth space")]):
    up, down, r = maps[sp]
    im = ax.imshow(up.T, origin="lower", extent=[0.07, 0.93, 0.07, 0.93], aspect="auto", cmap="Reds", vmin=0, vmax=vmax)
    ax.set_title(f"{lab}: J_up/J₀ = {r['J_up']:.4f},  J_val/J₀ = {r['J_val']:.3f}", fontsize=8, loc="left")
    ax.set_xlabel("u  (front → rear)")
axs[0].set_ylabel("v  (left rocker → roof → right rocker)")
cb = fig.colorbar(im, ax=axs, fraction=0.025, pad=0.02); cb.set_label("added energy density / J₀", fontsize=7.5)
fig.savefig(os.path.join(OUT, "fig_field.pdf"), bbox_inches="tight"); fig.savefig(os.path.join(OUT, "fig_field.png"), dpi=220, bbox_inches="tight")
if PAPER: os.system(f"cp {os.path.join(OUT, 'fig_field.pdf')} {PAPER}/figs/")

