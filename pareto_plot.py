#!/usr/bin/env python3
"""Trade-off curves J_val vs J_up for L-BFGS-B over a delta sweep, full vs smooth space.
Usage: python3 pareto_plot.py results [paper_dir]"""
import json, sys, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT = sys.argv[1]; PAPER = sys.argv[2] if len(sys.argv) > 2 else None
df = pd.DataFrame([json.loads(l) for l in open(os.path.join(OUT, "pareto.jsonl"))])
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#888"})
fig, axs = plt.subplots(1, 4, figsize=(7.4, 2.35), sharey=True)
res = []
for ax, sid in zip(axs, sorted(df.sid.unique())):
    for sp, col, mk, lab in [("full", "#6250d6", "o", "full net"), ("reduced", "#1baf7a", "s", "smooth space")]:
        d = df[(df.sid == sid) & (df.space == sp)].sort_values("delta_mm")
        ax.plot(d.J_val, d.J_up, color=col, lw=1.5, marker=mk, ms=3.5, mfc="white" if sp == "full" else col, label=lab)
    # J_up of each space interpolated at common J_val levels (log-linear)
    f = df[(df.sid == sid) & (df.space == "full")].sort_values("J_val"); r = df[(df.sid == sid) & (df.space == "reduced")].sort_values("J_val")
    lo, hi = max(f.J_val.min(), r.J_val.min()), min(f.J_val.max(), r.J_val.max())
    for jv in np.linspace(lo, hi, 7)[1:-1]:
        uf = np.exp(np.interp(jv, f.J_val, np.log(f.J_up))); ur = np.exp(np.interp(jv, r.J_val, np.log(r.J_up)))
        res.append({"sid": sid, "J_val": jv, "Jup_full": uf, "Jup_smooth": ur, "ratio": uf / ur})
    ax.set_yscale("log"); ax.set_title(sid, fontsize=8.5, loc="left"); ax.set_xlabel("J_val / J₀"); ax.grid(color="#eee", lw=0.7)
axs[0].set_ylabel("J_up / J₀ (log)"); h, l = axs[0].get_legend_handles_labels(); fig.legend(h, l, frameon=False, fontsize=7.5, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.02))
fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig(os.path.join(OUT, "fig_pareto.pdf")); fig.savefig(os.path.join(OUT, "fig_pareto.png"), dpi=220)
R = pd.DataFrame(res); R.to_csv(os.path.join(OUT, "pareto_ratios.csv"), index=False)
print(R.round(4).to_string()); print("ratio min/median/max:", R.ratio.min().round(1), R.ratio.median().round(1), R.ratio.max().round(1))
if PAPER:
    os.system(f"cp {os.path.join(OUT, 'fig_pareto.pdf')} {PAPER}/figs/")
    with open(os.path.join(PAPER, "numbers.tex"), "a") as fh:
        fh.write(f"\\newcommand{{\\ParetoRatioMin}}{{{R.ratio.min():.0f}}}\n\\newcommand{{\\ParetoRatioMed}}{{{R.ratio.median():.0f}}}\n\\newcommand{{\\ParetoRatioMax}}{{{R.ratio.max():.0f}}}\n")
        d = df.groupby("space").evals.median()
        fh.write(f"\\newcommand{{\\LBevalsSmooth}}{{{int(d['reduced'])}}}\n")
