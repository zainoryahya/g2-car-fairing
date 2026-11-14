#!/usr/bin/env python3
"""Analysis for the multi-surface, multi-delta fairing study.
Usage: python3 analyze.py results [paper_dir]
Writes into results_dir: summary_v2.csv, tests_v2.csv, report_v2.md, fig_*.pdf/png
and (if paper_dir given) LaTeX tables + numbers.tex macros into paper_dir."""
import json, sys, os
import numpy as np, pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = sys.argv[1] if len(sys.argv) > 1 else "results"
PAPER = sys.argv[2] if len(sys.argv) > 2 else None
cfg = json.load(open(os.path.join(OUT, "config.json")))
df = pd.DataFrame([json.loads(l) for l in open(os.path.join(OUT, "runs.jsonl"))])
df["inst"] = df.sid + "/" + df.delta_mm.astype(int).astype(str)

ALG = {"tp": "TP", "lbfgsb": "L-BFGS-B", "jde": "jDE", "pso": "PSO", "gwo": "GWO"}
SPC = {"full": "F", "reduced": "S"}
df["config"] = df.algo.map(ALG) + "-" + df.space.map(SPC)
CONFIGS = ["TP-F", "L-BFGS-B-F", "L-BFGS-B-S", "jDE-F", "jDE-S", "PSO-F", "PSO-S", "GWO-F", "GWO-S"]
CONFIGS = [c for c in CONFIGS if c in set(df.config)]
STOCH = [c for c in CONFIGS if c.split("-")[0] in ("jDE", "PSO", "GWO")]
INSTS = [f"{s}/{d}" for s in cfg["surfaces"] for d in cfg["deltas"] if f"{s}/{d}" in set(df.inst)]
COL = {"jDE": "#2a78d6", "PSO": "#eb6834", "GWO": "#1baf7a", "L-BFGS-B": "#6250d6", "TP": "#52514e"}

def a12(x, y):
    x, y = np.asarray(x), np.asarray(y)
    return ((x[:, None] < y[None, :]).sum() + 0.5 * (x[:, None] == y[None, :]).sum()) / (len(x) * len(y))
def holm(p):
    p = np.asarray(p, float); idx = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for r, i in enumerate(idx): run = max(run, min(1, (m - r) * p[i])); adj[i] = run
    return adj

# ---------- medians per instance × config ----------
med = {m: df.groupby(["inst", "config"])[m].median().unstack()[CONFIGS].loc[INSTS] for m in ["J_val", "J_up", "J_bend", "rough_mm", "sec"]}
for m, t in med.items(): t.to_csv(os.path.join(OUT, f"median_{m}.csv"))

# ---------- Friedman across instances (blocks) on medians ----------
fried = {}
for m in ["J_val", "J_up"]:
    M = med[m].values
    st, p = stats.friedmanchisquare(*[M[:, j] for j in range(M.shape[1])])
    ranks = pd.DataFrame(M, columns=CONFIGS).rank(axis=1).mean().sort_values()
    fried[m] = (st, p, ranks)

# ---------- per-instance tests ----------
rows = []
for inst in INSTS:
    d = df[df.inst == inst]
    for a in ["jDE", "PSO", "GWO"]:
        for m in ["J_val", "J_up", "rough_mm"]:
            x = d[d.config == f"{a}-S"][m].values; y = d[d.config == f"{a}-F"][m].values
            if len(x) and len(y):
                U, p = stats.mannwhitneyu(x, y, alternative="two-sided")
                rows.append({"inst": inst, "test": f"{a} S vs F", "metric": m, "p": p, "A12": a12(x, y)})
    # each stochastic config vs deterministic L-BFGS-B in the same space (one-sample signed-rank)
    for c in STOCH:
        sp = c.split("-")[1]; ref = d[d.config == f"L-BFGS-B-{sp}"]
        if len(ref):
            for m in ["J_val", "J_up"]:
                v = d[d.config == c][m].values - ref[m].values[0]
                if np.all(v == 0): continue
                W, p = stats.wilcoxon(v, alternative="two-sided")
                rows.append({"inst": inst, "test": f"{c} vs L-BFGS-B-{sp}", "metric": m, "p": p, "A12": float(np.mean(v < 0) + 0.5 * np.mean(v == 0)), "median_diff": float(np.median(v))})
T = pd.DataFrame(rows)
T["p_holm"] = np.nan
for (test, metric), g in T.groupby(["test", "metric"]):
    T.loc[g.index, "p_holm"] = holm(g.p.values)
def wtl(g, alpha=0.05):  # win/tie/loss for the first-named group (lower metric better)
    w = ((g.p_holm < alpha) & (g.A12 > 0.5)).sum(); l = ((g.p_holm < alpha) & (g.A12 < 0.5)).sum()
    return int(w), int(len(g) - w - l), int(l)
T.to_csv(os.path.join(OUT, "tests_v2.csv"), index=False)
summ = []
for (test, metric), g in T.groupby(["test", "metric"], sort=False):
    w, t_, l = wtl(g); summ.append({"test": test, "metric": metric, "win": w, "tie": t_, "loss": l, "A12_median": g.A12.median()})
S = pd.DataFrame(summ); S.to_csv(os.path.join(OUT, "wtl_v2.csv"), index=False)

# ---------- figures ----------
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#888",
                     "xtick.color": "#444", "ytick.color": "#444", "axes.labelcolor": "#222"})
def save(fig, name):
    fig.tight_layout(); fig.savefig(os.path.join(OUT, name + ".pdf")); fig.savefig(os.path.join(OUT, name + ".png"), dpi=220); plt.close(fig)
    if PAPER: fig_dir = os.path.join(PAPER, "figs"); os.makedirs(fig_dir, exist_ok=True); os.system(f"cp {os.path.join(OUT, name + '.pdf')} {fig_dir}/")

def style(c):
    a, s = c.rsplit("-", 1); return COL[a], ("o" if s == "F" else "s"), (s == "S")

# Fig: trade-off (J_val vs J_up) per instance, medians
fig, axs = plt.subplots(len(cfg["deltas"]), len(cfg["surfaces"]), figsize=(7.2, 1.85 * len(cfg["deltas"]) + 0.5), sharex=False, sharey=False)
axs = np.atleast_2d(axs)
for i, dmm in enumerate(cfg["deltas"]):
    for j, sid in enumerate(cfg["surfaces"]):
        ax = axs[i, j]; inst = f"{sid}/{dmm}"
        if inst not in INSTS: ax.axis("off"); continue
        for c in CONFIGS:
            col, mk, filled = style(c)
            ax.scatter(med["J_val"].loc[inst, c], max(med["J_up"].loc[inst, c], 1e-5), s=26, marker=mk, facecolor=col if filled else "white", edgecolor=col, lw=1.2, zorder=3)
        ax.set_yscale("log"); ax.grid(color="#eee", lw=0.7); ax.set_axisbelow(True)
        ax.set_title(f"{sid}, δ = {dmm} mm", fontsize=8, loc="left")
        if i == len(cfg["deltas"]) - 1: ax.set_xlabel("J_val / J₀")
        if j == 0: ax.set_ylabel("J_up / J₀ (log)")
from matplotlib.lines import Line2D
h = [Line2D([], [], marker=style(c)[1], ls="", mfc=style(c)[0] if style(c)[2] else "white", mec=style(c)[0], label=c) for c in CONFIGS]
fig.legend(handles=h, loc="lower center", ncol=len(CONFIGS), frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.01))
fig.subplots_adjust(bottom=0.12); save(fig, "fig_tradeoff")

# Fig: convergence on representative instance(s)
rep = [f"S1/{cfg['deltas'][len(cfg['deltas']) // 2]}"]
for inst in rep:
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.7), sharey=True)
    x = np.linspace(0, cfg["budget"], cfg["checkpoints"] + 1)
    for ax, sp in zip(axs, ["F", "S"]):
        for a in ["jDE", "PSO", "GWO", "L-BFGS-B"]:
            d = df[(df.inst == inst) & (df.config == f"{a}-{sp}")]
            if not len(d): continue
            C = np.array(d.curve.tolist()); m_ = np.median(C, 0)
            if len(C) > 1: ax.fill_between(x, np.percentile(C, 25, 0), np.percentile(C, 75, 0), color=COL[a], alpha=0.15, lw=0)
            ax.plot(x, m_, color=COL[a], lw=1.8, ls="--" if a == "L-BFGS-B" else "-", label=a)
        D = int(df[(df.inst == inst) & (df.space == ("full" if sp == "F" else "reduced"))].D.iloc[0])
        ax.set_title(f"{inst.replace('/', ', δ = ')} mm — {'full net' if sp == 'F' else 'smooth space'} (D = {D})", fontsize=8, loc="left")
        ax.set_xlabel("function evaluations"); ax.grid(color="#eee", lw=0.7)
    axs[0].set_ylabel("penalised objective"); axs[1].legend(frameon=False, fontsize=7.5)
    save(fig, "fig_convergence_v2")

# Fig: J_up boxplots across all instances for jDE F vs S and L-BFGS-B
fig, ax = plt.subplots(figsize=(7.2, 2.6))
pos = 0; ticks = []; labels = []
for inst in INSTS:
    for c, off in [("jDE-F", -0.22), ("jDE-S", 0.22)]:
        v = df[(df.inst == inst) & (df.config == c)].J_up.values
        b = ax.boxplot([np.maximum(v, 1e-5)], positions=[pos + off], widths=0.36, patch_artist=True, showfliers=False,
                       medianprops={"color": "#111"}, whiskerprops={"color": "#777"}, capprops={"color": "#777"})
        col = COL["jDE"]; b["boxes"][0].set_facecolor(col if c.endswith("S") else "white"); b["boxes"][0].set_edgecolor(col)
        if c.endswith("F"): b["boxes"][0].set_hatch("////")
    for c, mk in [("L-BFGS-B-F", "o"), ("L-BFGS-B-S", "s")]:
        v = df[(df.inst == inst) & (df.config == c)].J_up.values
        if len(v): ax.scatter(pos, max(v[0], 1e-5), marker=mk, facecolor="white" if c.endswith("F") else COL["L-BFGS-B"], edgecolor=COL["L-BFGS-B"], s=22, zorder=4)
    ticks.append(pos); labels.append(inst.replace("/", "\nδ=")); pos += 1
ax.set_yscale("log"); ax.set_xticks(ticks); ax.set_xticklabels(labels, fontsize=7); ax.set_ylabel("J_up / J₀ (log)"); ax.grid(axis="y", color="#eee")
h = [plt.Rectangle((0, 0), 1, 1, fc="white", ec=COL["jDE"], hatch="////", label="jDE full"), plt.Rectangle((0, 0), 1, 1, fc=COL["jDE"], ec=COL["jDE"], label="jDE smooth"),
     Line2D([], [], marker="o", ls="", mfc="white", mec=COL["L-BFGS-B"], label="L-BFGS-B full"), Line2D([], [], marker="s", ls="", mfc=COL["L-BFGS-B"], mec=COL["L-BFGS-B"], label="L-BFGS-B smooth")]
ax.legend(handles=h, frameon=False, fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.16))
save(fig, "fig_jup_all")

# ---------- report ----------
R = ["# Multi-surface fairing study — results", "",
     f"Surfaces {cfg['surfaces']}, δ ∈ {cfg['deltas']} mm, budget {cfg['budget']}, {cfg['runs']} runs per stochastic configuration.", ""]
for m in ["J_val", "J_up"]:
    st, p, ranks = fried[m]
    R += [f"## {m}: Friedman χ² = {st:.1f}, p = {p:.2e} (blocks = {len(INSTS)} instances)", "", "Average ranks: " + ", ".join(f"{k} {v:.2f}" for k, v in ranks.items()), ""]
    R += ["| inst | " + " | ".join(CONFIGS) + " |", "|---" * (len(CONFIGS) + 1) + "|"]
    for inst in INSTS:
        R.append(f"| {inst} | " + " | ".join(f"{med[m].loc[inst, c]:.4f}" for c in CONFIGS) + " |")
    R.append("")
R += ["## Win/tie/loss across instances (Holm-corrected per test family, α = 0.05; first-named group, lower is better)", "",
      "| test | metric | W/T/L | median A12 |", "|---|---|---|---|"]
for _, r in S.iterrows(): R.append(f"| {r.test} | {r.metric} | {r.win}/{r.tie}/{r.loss} | {r.A12_median:.2f} |")
open(os.path.join(OUT, "report_v2.md"), "w").write("\n".join(R))
print("\n".join(R))

# ---------- LaTeX tables + macros ----------
if PAPER:
    os.makedirs(PAPER, exist_ok=True)
    def table(m, cap, lab, k):
        t = med[m]; L = [r"\begin{table*}[t]\centering\footnotesize", r"\caption{" + cap + "}", r"\label{" + lab + "}",
                         r"\begin{tabular}{l" + "r" * len(CONFIGS) + "}", r"\toprule",
                         "Instance & " + " & ".join(c.replace("L-BFGS-B", "LB") for c in CONFIGS) + r"\\", r"\midrule"]
        for inst in INSTS:
            row = t.loc[inst]; best = row.min()
            cells = [(r"\textbf{%s}" if abs(v - best) <= 1e-12 * max(1, abs(best)) else "%s") % (f"{v:.{k}f}") for v in row.values]
            sid, d = inst.split("/"); L.append(f"{sid}, $\\delta={d}$ & " + " & ".join(cells) + r"\\")
        st, p, ranks = fried[m]
        L += [r"\midrule", "Avg.\\ rank & " + " & ".join(f"{ranks[c]:.2f}" for c in CONFIGS) + r"\\", r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
        open(os.path.join(PAPER, f"table_{m}.tex"), "w").write("\n".join(L))
    table("J_val", r"Median relative curvature variation $J_{val}/J_0$ on the fine validation grid (lower is better; best per row in bold). F = full net, S = smooth reduced space, LB = L-BFGS-B, TP = linear thin-plate. Stochastic methods: median of %d runs." % cfg["runs"], "tab:jval", 3)
    table("J_up", r"Median local degradation $J_{up}/J_0$ (energy added where $|\nabla H|^2$ increased; lower is better).", "tab:jup", 4)
    # macros
    def mac(name, val): return f"\\newcommand{{\\{name}}}{{{val}}}\n"
    s = ""
    s += mac("NRuns", cfg["runs"]) + mac("Budget", cfg["budget"]) + mac("NInst", len(INSTS))
    for m in ["J_val", "J_up"]:
        st, p, ranks = fried[m]; key = "Jval" if m == "J_val" else "Jup"
        s += mac(f"Fried{key}Chi", f"{st:.1f}") + mac(f"Fried{key}P", f"{p:.1e}".replace("e-0", "e-").replace("e", r"\times10^{") + "}")
        for c in CONFIGS: s += mac(f"Rank{key}" + c.replace("-", "").replace("LBFGSB", "LB"), f"{ranks[c]:.2f}")
    # ratios jDE: full vs smooth medians of J_up and J_val (geometric mean over instances)
    gm = lambda a: float(np.exp(np.mean(np.log(a))))
    s += mac("JupRatioJDE", f"{gm(med['J_up']['jDE-F'] / med['J_up']['jDE-S']):.0f}")
    s += mac("JupRatioLB", f"{gm(med['J_up']['L-BFGS-B-F'] / med['J_up']['L-BFGS-B-S']):.0f}") if "L-BFGS-B-S" in CONFIGS else ""
    s += mac("JvalRedJDEF", f"{(1 - med['J_val']['jDE-F'].mean()) * 100:.1f}") + mac("JvalRedJDES", f"{(1 - med['J_val']['jDE-S'].mean()) * 100:.1f}")
    if "L-BFGS-B-F" in CONFIGS:
        s += mac("JvalRedLBF", f"{(1 - med['J_val']['L-BFGS-B-F'].mean()) * 100:.1f}") + mac("JvalRedLBS", f"{(1 - med['J_val']['L-BFGS-B-S'].mean()) * 100:.1f}")
    s += mac("JvalTPmean", f"{med['J_val']['TP-F'].mean():.3f}") + mac("JupTPmean", f"{med['J_up']['TP-F'].mean():.3f}")
    s += mac("SecMedian", f"{df[df.config.isin(STOCH)].sec.median():.1f}")
    for _, r in S.iterrows():
        key = r.test.replace(" ", "").replace("-", "").replace("LBFGSB", "LB").replace("vs", "Vs") + r.metric.replace("_", "").replace("mm", "")
        s += mac("WTL" + "".join(ch for ch in key if ch.isalpha()), f"{r.win}/{r.tie}/{r.loss}")
    open(os.path.join(PAPER, "numbers.tex"), "w").write(s)
    # surfaces table
    L = [r"\begin{table}[t]\centering\footnotesize", r"\caption{Test surfaces. Package parameters (m), control-net size $m\times n$ and number of free variables in the full ($D$) and smooth ($D_r$) design spaces.}", r"\label{tab:surfaces}",
         r"\begin{tabular}{llrrrrrrcrr}", r"\toprule", r"ID & Body style & $L$ & $W$ & $H$ & rake & rear & flare & $m\times n$ & $D$ & $D_r$\\", r"\midrule"]
    for sid, inf in cfg["surfaces_info"].items():
        d = inf["design"]
        L.append(f"{sid} & {inf['name']} & {d['L']:.2f} & {d['W']:.2f} & {d['H']:.2f} & {d['rake']:+.2f} & {d['rear']:.1f} & {d['flare']:.3f} & ${inf['ctrlNet'][0]}\\times{inf['ctrlNet'][1]}$ & {inf['freeVars']} & {inf['reducedVars']}\\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(PAPER, "table_surfaces.tex"), "w").write("\n".join(L))
