#!/usr/bin/env python3
"""Extra macros used in the text (appended to paper/numbers.tex). Usage: python3 extra_numbers.py results paper_dir"""
import json, sys, os
import numpy as np, pandas as pd
OUT, PAPER = sys.argv[1], sys.argv[2]
df = pd.DataFrame([json.loads(l) for l in open(os.path.join(OUT, "runs.jsonl"))])
med = lambda m: pd.read_csv(os.path.join(OUT, f"median_{m}.csv"), index_col=0)
jv, ju = med("J_val"), med("J_up")
lb = df[df.algo == "lbfgsb"]; lbs = lb[lb.space == "reduced"]; lbf = lb[lb.space == "full"]
r = ju["L-BFGS-B-F"] / ju["L-BFGS-B-S"]; gap = jv["L-BFGS-B-S"] - jv["L-BFGS-B-F"]; dj = (jv["jDE-S"] - jv["L-BFGS-B-S"]).abs()
M = {
 "LBevalsSmoothMin": int(lbs.evals.min()), "LBevalsSmoothMax": int(lbs.evals.max()), "LBsecSmoothMax": f"{lbs.sec.max():.1f}",
 "LBfullHitBudget": int((lbf.evals >= 6000).sum()), "LBsecFullMax": f"{lbf.sec.max():.1f}",
 "JupRatioLBMin": f"{r.min():.0f}", "JupRatioLBMax": f"{r.max():.0f}",
 "JvalGapMin": f"{gap.min():.3f}", "JvalGapMax": f"{gap.max():.3f}", "JvalGapMed": f"{gap.median():.3f}",
 "JdeLBsmoothMaxDiff": f"{dj.max():.4f}",
 "TPJvalMin": f"{jv['TP-F'].min():.3f}", "TPJvalMax": f"{jv['TP-F'].max():.3f}",
 "RoughFullMin": f"{df[(df.space=='full') & (df.algo!='tp')].groupby(['sid','delta_mm','algo']).rough_mm.median().min():.1f}",
 "RoughFullMax": f"{df[(df.space=='full') & (df.algo!='tp')].groupby(['sid','delta_mm','algo']).rough_mm.median().max():.1f}",
 "RoughSmoothMax": f"{df[df.space=='reduced'].groupby(['sid','delta_mm','algo']).rough_mm.median().max():.2f}",
 "MaxDevExcess": f"{(df.maxdev_mm - df.delta_mm).max():.0e}",
 "NTotalRuns": len(df),
}
with open(os.path.join(PAPER, "numbers.tex"), "a") as fh:
    for k, v in M.items(): fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print(M)
