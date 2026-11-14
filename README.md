# g2-car-fairing

Code and data for the paper

> **Integral fairness is not enough: deviation-bounded fairing of G² B-spline car-body surfaces in a smooth reduced design space**
> Zainor Ridzuan Yahya, Universiti Malaysia Perlis (UniMAP). *Submitted.*

The repository contains:
- four bicubic B-spline car-body test surfaces;
- a curvature-variation fairness objective and a one-sided *local degradation* measure;
- a smooth reduced design space;
- five optimisers: linear thin-plate, L-BFGS-B, jDE, PSO and GWO;
- the raw results of all 2196 optimisation runs;
- the statistical analysis that produces every table, figure and number in the paper;
- an interactive browser studio for inspecting the surfaces (zebra, isophotes, curvature maps, curvature combs, live fairing).

## Main findings
- Optimising every control point independently gives the lowest integral energy but leaves local defects. At equal integral energy, the smooth reduced space (36 variables) has **10–47× less local degradation** (median 23×) on all four surfaces.
- Linear thin-plate fairing hardly reduces curvature variation under a tight deviation bound.
- On this smooth objective, **L-BFGS-B ranks first in every instance**. jDE comes close, PSO stagnates early and GWO converges slowly.

## Requirements
- Node.js ≥ 18 (no npm packages needed)
- Python ≥ 3.9 with numpy, scipy, pandas and matplotlib: `pip install -r requirements.txt`

## Quick start
```bash
make smoke        # ~10 s: one run of each optimiser on S1, delta = 10 mm
make analysis     # minutes: rebuilds all tables, figures and numbers from the shipped raw results into tex/
make experiment   # ~1 h on 2 cores: re-runs every optimisation from scratch (overwrites results/)
```
Without `make`, run the commands listed in the `Makefile` directly.

## Repository layout
| Path | Contents |
|---|---|
| `src/geom.js` | B-spline basis and derivatives (Piegl & Tiller, Alg. A2.3); car control-net construction; mean and Gaussian curvature; G¹/G² continuity check |
| `src/opt.js` | free variables (normal offsets at Greville points), curvature-variation objective, local degradation `J_up`, thin-plate solver, smooth reduced space, jDE / PSO / GWO |
| `src/problems.js` | the four test surfaces S1–S4 and the evaluation metrics |
| `src/evalserver.js` | line-delimited JSON server exposing the objective to Python (used by L-BFGS-B) |
| `run.js` | parallel runner for the metaheuristics and the thin-plate baseline |
| `lbfgs.py` | L-BFGS-B baseline (SciPy) on the same objective and evaluation budget; also the δ sweep |
| `analyze.py` | Friedman, Mann–Whitney U and Wilcoxon tests (Holm-corrected), Vargha–Delaney A12, LaTeX tables, figures |
| `pareto_plot.py`, `fig_field.py`, `extra_numbers.py` | trade-off curves, spatial maps of local degradation, extra macros for the text |
| `results/runs.jsonl` | one JSON line per run: configuration, all metrics, convergence curve and the best offset vector `t` |
| `results/pareto.jsonl` | L-BFGS-B δ sweep (2–40 mm) |
| `docs/index.html` | interactive surface studio (open locally or via GitHub Pages) |

## Problem in brief
- **Surface:** clamped uniform bicubic B-spline, 13 control points across the body and 30–36 along it. It is C² (hence G²) across all interior knots.
- **Variables:** each free control point moves by `t_ij` along the surface normal at its Greville point. Offsets are mirror-symmetric. The bumper rows and the rocker/wheel-arch edge are fixed. A box `|t_ij| ≤ δ` bounds the surface displacement by δ (partition of unity).
- **Smooth reduced space:** `t = Φc`, where Φ is a 9 × 4 coarse cubic B-spline over the control-index grid (36 variables).
- **Objective:** `∫|∇H|² dA`, discretised on a 64 × 32 grid and validated on an independent 128 × 64 grid.
- **Local degradation:** `J_up`, the energy added wherever the per-sample contribution increased, relative to `J(0)`.

## Test surfaces
| ID | Style | L (m) | W (m) | H (m) | Control net | D (full) | D (smooth) |
|---|---|---|---|---|---|---|---|
| S1 | Sedan (notchback) | 4.60 | 1.82 | 1.44 | 32 × 13 | 156 | 36 |
| S2 | Fastback coupe | 4.70 | 1.86 | 1.38 | 32 × 13 | 156 | 36 |
| S3 | Tall hatch | 4.30 | 1.80 | 1.60 | 30 × 13 | 144 | 36 |
| S4 | Long GT | 4.95 | 1.92 | 1.32 | 36 × 13 | 180 | 36 |

## Interactive studio
Open `docs/index.html` in a browser, or enable GitHub Pages on the `docs/` folder to get
`https://zainoryahya.github.io/g2-car-fairing/`. The studio loads three.js from a CDN.

## Citation
See `CITATION.cff`. Once the paper is published, please cite the article.

## Licence
Code: MIT (see `LICENSE`). Raw results in `results/`: CC BY 4.0.
