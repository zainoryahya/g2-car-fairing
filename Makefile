# Reproduce the study. `make analysis` re-derives every table, figure and number from the shipped raw results (minutes).
# `make experiment` re-runs all optimisations from scratch (~1 h on 2 cores) and overwrites results/.
PY ?= python3
OUT ?= results
TEX ?= tex

.PHONY: analysis experiment smoke clean
analysis:
	mkdir -p $(TEX)/figs
	$(PY) analyze.py $(OUT) $(TEX)
	$(PY) pareto_plot.py $(OUT) $(TEX)
	$(PY) extra_numbers.py $(OUT) $(TEX)
	$(PY) fig_field.py $(OUT) S1 10 $(TEX)

experiment:
	rm -rf $(OUT)
	node run.js --runs 30 --budget 6000 --deltas 5,10,20 --surfaces S1,S2,S3,S4 --algos jde,pso,gwo --spaces full,reduced --out $(OUT)
	$(PY) lbfgs.py $(OUT) 6000
	$(PY) lbfgs.py $(OUT) 6000 2,3,4,5,7,10,14,20,28,40 pareto.jsonl

smoke:
	rm -rf smoke
	node run.js --runs 1 --budget 300 --surfaces S1 --deltas 10 --out smoke
	$(PY) lbfgs.py smoke 300

clean:
	rm -rf smoke $(TEX) $(OUT)/*.png $(OUT)/*.pdf $(OUT)/*.csv $(OUT)/report_v2.md
