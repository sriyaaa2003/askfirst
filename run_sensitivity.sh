#!/bin/sh
# Re-runs the study with the cost assumptions scaled, to show how much the conclusions depend on them.
export PYTHONPATH=src
python -m askfirst.experiment --out results/sens_margin_half --margin-scale 0.5
python -m askfirst.experiment --out results/sens_margin_double --margin-scale 2
python -m askfirst.experiment --out results/sens_over_half --over-scale 0.5
python -m askfirst.experiment --out results/sens_over_double --over-scale 2
