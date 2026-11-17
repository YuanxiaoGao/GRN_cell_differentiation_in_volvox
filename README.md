# GRN_cell_differentiation_in_volvox
Code for a four-variable model of cell-size-driven germ-soma differentiation in Volvox carteri.
This repository contains the Python code used to generate the main figures and supporting information for the manuscript:

> **How cell size breaks germ-soma symmetry in a gene regulatory network: a mathematical model of _Volvox carteri_ cell differentiation**

The code implements a four-variable ordinary differential equation model for the cell-size signal `S`, the HSP70A module `H`, the germ determinant `lag` (`L`), and the somatic repressor `regA` (`R`). It covers bifurcation analysis, full-dynamics simulations, Morris screening, Sobol sensitivity analysis, and data-constrained predictions of the `lag` expression ratio.

Repository structure
code/
|-- fig2/
|   `-- fig31_3x3.py
|-- fig3/
|   |-- Fig32.py
|   `-- Fig32_sobol.py
|-- fig4/
|   |-- fig4_final.py
|   |-- fig4_selfcheck.py
|   |-- model_v7.py
|   |-- plot_Htrace_v11.py
|   |-- check_exchange.py
|   `-- check_table1_allfigs.py
|-- fig5/
|   `-- fig5_v4.py
`-- si/
    |-- fig31_SI.py
    |-- sref_window_stats.py
    |-- sref_ab_scan.py
    |-- make_si_fig2_composite.py
    |-- sref_vs_fixedS.py
    `-- check_nan_repair.py
## Requirements
The scripts were written for Python 3 and require the following packages:
```text
numpy
scipy
matplotlib
SALib
```
Install them from the repository root with:
```bash
python -m pip install numpy scipy matplotlib SALib
```
All scripts should be run from the repository root so that imports and relative paths resolve consistently.
## Quick start
To regenerate the final main figures:

```bash
python fig2/fig31_3x3.py
python fig3/Fig32.py
python fig4/fig4_final.py --n 4096 --nproc 8 --seed 20241011
python fig5/fig5_v4.py --recompute
```
The most expensive step is `fig4/fig4_final.py`. If a cached `fig4-final.json` is already available, redraw the figure with:
```bash
python fig4/fig4_final.py --plot-only
```
If a cached `fig5` result is available, redraw the figure with:
```bash
python fig5/fig5_v4.py --plot-only
```
## Script guide

### Main figures

| Script                         | Purpose                                                                                                                                                                                                               | Manuscript item                                              |
| ------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------ |
| `fig2/fig31_3x3.py`            | Solves the saddle-node bifurcations at `S_SN1 ~= 0.318` and `S_SN2 ~= 1.637`, then generates the phase diagrams, full dynamics, and fate-separation panels. The script produces all nine subpanels in one run.        | Fig. 2A-2C (nine subpanels)                                  |
| `fig3/Fig32.py`                | Computes Morris elementary effects for 15 parameters and plots `mu*` (effect strength) against `sigma` (nonlinearity/interaction) to identify parameters retained for Sobol analysis.                                 | Fig. 3                                                       |
| `fig3/Fig32_sobol.py`          | Runs a reduced Sobol analysis using the parameters retained by Morris screening. This is an earlier check that the screened parameters are indeed unimportant; its numerical values are not used in the final Fig. 4. | Precursor to Fig. 4, not cited numerically in the manuscript |
| `fig4/fig4_final.py`           | Main Sobol workflow for Fig. 4. Uses Saltelli sampling with `N=4096` and seed `20241011` to calculate `S1` and `ST`, then generates the four-panel main figure and four inset panels.                                 | Fig. 4                                                       |
| `fig4/fig4_selfcheck.py`       | Reads `fig4-final.json` and runs four types of checks: `L <-> R` exchange symmetry, `sum(ST) ~= 1`, out-of-range Sobol indices, and label overlap.                                                                    | Reliability checks supporting Fig. 4                         |
| `fig4/check_table1_allfigs.py` | Extracts the 15 model parameters from the four plotting scripts, compares them with Table 1 of the manuscript, and reports differences.                                                                               | Consistency between Table 1 and all figures                  |
| `fig4/plot_Htrace_v11.py`      | Plots the full `H(t)` trajectory and marks the peak and plateau, providing the peak-to-plateau value `6.03` used in Fig. 4B.                                                                                          | Source of the peak-to-plateau value in Fig. 4B               |
| `fig4/model_v7.py`             | Model backend that splits the merged parameter `K` into seven independent half-saturation constants and tests whether these thresholds should be treated separately. Also used by the SI scripts.                     | Physical-threshold analysis for Fig. 4 and SI backend        |
| `fig4/check_exchange.py`       | Recomputes `S_th` after exchanging `L` and `R`, compares the two branches, and confirms that the identical main bars in Fig. 4C and 4D are not accidental.                                                            | Verification supporting Fig. 4C and 4D                       |
| `fig5/fig5_v4.py`              | Generates the eight panels (A-H) of Fig. 5, including the `alpha`, `eta`, and `K` mechanisms, `lag` ratio statistics, bootstrap checks, and replication of quantiles across five random seeds.                        | Fig. 5, including the seed-robustness check in Fig. 5E       |

### Supporting information

| Script                         | Purpose                                                                                                                                                                                                            | Manuscript item                              |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------- |
| `si/fig31_SI.py`               | Removes mutual inhibition or self-activation separately and tests whether the bistable window disappears.                                                                                                          | Fig. S1                                      |
| `si/make_si_fig2_composite.py` | Combines the hit-rate comparison, rule-tracking panel, branch-separation profile, and `(a,b)` scan into a unified 2 x 2 composite with consistent fonts and colours.                                               | Fig. S2                                      |
| `si/sref_window_stats.py`      | Draws 150 parameter sets, computes the per-sample bistable window, and compares it with a fixed read-out. Detailed results are written to `sref_window_stats.csv`; this script is the source of the `78.7%` value. | Data source for Fig. S2A and S2B             |
| `si/sref_ab_scan.py`           | Runs Sobol analysis for seven `(a,b)` settings, with 4608 model rows per setting, and examines how the sensitivity ranking changes with the read-out rule.                                                         | Data source for Fig. S2D                     |
| `si/sref_vs_fixedS.py`         | Compares the fixed `S=0.6` read-out with the per-sample `S_ref` rule. The comparison shows that `33.4%` of design rows change status and that `sum(ST)` increases from about `1.0` to `1.590`.                     | Read-out convention discussed in the SI text |
| `si/check_nan_repair.py`       | Rechecks the missing-value repair log for the production run. The observed repair count is zero, supporting the statement that no post-processing was applied to the final Fig. 4 results.                         | Methods statement in the SI                  |

## Recommended execution order

### Main figures

```bash
# Fig. 2
python fig2/fig31_3x3.py

# Fig. 3
python fig3/Fig32.py

# Fig. 4
python fig4/fig4_final.py --n 4096 --nproc 8 --seed 20241011
python fig4/fig4_selfcheck.py --results
python fig4/check_exchange.py
python fig4/check_table1_allfigs.py
python fig4/plot_Htrace_v11.py

# Fig. 5
python fig5/fig5_v4.py --recompute
```

### Supporting information

```bash
# Fig. S1
python si/fig31_SI.py

# Fig. S2A and S2B
python si/sref_window_stats.py

# Fig. S2D data source
python si/sref_ab_scan.py --n 512 --nproc 8

# Composite Fig. S2
python si/make_si_fig2_composite.py

# Fixed-S versus per-sample-S comparison
python si/sref_vs_fixedS.py

# Check the missing-value repair log
python si/check_nan_repair.py
```


    
