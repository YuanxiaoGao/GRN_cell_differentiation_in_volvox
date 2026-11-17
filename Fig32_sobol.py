import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from SALib.sample import saltelli
from SALib.analyze import sobol

import Fig32

OUT = Fig32.OUT


def run_group(names, output_cols, compute_dyn, N, tag):
    bounds = [Fig32.BOUNDS[Fig32.NAMES.index(nm)] for nm in names]
    problem = {'num_vars': len(names), 'names': names, 'bounds': bounds}
    X = saltelli.sample(problem, N)
    Y = []
    for row in X:
        full = np.array([Fig32.BASE[nm] for nm in Fig32.NAMES], dtype=float)
        for j, nm in enumerate(names):
            full[Fig32.NAMES.index(nm)] = row[j]
        Y.append(Fig32.model_output(full, compute_dyn=compute_dyn))
    Y = np.array(Y)
    results = {}
    for col, outname in output_cols:
        Si = sobol.analyze(problem, Y[:, col])
        results[outname] = {
            'S1': [float(v) for v in Si['S1']],
            'ST': [float(v) for v in Si['ST']],
            'S1_conf': [float(v) for v in Si['S1_conf']],
            'ST_conf': [float(v) for v in Si['ST_conf']],
        }
    return problem, results


def plot_sobol(panels, fig_name):
    n_panels = len(panels)
    ncols = 2
    nrows = int(np.ceil(n_panels / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(13, 9))
    axs = np.array(axs).ravel()
    for k, (outname, res, problem, group_label) in enumerate(panels):
        ax = axs[k]
        order = np.argsort(res['ST'])[::-1]
        names_sorted = [problem['names'][i] for i in order]
        s1 = [res['S1'][i] for i in order]
        st = [res['ST'][i] for i in order]
        s1_conf = [res['S1_conf'][i] for i in order]
        st_conf = [res['ST_conf'][i] for i in order]
        xpos = np.arange(len(names_sorted))
        w = 0.35
        ax.bar(xpos - w/2, s1, w, yerr=s1_conf, capsize=3,
               ecolor='#888888', label='S1', color='#a65628')
        ax.bar(xpos + w/2, st, w, yerr=st_conf, capsize=3,
               ecolor='#888888', label='ST', color='#377eb8')
        ax.set_xticks(xpos)
        ax.set_xticklabels(
            [Fig32.GREEK_NAMES[Fig32.NAMES.index(nm)]
             for nm in names_sorted], fontsize=9)
        ax.set_ylabel('Sobol index')
        ax.set_title(
            Fig32.OUTPUT_TITLES[Fig32.OUTPUT_NAMES.index(outname)],
            fontsize=11)
        ax.text(0.98, 0.97, group_label, transform=ax.transAxes,
                ha='right', va='top', fontsize=10)
        ax.legend(loc='upper right', bbox_to_anchor=(1.0, 0.90),
                  fontsize=9)
    for ax in axs[n_panels:]:
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(fig_name, dpi=200, bbox_inches='tight')
    plt.close(fig)


def build_panels(lr_problem, lr_res, hs_problem, hs_res):
    return [
        ('S_th', lr_res['S_th'], lr_problem, 'L-R group'),
        ('H peak/plateau', hs_res['H peak/plateau'], hs_problem,
         'H/S group'),
        ('log10(L_high/R_low)', lr_res['log10(L_high/R_low)'],
         lr_problem, 'L-R group'),
        ('log10(R_high/R_low)', lr_res['log10(R_high/R_low)'],
         lr_problem, 'L-R group'),
    ]


def main(n_lr, n_hs):
    with open(os.path.join(OUT, 'fig32_morris.json'), 'r') as f:
        morris = json.load(f)
    lr_names = morris['lr_keep']
    hs_names = morris['hs_keep']
    print('LR group:', lr_names)
    print('HS group:', hs_names)

    lr_problem, lr_res = run_group(
        lr_names,
        [(0, 'S_th'),
         (1, 'log10(L_high/R_low)'),
         (3, 'log10(R_high/R_low)')],
        compute_dyn=False, N=n_lr, tag='lr')

    hs_problem, hs_res = run_group(
        hs_names, [(2, 'H peak/plateau')],
        compute_dyn=True, N=n_hs, tag='hs')

    fig_name = os.path.join(OUT, 'fig32_sobol.png')
    panels = build_panels(lr_problem, lr_res, hs_problem, hs_res)
    plot_sobol(panels, fig_name)

    out_json = os.path.join(OUT, 'fig32_sobol.json')
    with open(out_json, 'w') as f:
        json.dump({
            'lr_names': lr_names,
            'hs_names': hs_names,
            'lr_results': lr_res,
            'hs_results': hs_res,
        }, f, indent=2)

    print('Saved', fig_name)
    print('Saved', out_json)
    for name, res in lr_res.items():
        order = np.argsort(res['ST'])[::-1]
        top = [(lr_names[i], round(res['ST'][i], 3)) for i in order[:5]]
        print('{} ST top: {}'.format(name, top))
    for name, res in hs_res.items():
        order = np.argsort(res['ST'])[::-1]
        top = [(hs_names[i], round(res['ST'][i], 3)) for i in order[:5]]
        print('{} ST top: {}'.format(name, top))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_lr', type=int, default=120,
                        help='Saltelli N for L-R group')
    parser.add_argument('--n_hs', type=int, default=100,
                        help='Saltelli N for H/S group')
    parser.add_argument('--plot-only', action='store_true',
                        help='regenerate figure from saved JSON')
    args = parser.parse_args()
    if args.plot_only:
        with open(os.path.join(OUT, 'fig32_sobol.json'), 'r') as f:
            data = json.load(f)

        def make_problem(names):
            return {
                'num_vars': len(names), 'names': names,
                'bounds': [Fig32.BOUNDS[Fig32.NAMES.index(nm)]
                           for nm in names],
            }

        lr_problem = make_problem(data['lr_names'])
        hs_problem = make_problem(data['hs_names'])
        panels = build_panels(lr_problem, data['lr_results'],
                              hs_problem, data['hs_results'])
        fig_name = os.path.join(OUT, 'fig32_sobol.png')
        plot_sobol(panels, fig_name)
        print('Plot-only saved', fig_name)
    else:
        main(args.n_lr, args.n_hs)
