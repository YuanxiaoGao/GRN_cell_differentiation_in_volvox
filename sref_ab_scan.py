import argparse
import csv
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', 'fig4')))

import fig4_final as F
OUTDIR = os.path.join(HERE, 'sref_scan')
CACHE = os.path.join(OUTDIR, '_cache')
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(CACHE, exist_ok=True)

SEED = 20241011
OUT_C = 'log10(L_high/R_low)'
OUT_D = 'log10(R_high/R_low)'

CONFIGS = [
    ('a=2.0,b=0.50', ('min', 2.0, 0.50)),
    ('a=2.5,b=0.55', ('min', 2.5, 0.55)),
    ('a=3.0,b=0.60', ('min', 3.0, 0.60)),
    ('a=4.0,b=0.70', ('min', 4.0, 0.70)),
    ('a=5.0,b=0.80', ('min', 5.0, 0.80)),
    ('a=8.0,b=0.90', ('min', 8.0, 0.90)),
    ('geomid', ('geo', 0.0, 0.0)), 
]


def sref_of(w, mode):
    S_th, S_hi = w
    lo, hi = S_th * 1.02, S_hi * 0.98
    if mode[0] == 'geo':
        s = math.sqrt(S_th * S_hi) if S_hi > 0 else 1.5 * S_th
        br = 0
    else:
        _, a, b = mode
        s = min(a * S_th, b * S_hi) if S_hi > 0 else a * S_th
        br = 0 if a * S_th <= b * S_hi else 1
    c = float(np.clip(s, lo, hi))
    if not math.isclose(c, s, rel_tol=1e-12):
        br = 2 if c <= lo else 3
    return c, br


def _worker_ab(args):
    row, names, mode = args
    p = F.build_p(F.expand(row, names), F.FULL)
    w = F.bistable_window(p)
    if w is None:
        return [np.nan] * 6
    S_ref, br = sref_of(w, mode)
    st = F.stable_states(S_ref, p)
    if len(st) < 2:
        return [np.nan] * 6
    iL = max(range(len(st)), key=lambda k: st[k][0] - st[k][1])
    iR = max(range(len(st)), key=lambda k: st[k][1] - st[k][0])
    if iL == iR:
        return [np.nan] * 6
    Lh, Rl = st[iL]
    Ll, Rh = st[iR]
    return [float(w[0]), np.nan,
            float(np.log10(max(Lh, 1e-12) / max(Rl, 1e-12))),
            float(np.log10(max(Rh, 1e-12) / max(Ll, 1e-12))),
            S_ref, float(br)]


def ev_ab(X, names, mode, nproc):
    base_vec = np.array([F.base_of(n) for n in names], dtype=float)

    def run(rows):
        if nproc and nproc > 1:
            from multiprocessing import Pool
            with Pool(nproc) as pool:
                return np.array(pool.map(
                    _worker_ab, [(r, names, mode) for r in rows],
                    chunksize=32), dtype=float)
        return np.array([_worker_ab((r, names, mode)) for r in rows])

    Y = run(X)
    cols = [0, 2, 3]
    n_bad0 = int(np.any(np.isnan(Y[:, cols]), axis=1).sum())
    for _ in range(6):
        bad = np.any(np.isnan(Y[:, cols]), axis=1)
        if not bad.any():
            break
        X = X.copy()
        X[bad] = 0.5 * X[bad] + 0.5 * base_vec
        Y[bad] = run(X[bad])
    return X, Y, n_bad0, int(np.any(np.isnan(Y[:, cols]), axis=1).sum())


def problem_of(names):
    return {'num_vars': len(names), 'names': list(names),
            'bounds': [F.bounds_of(n) for n in names]}


def run_cfg(lab, mode, n, nproc, seed, force):
    from SALib.sample import saltelli
    names = F.LR_NAMES
    problem = problem_of(names)
    p_npz = os.path.join(CACHE, '_srefab_%s_n%d.npz'
                         % (lab.replace('=', '').replace(',', '_')
                            .replace('.', ''), n))
    if os.path.exists(p_npz) and not force:
        z = np.load(p_npz)
        X, Y, bad0, left = z['X'], z['Y'], int(z['n_bad0']), int(z['left'])
        print('   [%s] read %d rows from cache' % (lab, len(X)))
    else:
        np.random.seed(seed)
        X = saltelli.sample(problem, n, calc_second_order=False)
        t0 = time.time()
        X, Y, bad0, left = ev_ab(X, names, mode, nproc)
        print('   [%s] %d rows, %.0f s, initially invalid %d (%.1f%%), remaining after repair %d'
              % (lab, len(X), time.time() - t0, bad0,
                 100.0 * bad0 / len(X), left))
        np.savez_compressed(p_npz, X=X, Y=Y, n_bad0=bad0, left=left)
    res = {k: F.pack(F.sobol_analyze(problem, Y[:, F.OUT_ALL.index(k)], seed))
           for k in (OUT_C, OUT_D, 'S_th')}
    return X, Y, res, bad0, left


def diag_of(lab, Y, res, bad0, n):
    sth = Y[:, 0]
    yc = Y[:, 2]
    sref = Y[:, 4]
    br = Y[:, 5]
    ok = np.isfinite(sth) & np.isfinite(yc) & (sth > 0)
    return {
        'label': lab,
        'n_rows': int(n),
        'n_bad0': int(bad0),
        'pct_bad0': 100.0 * bad0 / n,
        'branch_lower_pct': 100.0 * float(np.mean(br == 0)),
        'branch_upper_pct': 100.0 * float(np.mean(br == 1)),
        'clip_lo_pct': 100.0 * float(np.mean(br == 2)),
        'clip_hi_pct': 100.0 * float(np.mean(br == 3)),
        'Sref_med': float(np.nanmedian(sref)),
        'Sref_p05': float(np.nanpercentile(sref, 5)),
        'Sref_p95': float(np.nanpercentile(sref, 95)),
        'ratio_med': float(np.nanmedian(sref / sth)),
        'yC_mean': float(np.nanmean(yc)),
        'yC_std': float(np.nanstd(yc)),
        'corr_yC_log10Sth': float(np.corrcoef(yc[ok], np.log10(sth[ok]))[0, 1]),
        'sumST_C': float(np.sum(res[OUT_C]['ST'])),
    }


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def plots(cfgs, diag, ref_key, outdir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    names = F.LR_NAMES
    labs = [c[0] for c in cfgs]
    n = len(cfgs)
    cols = ['#a65628' if labs[i] == ref_key else
            plt.get_cmap('Blues')(0.32 + 0.60 * i / max(n - 1, 1))
            for i in range(n)]

    fig, axs = plt.subplots(1, 2, figsize=(15.4, 6.0))
    vmax = 1.0
    for k in (OUT_C, OUT_D):
        for lab, Y, res, b0, lf in cfgs:
            vmax = max(vmax, float(np.max(np.array(res[k]['ST']))))
    for ax, k, ttl in ((axs[0], OUT_C,
                        'panel C  $y=\\log_{10}(L_{high}/R_{low})$'),
                       (axs[1], OUT_D,
                        'panel D  $y=\\log_{10}(R_{high}/L_{low})$')):
        x = np.arange(len(names))
        w = 0.80 / n
        for i, (lab, Y, res, b0, lf) in enumerate(cfgs):
            st = np.array(res[k]['ST'], dtype=float)
            ax.bar(x + (i - (n - 1) / 2.0) * w, st, w, label=lab,
                   color=cols[i], edgecolor='white', lw=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels([F.GREEK[nm] for nm in names], fontsize=12)
        ax.set_ylabel('Sobol total-effect index  $S_T$', fontsize=11)
        ax.set_title(ttl, fontsize=12)
        ax.axhline(0, color='k', lw=0.8)
        ax.grid(axis='y', alpha=0.25)
        ax.set_xlim(-0.6, len(names) - 0.4)
        ax.set_ylim(0, vmax * 1.12)
    h, lb = axs[0].get_legend_handles_labels()
    fig.legend(h, lb, fontsize=9.5, ncol=n, frameon=False,
               loc='lower center', bbox_to_anchor=(0.5, -0.02))
    fig.suptitle('Fig.4 C/D: $S_T$ of the 7 (L,R) parameters under different '
                 '$S_{\\rm ref}$ conventions ($N$=512); brown = the published '
                 'rule $S_{\\rm ref}=\\min(3S_{th},0.6S_{hi})$', fontsize=11)
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    for ext in ('.png', '.pdf'):
        fig.savefig(os.path.join(outdir, 'sref_ab_ST' + ext), dpi=200)
    plt.close(fig)

    fig, axs = plt.subplots(2, 2, figsize=(12.6, 8.2))
    axs = axs.ravel()
    xs = np.arange(n)

    def panel(ax, key, ttl, ylab, fmt='%.3f', log=False, signed=False):
        v = [dg[key] for dg in diag]
        ax.bar(xs, v, color=cols, edgecolor='#333333', lw=0.5)
        if log:
            from matplotlib.ticker import NullFormatter, ScalarFormatter
            ax.set_yscale('log')
            ax.yaxis.set_major_formatter(ScalarFormatter())
            ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_xticks(xs)
        ax.set_xticklabels(labs, fontsize=8.5, rotation=18, ha='right')
        ax.set_title(ttl, fontsize=10.5)
        ax.set_ylabel(ylab, fontsize=10)
        ax.grid(axis='y', alpha=0.25)
        vmax, vmin = max(v), min(v)
        for i, val in enumerate(v):
            if signed:
                ax.text(i, val - 0.030 * abs(vmin), fmt % val, ha='center',
                        va='top', fontsize=8)
            else:
                ax.text(i, val + 0.02 * (vmax - vmin), fmt % val, ha='center',
                        va='bottom', fontsize=8)
        if signed:
            ax.set_ylim(vmin * 1.38, 0.30 * abs(vmin))

    panel(axs[0], 'Sref_med', '(a) median of $S_{\\rm ref}$ across the design '
          '(note the log scale)', 'median $S_{\\rm ref}$', '%.2f', log=True)
    panel(axs[1], 'yC_std', '(b) sd( y ): how much variance is left for Sobol',
          'sd of y')
    panel(axs[2], 'corr_yC_log10Sth', '(c) corr( y , log10 $S_{th}$ ): lower '
          'in magnitude = better decoupled from the trivial threshold effect',
          'Pearson r', signed=True)
    panel(axs[3], 'sumST_C', '(d) $\\sum S_T$ of panel C (should stay near 1)',
          '$\\sum S_T$')
    fig.suptitle('Diagnostics vs the choice of $(a,b)$ in '
                 '$S_{\\rm ref}=\\min(a\\,S_{th},\\,b\\,S_{hi})$', fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for ext in ('.png', '.pdf'):
        fig.savefig(os.path.join(outdir, 'sref_ab_diagnostics' + ext), dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    p = dict(F.BASE)
    w = F.bistable_window(p)
    S_th = w[0]
    xs = np.geomspace(1.005 * S_th, 300 * S_th, 90)
    ys = []
    for s in xs:
        st = F.stable_states(float(s), p)
        if len(st) < 2:
            ys.append(np.nan)
            continue
        iL = max(range(len(st)), key=lambda k: st[k][0] - st[k][1])
        iR = max(range(len(st)), key=lambda k: st[k][1] - st[k][0])
        Lh, Rl = st[iL]
        Ll, Rh = st[iR]
        ys.append(math.log10(Lh / max(Rl, 1e-12)) if iL != iR else np.nan)
    ys = np.array(ys)
    plat = float(np.nanmax(ys))
    ax.plot(xs / S_th, ys, color='#377eb8', lw=2.0)
    ax.axhline(plat, color='#888888', ls=':', lw=1.2)
    info = []
    for m, c in ((2.0, '#D85A30'), (3.0, '#993C1D'), (5.0, '#3B6D11'),
                 (8.0, '#534AB7')):
        ax.axvline(m, color=c, ls='--', lw=1.2, alpha=0.85)
        i = int(np.nanargmin(np.abs(xs / S_th - m)))
        ax.plot([m], [ys[i]], 'o', ms=6.5, color=c, zorder=5, mec='white',
                mew=1.2)
        info.append((m, c, ys[i]))
    for k, (m, c, v) in enumerate(info):
        ax.text(0.615, 0.93 - 0.072 * k,
                r'$%.0f\,S_{th}$:  y=%.2f   (%.0f%% of plateau)'
                % (m, v, 100 * v / plat),
                transform=ax.transAxes, color=c, fontsize=9.5, va='top',
                bbox=dict(fc='white', ec='none', alpha=0.8, pad=1.5))
    ax.plot([], [], ' ', label='dotted: plateau y=%.3f' % plat)
    ax.legend(loc='lower right', fontsize=9, framealpha=0.9)
    ax.set_xscale('log')
    ax.set_xlabel('$S\\,/\\,S_{th}$  (log scale)', fontsize=11)
    ax.set_ylabel('$\\log_{10}(L_{high}/R_{low})$', fontsize=11)
    ax.set_title('BASE: how the branch separation saturates away from the '
                 'lower fold ($S_{th}=%.4f$)' % S_th, fontsize=11)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    for ext in ('.png', '.pdf'):
        fig.savefig(os.path.join(outdir, 'sref_profile' + ext), dpi=200)
    plt.close(fig)
    print('saved sref_ab_ST / sref_ab_diagnostics / sref_profile')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=512)
    ap.add_argument('--nproc', type=int, default=8)
    ap.add_argument('--seed', type=int, default=SEED)
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()

    ref_key = 'a=3.0,b=0.60'
    cfgs, diags = [], []
    t_all = time.time()
    for lab, mode in CONFIGS:
        X, Y, res, bad0, left = run_cfg(lab, mode, args.n, args.nproc,
                                        args.seed, args.force)
        cfgs.append((lab, Y, res, bad0, left))
        diags.append(diag_of(lab, Y, res, bad0, len(X)))
    print('\nall done, %.0f s' % (time.time() - t_all))

    ref_y = None
    ref_st = None
    for lab, Y, res, b0, lf in cfgs:
        if lab == ref_key:
            ref_st = np.array(res[OUT_C]['ST'], dtype=float)
            ref_y = Y[:, 2]
    ref_res = next(r for lab, Y, r, b0, lf in cfgs if lab == ref_key)

    print('\n=== ST of panel C ===')
    hdr = ['param'] + [c[0] for c in cfgs]
    print('%-8s' % hdr[0] + ''.join('%14s' % h for h in hdr[1:]))
    for i, nm in enumerate(F.LR_NAMES):
        print('%-8s' % nm + ''.join(
            '%14.3f' % np.array(r[OUT_C]['ST'], float)[i]
            for lab, Y, r, b0, lf in cfgs))

    print('\n=== ranking, sum(ST), agreement with (3, 0.6) ===')
    for lab, Y, res, b0, lf in cfgs:
        st = np.array(res[OUT_C]['ST'], dtype=float)
        o = np.argsort(-st)
        print('%-14s Σ=%.3f  r=%+.4f  rho=%+.3f  %s'
              % (lab, st.sum(), np.corrcoef(st, ref_st)[0, 1],
                 spearman(st, ref_st),
                 ' > '.join(F.LR_NAMES[i] for i in o)))

    print('\n=== diagnostics ===')
    keys = ['label', 'Sref_med', 'ratio_med', 'pct_bad0', 'branch_lower_pct',
            'branch_upper_pct', 'clip_hi_pct', 'yC_mean', 'yC_std',
            'corr_yC_log10Sth', 'sumST_C']
    with open(os.path.join(OUTDIR, 'sref_ab_diagnostics.csv'), 'w',
              newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(keys)
        for dg in diags:
            w.writerow([('%.6g' % dg[k]) if isinstance(dg[k], float) else dg[k]
                        for k in keys])
    with open(os.path.join(OUTDIR, 'sref_ab_ST.csv'), 'w', newline='',
              encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['parameter'] + [c[0] for c in cfgs])
        for i, nm in enumerate(F.LR_NAMES):
            w.writerow([nm] + ['%.6f' % np.array(r[OUT_C]['ST'], float)[i]
                               for lab, Y, r, b0, lf in cfgs])
    print('%-14s %8s %8s %8s %8s %8s %8s'
          % ('label', 'Sref median', 'S/Sth', 'invalid %', 'lower branch', 'upper branch', ' clipped high'))
    for dg in diags:
        print('%-14s %8.3f %8.2f %8.1f %8.1f %8.1f %8.1f'
              % (dg['label'], dg['Sref_med'], dg['ratio_med'], dg['pct_bad0'],
                 dg['branch_lower_pct'], dg['branch_upper_pct'],
                 dg['clip_hi_pct']))

    plots(cfgs, diags, ref_key, OUTDIR)
    print('output directory: %s' % OUTDIR)


if __name__ == '__main__':
    main()
