# -*- coding: utf-8 -*-
"""
fig4_final.py -- Fig. 4

"""
import argparse
import csv
import json
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

GAMMA = 0.35

BASE = {
    'aH': 3.5, 'aL': 3.5, 'eta': 0.7, 'beta': 0.0035,
    'K_SH': 0.5, 'K_SL': 0.5, 'K_SR': 0.5,
    'K_RL': 0.5, 'K_LR': 0.5, 'K_L': 0.5, 'K_R': 0.5,
    'Kc': 1.5, 'delta': 7.0, 'p': 3, 'n': 3, 'm': 3, 'q': 2,
    'b0': 0.055, 'bH': 0.9, 'Khs': 0.5, 'gS': 0.1,
}

BOUNDS = {
    'aH': [1.75, 5.25], 'aL': [1.75, 5.25], 'eta': [0.35, 1.05],
    'beta': [0.00175, 0.00525],
    'K_SH': [0.5, 1.5], 'K_SL': [0.5, 1.5], 'K_SR': [0.5, 1.5],
    'K_RL': [0.5, 1.5], 'K_LR': [0.5, 1.5],
    'K_L': [0.5, 1.5], 'K_R': [0.5, 1.5],
    'Kc': [1.0, 2.0], 'delta': [3.5, 10.5],
    'p': [2, 4], 'n': [2, 4], 'm': [2, 4], 'q': [1, 3],
    'b0': [0.03, 0.08], 'bH': [0.5, 1.3], 'Khs': [0.3, 0.8],
    'gS': [0.05, 0.2],
}

K7 = ['K_SH', 'K_SL', 'K_SR', 'K_RL', 'K_LR', 'K_L', 'K_R']
LR_NAMES = ['aL', 'eta', 'beta', 'K', 'p', 'n', 'q']
HS_NAMES = ['aH', 'K', 'Kc', 'delta', 'n', 'm', 'b0', 'bH', 'Khs', 'gS']
FULL = ['aL', 'eta', 'beta'] + K7 + ['p', 'n', 'q'] + \
       ['aH', 'Kc', 'delta', 'm', 'b0', 'bH', 'Khs', 'gS']


OUT_ALL = ['S_th', 'H peak/plateau',
           'log10(L_high/R_low)', 'log10(R_high/R_low)']

DESIGNS = [
    ('lr', LR_NAMES, ['S_th', 'log10(L_high/R_low)', 'log10(R_high/R_low)']),
    ('hs', HS_NAMES, ['H peak/plateau']),
]
WK_DESIGNS = [
    ('lr', K7, ['S_th', 'log10(L_high/R_low)', 'log10(R_high/R_low)']),
    ('hs', K7, ['H peak/plateau']),
]


def build_p(values, names):
    p = dict(BASE)
    for nm, v in zip(names, values):
        p[nm] = float(v)
    return p

def lr_rhs(L, R, S, p):
    L = max(L, 0.0)
    R = max(R, 0.0)
    n, pe, q = p['n'], p['p'], p['q']
    FL = S**n / (p['K_SL']**n + S**n)
    FR = S**n / (p['K_SR']**n + S**n)
    rR = p['K_RL']**pe / (p['K_RL']**pe + R**pe)
    rL = p['K_LR']**pe / (p['K_LR']**pe + L**pe)
    hL = L**q / (p['K_L']**q + L**q)
    hR = R**q / (p['K_R']**q + R**q)
    dL = p['beta'] + FL * (p['aL'] + p['eta'] * hL) * rR - GAMMA * L
    dR = p['beta'] + FR * (p['aL'] + p['eta'] * hR) * rL - GAMMA * R
    return dL, dR


def jac_lr(L, R, S, p):
    n, pe, q = p['n'], p['p'], p['q']
    FL = S**n / (p['K_SL']**n + S**n)
    FR = S**n / (p['K_SR']**n + S**n)
    rR = p['K_RL']**pe / (p['K_RL']**pe + R**pe)
    rL = p['K_LR']**pe / (p['K_LR']**pe + L**pe)
    hL = L**q / (p['K_L']**q + L**q)
    hR = R**q / (p['K_R']**q + R**q)
    dhL = q * p['K_L']**q * L**(q - 1) / (p['K_L']**q + L**q)**2
    dhR = q * p['K_R']**q * R**(q - 1) / (p['K_R']**q + R**q)**2
    dgR = -pe * p['K_RL']**pe * R**(pe - 1) / (p['K_RL']**pe + R**pe)**2
    dgL = -pe * p['K_LR']**pe * L**(pe - 1) / (p['K_LR']**pe + L**pe)**2
    return np.array([
        [FL * p['eta'] * dhL * rR - GAMMA,
         FL * (p['aL'] + p['eta'] * hL) * dgR],
        [FR * (p['aL'] + p['eta'] * hR) * dgL,
         FR * p['eta'] * dhR * rL - GAMMA],
    ])


GUESS = [[10.0, 0.01], [0.01, 10.0], [1.0, 1.0],
         [5.0, 5.0], [0.1, 8.0], [8.0, 0.1]]


def stable_states(S, p, guesses=None):
    from scipy.optimize import root
    if guesses is None:
        guesses = GUESS
    sols = []
    for g in guesses:
        try:
            r = root(lambda xy: np.array(lr_rhs(xy[0], xy[1], S, p)),
                     g, method='hybr', options={'xtol': 1e-12})
        except Exception:
            continue
        if not r.success:
            continue
        x = r.x
        if x[0] < 0 or x[1] < 0 or not np.all(np.isfinite(x)):
            continue
        if any(np.max(np.abs(x - s)) < 1e-4 for s in sols):
            continue
        try:
            ev = np.linalg.eigvals(jac_lr(x[0], x[1], S, p))
        except Exception:
            continue
        if np.all(np.real(ev) < 0):
            sols.append(x)
    return sols


_SGRID = np.geomspace(0.03, 60.0, 40)


def _bisect_flag(f, lo, hi, iters=20):
    flo = f(lo)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if f(mid) == flo:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def bistable_window(p, sgrid=None):
    if sgrid is None:
        sgrid = _SGRID
    cnt = [len(stable_states(S, p)) for S in sgrid]
    idx2 = [i for i, c in enumerate(cnt) if c >= 2]
    if not idx2:
        return None
    i0, i1 = idx2[0], idx2[-1]

    def f(z):
        return len(stable_states(z, p)) >= 2

    lo = sgrid[0] if i0 == 0 else _bisect_flag(f, sgrid[i0 - 1], sgrid[i0])
    hi = (sgrid[-1] if i1 == len(sgrid) - 1
          else _bisect_flag(f, sgrid[i1], sgrid[i1 + 1]))
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
        return None
    return float(lo), float(hi)


def lr_outputs(p):
    w = bistable_window(p)
    if w is None:
        return None
    S_th, S_hi = w
    S_ref = min(3.0 * S_th, 0.6 * S_hi) if S_hi > 0 else 3.0 * S_th
    S_ref = float(np.clip(S_ref, S_th * 1.02, S_hi * 0.98))
    st = stable_states(S_ref, p)
    if len(st) < 2:
        return None
    i_L = max(range(len(st)), key=lambda k: st[k][0] - st[k][1])
    i_R = max(range(len(st)), key=lambda k: st[k][1] - st[k][0])
    if i_L == i_R:
        return None
    L_high, R_low = st[i_L]
    L_low, R_high = st[i_R]
    y2 = float(np.log10(max(L_high, 1e-12) / max(R_low, 1e-12)))
    y4 = float(np.log10(max(R_high, 1e-12) / max(L_low, 1e-12)))
    return float(S_th), y2, y4


def full_rhs(t, y, p):
    H, S, L, R = y
    L = max(L, 0.0)
    R = max(R, 0.0)
    n, pe, q = p['n'], p['p'], p['q']
    FL = S**n / (p['K_SL']**n + S**n)
    FR = S**n / (p['K_SR']**n + S**n)
    FH = S**n / (p['K_SH']**n + S**n)
    G = S**p['m'] / (p['Kc']**p['m'] + S**p['m'])
    dH = p['aH'] * FH - GAMMA * H - p['delta'] * G * H
    dS = p['b0'] + p['bH'] * H**n / (p['Khs']**n + H**n) - p['gS'] * S
    rR = p['K_RL']**pe / (p['K_RL']**pe + R**pe)
    rL = p['K_LR']**pe / (p['K_LR']**pe + L**pe)
    hL = L**q / (p['K_L']**q + L**q)
    hR = R**q / (p['K_R']**q + R**q)
    dL = p['beta'] + FL * (p['aL'] + p['eta'] * hL) * rR - GAMMA * L
    dR = p['beta'] + FR * (p['aL'] + p['eta'] * hR) * rL - GAMMA * R
    return [dH, dS, dL, dR]


def full_trajectory(p, T_MAX=36.0, rtol=1e-9, atol=1e-11, max_step=0.1):
    from scipy.integrate import solve_ivp
    t_pre = np.arange(0, 3.000001, 0.25)
    t_post = np.arange(3.0, T_MAX + 1e-6, 0.25)
    y0 = [0.01, 0.05, 0.05, 0.05]
    pre = solve_ivp(full_rhs, [0, 3.0], y0, args=(p,), t_eval=t_pre,
                    method='LSODA', max_step=max_step, rtol=rtol, atol=atol)
    y3 = pre.y[:, -1].copy()
    y3[2] += 0.02
    y3[3] -= 0.02
    post = solve_ivp(full_rhs, [3.0, T_MAX], y3, args=(p,), t_eval=t_post,
                     method='LSODA', max_step=max_step, rtol=rtol, atol=atol)
    t = np.hstack([pre.t, post.t[1:]])
    Y = np.hstack([pre.y, post.y[:, 1:]])
    return t, Y


def full_dynamics_ratio(p, T_MAX=36.0):
    t, Y = full_trajectory(p, T_MAX)
    H = Y[0]
    return float(H.max() / max(H[-1], 1e-12))


def expand(row, names):
    p = dict(BASE)
    for nm, v in zip(names, row):
        p[nm] = float(v)
    if 'K' in names:
        k = float(p['K'])
        for kk in K7:
            p[kk] = k
    return [p[n] for n in FULL]


def _worker(args):
    row, names, what = args
    p = build_p(expand(row, names), FULL)
    out = {}
    if 'S_th' in what:
        r = lr_outputs(p)
        if r is None:
            out['S_th'] = np.nan
            out['log10(L_high/R_low)'] = np.nan
            out['log10(R_high/R_low)'] = np.nan
        else:
            out['S_th'] = r[0]
            out['log10(L_high/R_low)'] = r[1]
            out['log10(R_high/R_low)'] = r[2]
    if 'H peak/plateau' in what:
        try:
            out['H peak/plateau'] = full_dynamics_ratio(p)
        except Exception:
            out['H peak/plateau'] = np.nan
    return [out.get(k, np.nan) for k in OUT_ALL]


def base_of(nm):
    return BASE[nm] if nm in BASE else 0.5


def bounds_of(nm):
    return BOUNDS[nm] if nm in BOUNDS else BOUNDS['K_SH']


def evaluate(X, names, what, nproc):
    base_vec = np.array([base_of(n) for n in names], dtype=float)

    def ev(rows):
        if nproc and nproc > 1:
            from multiprocessing import Pool
            with Pool(nproc) as pool:
                res = pool.map(_worker, [(r, names, what) for r in rows],
                               chunksize=32)
            return np.array(res, dtype=float)
        return np.array([_worker((r, names, what)) for r in rows])

    Y = ev(X)
    cols = [OUT_ALL.index(w) for w in what]
    n_bad0 = int(np.any(np.isnan(Y[:, cols]), axis=1).sum())
    for _ in range(6):
        bad = np.any(np.isnan(Y[:, cols]), axis=1)
        if not bad.any():
            break
        X = X.copy()
        X[bad] = 0.5 * X[bad] + 0.5 * base_vec
        Y[bad] = ev(X[bad])
    return X, Y, n_bad0, int(np.any(np.isnan(Y[:, cols]), axis=1).sum())


def make_problem(names):
    return {'num_vars': len(names), 'names': names,
            'bounds': [bounds_of(n) for n in names]}


def sobol_analyze(problem, y, seed):
    from SALib.analyze import sobol
    try:
        return sobol.analyze(problem, y, calc_second_order=False, seed=seed)
    except TypeError:
        np.random.seed(seed)
        return sobol.analyze(problem, y, calc_second_order=False)


def pack(Si):
    return {'S1': [float(v) for v in Si['S1']],
            'ST': [float(v) for v in Si['ST']],
            'S1_conf': [float(v) for v in Si['S1_conf']],
            'ST_conf': [float(v) for v in Si['ST_conf']]}


def cache_path(n, prefix, tag):
    return os.path.join(HERE, '_fig4final_n%d_%s_%s.npz' % (n, prefix, tag))


LEGACY = {('main', 'lr'): '_fig4_v10_main_lr.npz',
          ('main', 'hs'): '_fig4_v10_main_hs.npz',
          ('within', 'lr'): '_fig4_v10_within_lr.npz',
          ('within', 'hs'): '_fig4_v10_within_hs.npz'}


def run_design(tag, names, what, n, nproc, prefix, seed, force):
    from SALib.sample import saltelli
    p_npz = cache_path(n, prefix, tag)
    problem = make_problem(names)
    if os.path.exists(p_npz) and not force:
        z = np.load(p_npz)
        X, Y = z['X'], z['Y']
        print('  [%s/%s] read %d rows from cache' % (prefix, tag, len(X)))
    elif (not force and n == 4096 and os.path.exists(
            os.path.join(HERE, LEGACY[(prefix, tag)]))):
        z = np.load(os.path.join(HERE, LEGACY[(prefix, tag)]))
        X, Y = z['X'], z['Y']
        np.savez_compressed(p_npz, X=X, Y=Y)
        print('  [%s/%s] reuse history cache, %d rows -> %s'
              % (prefix, tag, len(X), os.path.basename(p_npz)))
    else:
        np.random.seed(seed)
        X = saltelli.sample(problem, n, calc_second_order=False)
        print('  [%s/%s] sampling %d rows (N=%d, D=%d)...'
              % (prefix, tag, len(X), n, len(names)))
        t0 = time.time()
        X, Y, bad0, left = evaluate(X, names, what, nproc)
        print('  [%s/%s] %.0f s, initial NaN %d, remaining after repair %d'
              % (prefix, tag, time.time() - t0, bad0, left))
        np.savez_compressed(p_npz, X=X, Y=Y)
    res = {w: pack(sobol_analyze(problem, Y[:, OUT_ALL.index(w)], seed))
           for w in what}
    return res, len(X)


def compute(n, nproc, seed, force):
    out = {'meta': {
        'version': 'fig4-final', 'N_base': n,
        'calc_second_order': False, 'seed': seed, 'n_proc': nproc,
        'model': 'four-variable Volvox ODE; single K assigned to all 7 '
                 'thresholds in the main panels, 7 K perturbed independently '
                 'in the insets (all other parameters pinned to BASE)',
        'outputs': OUT_ALL,
        'note': 'log10(R_high/R_low) column is actually log10(R*_high/L*_low); '
                'under the merged-K design the two branch ratios coincide, '
                'and they differ only inside the insets',
    }}
    t_all = time.time()

    print('=== main panels (merged K) ===')
    for tag, names, what in DESIGNS:
        res, nrows = run_design(tag, names, what, n, nproc, 'main', seed, force)
        out['main_%s_names' % tag] = list(names)
        for w, r in res.items():
            out.setdefault('main_results', {})[w] = r
        out['meta']['n_rows_main_' + tag] = int(nrows)
        for w, r in res.items():
            st = np.array(r['ST'])
            print('   %-22s ST: %s' % (w, ', '.join(
                '%s=%.3f' % (names[i], st[i]) for i in np.argsort(-st))))

    print('=== insets (only the 7 K are perturbed) ===')
    out['within_names'] = list(K7)
    for tag, names, what in WK_DESIGNS:
        res, nrows = run_design(tag, names, what, n, nproc, 'within', seed,
                                force)
        for w, r in res.items():
            out.setdefault('within_results', {})[w] = r
        out['meta']['n_rows_within_' + tag] = int(nrows)
        for w, r in res.items():
            st = np.array(r['ST'])
            print('   %-22s ST: %s' % (w, ', '.join(
                '%s=%.4f' % (names[i], st[i]) for i in np.argsort(-st))))

    p_json = os.path.join(HERE, 'fig4-final.json')
    with open(p_json, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print('saved %s (total %.0f s)' % (os.path.basename(p_json),
                                       time.time() - t_all))
    return out


C_S1 = '#a65628'          # brown
C_ST = '#377eb8'          # blue
C_ERR = '#888888'

GREEK = {
    'aH': r'$\alpha_H$', 'aL': r'$\alpha_L$', 'eta': r'$\eta$',
    'beta': r'$\beta$', 'K': r'$K$', 'Kc': r'$K_c$', 'delta': r'$\delta$',
    'p': r'$p$', 'n': r'$n$', 'm': r'$m$', 'q': r'$q$',
    'b0': r'$\beta_0$', 'bH': r'$\beta_H$', 'Khs': r'$K_{HS}$',
    'gS': r'$\gamma_S$',
}
KG = {'K_SH': r'$K_{SH}$', 'K_SL': r'$K_{SL}$', 'K_SR': r'$K_{SR}$',
      'K_RL': r'$K_{RL}$', 'K_LR': r'$K_{LR}$',
      'K_L': r'$K_{L}$', 'K_R': r'$K_{R}$'}

EXCH = [('K_SL', 'K_SR'), ('K_RL', 'K_LR'), ('K_L', 'K_R')]
KROLE = {'K_SH': 'S->H gate', 'K_SL': 'S->L gate',
         'K_SR': 'S->R gate', 'K_RL': 'R -| L',
         'K_LR': 'L -| R', 'K_L': 'L self-activation',
         'K_R': 'R self-activation'}


TITLES = {'S_th': r'$S_{\rm th}$',
          'H peak/plateau': r'$H_{\rm peak}/H_{\rm plateau}$',
          'log10(L_high/R_low)':
              r'$\log_{10}(L^{*}_{\rm high}/R^{*}_{\rm low})$',
          'log10(R_high/R_low)':
              r'$\log_{10}(R^{*}_{\rm high}/L^{*}_{\rm low})$'}

SYM = {'S_th': True, 'H peak/plateau': True,
       'log10(L_high/R_low)': False, 'log10(R_high/R_low)': False}

PANELS = [('S_th', 'lr', 'L-R group'),
          ('H peak/plateau', 'hs', 'H/S group, full dynamics'),
          ('log10(L_high/R_low)', 'lr', 'L-R group'),
          ('log10(R_high/R_low)', 'lr', 'L-R group')]

INSET_YLIM = (0.0, 1.0)
INSET_TITLE = '$K$ split into 7 thresholds'
LBL_GAP = 0.12


def sort_by_st(res):
    st = np.array(res['ST'], dtype=float)
    o = list(np.argsort(-st))
    return (o, np.array(res['S1'])[o], st[o],
            np.array(res['S1_conf'])[o], np.array(res['ST_conf'])[o])


def nice_ticks(ymin, ymax, n=5):
    return [round(i * ymax / n, 6) for i in range(n + 1)]


def main_axes(ax, names, res):
    o = list(np.argsort(-np.array(res['ST'], dtype=float)))
    nm = [names[i] for i in o]
    s1 = np.array(res['S1'])[o]
    st = np.array(res['ST'])[o]
    c1 = np.array(res['S1_conf'])[o]
    ct = np.array(res['ST_conf'])[o]
    x = np.arange(len(nm))
    w = 0.35
    ax.bar(x - w / 2, s1, w, yerr=c1, capsize=3, ecolor=C_ERR, label='S1',
           color=C_S1)
    ax.bar(x + w / 2, st, w, yerr=ct, capsize=3, ecolor=C_ERR, label='ST',
           color=C_ST)
    ax.set_xticks(x)
    ax.set_xticklabels([GREEK[n] for n in nm], fontsize=9)
    ax.set_ylabel('Sobol index')
    return nm, s1, st, c1, ct


def bar_top_frac(ax, nm, s1, st, c1, ct, x0, x1, margin=0.05):
    lo, hi = ax.get_ylim()
    n = len(nm)
    span = n + 0.4
    worst = -np.inf
    who = ''
    for i in range(n):
        b0 = (i + 0.7 - 0.35) / span - margin
        b1 = (i + 0.7 + 0.35) / span + margin
        if b1 <= x0 or b0 >= x1:
            continue
        top = max(s1[i] + c1[i], st[i] + ct[i])
        f = (top - lo) / (hi - lo)
        if f > worst:
            worst, who = f, nm[i]
    return worst, who


def legend_bottom_frac(fig, ax, legend):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    bb = legend.get_window_extent(r)
    inv = ax.transAxes.inverted()
    return float(inv.transform((bb.x0, bb.y0))[1])


def choose_box(ax, nm, s1, st, c1, ct, top):
    best = None
    for x0 in np.arange(0.40, 0.725, 0.02):
        x0 = float(round(x0, 3))
        w = 0.98 - x0
        worst, who = bar_top_frac(ax, nm, s1, st, c1, ct, x0, 0.98)
        worst = worst if np.isfinite(worst) else 0.0
        y0 = max(0.20, worst + LBL_GAP)
        h = top - y0
        if h < 0.30 or w < 0.26:
            continue
        area = w * min(h, 0.42)
        if best is None or area > best[0]:
            best = (area, (x0, y0, w, h), worst, who)
    if best is None:
        x0 = 0.62
        worst, who = bar_top_frac(ax, nm, s1, st, c1, ct, x0, 0.98)
        worst = worst if np.isfinite(worst) else 0.0
        y0 = max(0.20, worst + LBL_GAP)
        best = (0.0, (x0, y0, 0.36, max(0.30, top - y0)), worst, who)
    return best[1], best[2], best[3]


def draw_inset(ax, res, box, label_values=True, fontsize_lab=6.0,
               ylim=None, inset_title=True):
    iax = ax if box is None else ax.inset_axes(box)
    o, s1, st, c1, ct = sort_by_st(res)
    nm = [K7[i] for i in o]
    x = np.arange(len(nm))
    w = 0.35
    b1 = iax.bar(x - w / 2, s1, w, yerr=c1, capsize=2, ecolor=C_ERR,
                 label='S1', color=C_S1, lw=0)
    b2 = iax.bar(x + w / 2, st, w, yerr=ct, capsize=2, ecolor=C_ERR,
                 label='ST', color=C_ST, lw=0)
    for cont in (b1, b2):
        eb = getattr(cont, 'errorbar', None)
        if eb is not None:
            for ln in eb.get_children():
                try:
                    ln.set_clip_on(True)
                    ln.set_clip_box(iax.bbox)
                except AttributeError:
                    pass
    if ylim is None:
        top = float(np.max(np.maximum(s1 + c1, st + ct)))
        lo = min(float(np.min(np.minimum(s1 - c1, st - ct))), 0.0)
        ymin = lo - 0.03 * (max(0.12, top + 0.30 * (top - lo)) - lo)
        ymax = max(0.12, top + 0.30 * (top - lo))
    else:
        ymin, ymax = float(ylim[0]), float(ylim[1])
    iax.set_ylim(ymin, ymax)
    tk = nice_ticks(ymin, ymax, 4)
    iax.set_yticks(tk)
    iax.set_yticklabels(['%g' % t for t in tk], fontsize=fontsize_lab)
    iax.set_xlim(-0.62, len(nm) - 0.38)
    iax.set_xticks(x)
    iax.set_xticklabels([KG[n] for n in nm], fontsize=fontsize_lab + 1.2)
    iax.tick_params(axis='y', labelsize=fontsize_lab)
    if label_values:
        for i in range(len(nm)):
            if st[i] < 0.01:
                continue
            above = st[i] + ct[i] + 0.02 * (ymax - ymin)
            if st[i] + ct[i] > 0.97 * ymax:
                iax.text(x[i] + w / 2, 0.90 * ymax, '%.2f' % st[i],
                         ha='center', va='top', fontsize=fontsize_lab + 0.6,
                         color='white', zorder=6)
            else:
                iax.text(x[i] + w / 2, above, '%.2f' % st[i], ha='center',
                         va='bottom', fontsize=fontsize_lab - 0.4,
                         rotation=90, color='#111111', zorder=6)
    for sp in ('top', 'right'):
        iax.spines[sp].set_visible(False)
    for sp in ('left', 'bottom'):
        iax.spines[sp].set_linewidth(0.7)
    if inset_title:
        iax.set_title(INSET_TITLE, fontsize=fontsize_lab + 1.2, pad=1.5)
    return iax


def rank_text(res, thresh=0.015, sym=True):
    st = np.array(res['ST'], dtype=float)
    o = np.argsort(-st)
    used, parts = set(), []
    for i in o:
        nm = K7[i]
        if nm in used:
            continue
        pair = None
        if sym:
            for a, b in EXCH:
                if nm in (a, b):
                    pair = (a, b)
        if pair is not None:
            used.update(pair)
            v = 0.5 * (st[K7.index(pair[0])] + st[K7.index(pair[1])])
            if v >= thresh:
                parts.append('%s$\\equiv$%s %.2f'
                             % (KG[pair[0]], KG[pair[1]], v))
        else:
            used.add(nm)
            if st[i] >= thresh:
                parts.append('%s %.2f' % (KG[nm], st[i]))
    tail = [KG[n] for n in K7 if n not in used]
    if tail:
        parts.append(('rest ($%d$) ' % len(tail)) + r'$\lesssim$ %.2f'
                     % thresh)
    return ' $>$ '.join(parts)


def plot_all(d, figsize=(13.5, 10.1), dpi=200, prefix='fig4-final',
             qa=None):
    """Plot and export CSV.  qa is an optional QA callback qa(fig, tag); see
    fig4_selfcheck.py."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 2, figsize=figsize)
    axs = np.array(axs).ravel()
    info = []
    for k, (key, grp, glabel) in enumerate(PANELS):
        ax = axs[k]
        names = d['main_' + grp + '_names']
        res = d['main_results'][key]
        nm, s1, st, c1, ct = main_axes(ax, names, res)
        ax.text(-0.10, 1.04, 'ABCD'[k], transform=ax.transAxes, ha='left',
                va='bottom', fontsize=13, fontweight='bold')
        ax.set_title(TITLES[key], fontsize=11)
        ax.text(0.98, 0.97, glabel, transform=ax.transAxes, ha='right',
                va='top', fontsize=10)
        leg = ax.legend(loc='upper right', bbox_to_anchor=(1.0, 0.90),
                        fontsize=9)
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + 0.13 * (hi - lo))
        lo, hi = ax.get_ylim()
        ax.set_yticks([t for t in ax.get_yticks()
                       if lo - 1e-9 <= t <= hi + 1e-9])
        info.append((ax, key, nm, s1, st, c1, ct, leg))
    fig.tight_layout(h_pad=3.0, w_pad=2.5)

    for ax, key, nm, s1, st, c1, ct, leg in info:
        top = legend_bottom_frac(fig, ax, leg) - 0.05
        box = choose_box(ax, nm, s1, st, c1, ct, top)[0]
        draw_inset(ax, d['within_results'][key], box, ylim=INSET_YLIM)

    if qa is not None:
        qa(fig, prefix)
    for ext in ('.png', '.pdf'):
        fig.savefig(os.path.join(HERE, prefix + ext), dpi=dpi,
                    bbox_inches='tight')
    plt.close(fig)


    fig2, ax2 = plt.subplots(1, 4, figsize=(16.5, 4.1))
    for k, (key, grp, glabel) in enumerate(PANELS):
        a = ax2[k]
        draw_inset(a, d['within_results'][key], None, label_values=True,
                   fontsize_lab=7.4, ylim=INSET_YLIM, inset_title=False)
        a.set_ylabel('Sobol index', fontsize=10)
        a.text(0.0, 1.16, 'ABCD'[k], transform=a.transAxes, ha='left',
               va='bottom', fontsize=12, fontweight='bold')
        a.text(0.06, 1.15, TITLES[key], transform=a.transAxes, ha='left',
               va='bottom', fontsize=10.5)
        a.text(0.5, 1.02, rank_text(d['within_results'][key], sym=SYM[key]),
               transform=a.transAxes, ha='center', va='bottom', fontsize=8.0)
        a.legend(loc='upper right', fontsize=8, framealpha=0.9)
    fig2.suptitle('Fig. 4 insets: the merged $K$ split into its 7 thresholds, '
                  'ranked by $S_T$ (only the $K$ family is perturbed, '
                  '$N$ = %s, all other parameters at BASE); all four panels '
                  'share the $y$-scale [%.1f, %.1f]'
                  % (format(d['meta']['N_base'], ','), *INSET_YLIM),
                  fontsize=10.5, y=1.0)
    fig2.text(0.5, 0.035,
              '$K_{SH}$: $S_1=S_T=1.00$ (the $(H,S)$ subsystem is decoupled '
              'from $(L,R)$, so $K_{SH}$ is the only threshold entering $H$; '
              'its 95%% CI $+%.2f$ is clipped at the axis top).   '
              'C and D are mirror images: $K_{SL}\\leftrightarrow K_{SR}$, '
              '$K_{RL}\\leftrightarrow K_{LR}$, $K_L\\leftrightarrow K_R$.'
              % float(np.max(np.array(
                  d['within_results']['H peak/plateau']['ST_conf']))),
              ha='center', va='center', fontsize=8.2, color='#333333')
    fig2.tight_layout()
    fig2.subplots_adjust(bottom=0.19)
    if qa is not None:
        qa(fig2, prefix + '_Kfamily')
    for ext in ('.png', '.pdf'):
        fig2.savefig(os.path.join(HERE, prefix + '_Kfamily' + ext), dpi=dpi,
                     bbox_inches='tight')
    plt.close(fig2)


    p = dict(BASE)
    t, Y = full_trajectory(p)
    H = Y[0]
    i_pk = int(np.argmax(H))
    ratio = H.max() / H[-1]
    fig3, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.plot(t, H, color='#377eb8', lw=1.9, label=r'$H(t)$  (full dynamics)')
    ax.plot(t[i_pk], H[i_pk], 'o', color='#a65628', ms=7, zorder=5,
            label=r'peak  $H_{\rm peak}=%.4f$ @ $t=%.2f$ h'
                  % (H[i_pk], t[i_pk]))
    ax.axhline(H[-1], color='#999999', ls='--', lw=1.2,
               label=r'plateau  $H_{\rm plateau}=%.4f$ @ $t=%.0f$ h'
                     % (H[-1], t[-1]))
    ax.annotate('', xy=(t[i_pk], H[i_pk]), xytext=(t[i_pk], H[-1]),
                arrowprops=dict(arrowstyle='<->', color='#444444', lw=1.1))
    ax.text(t[i_pk] + 0.7, 0.5 * (H[i_pk] + H[-1]),
            r'$\frac{H_{\rm peak}}{H_{\rm plateau}}=%.3f$' % ratio,
            fontsize=12, color='#222222')
    ax.set_xlabel('time  $t$  (h)', fontsize=11)
    ax.set_ylabel('$H$', fontsize=11)
    ax.set_title('Panel B: full-dynamics $H(t)$, $S$ free (not clamped)',
                 fontsize=11)
    ax.legend(loc='upper right', fontsize=9, frameon=False)
    ax.set_xlim(0, t[-1])
    ax.set_ylim(0, H.max() * 1.18)
    ax.tick_params(labelsize=9.5)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    fig3.tight_layout()
    for ext in ('.png', '.pdf'):
        fig3.savefig(os.path.join(HERE, prefix + '_Htrace' + ext), dpi=dpi,
                     bbox_inches='tight')
    plt.close(fig3)

    print('   (B) baseline (BASE): H_peak = %.4f @ %.2f h, H_plateau = %.4f, '
          'ratio = %.4f' % (H[i_pk], t[i_pk], H[-1], ratio))

    rows = []
    for key, grp, glabel in PANELS:
        names = d['main_' + grp + '_names']
        res = d['main_results'][key]
        for i, nm in enumerate(names):
            rows.append([key, grp, nm, res['S1'][i], res['S1_conf'][i],
                         res['ST'][i], res['ST_conf'][i]])
    with open(os.path.join(HERE, prefix + '.csv'), 'w', newline='',
              encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['output', 'group', 'parameter', 'S1', 'S1_95CI', 'ST',
                    'ST_95CI'])
        for r in rows:
            w.writerow(r[:3] + ['%.6f' % v for v in r[3:]])

    irows = []
    for key, grp, glabel in PANELS:
        wk = d['within_results'][key]
        o = list(np.argsort(-np.array(wk['ST'], dtype=float)))
        for rank, i in enumerate(o, 1):
            nm = K7[i]
            part = '-'
            for a, b in EXCH:
                if nm in (a, b):
                    part = b if nm == a else a
            irows.append([key, rank, nm, KROLE[nm],
                          '%.4f' % wk['S1'][i], '%.4f' % wk['S1_conf'][i],
                          '%.4f' % wk['ST'][i], '%.4f' % wk['ST_conf'][i],
                          part])
    with open(os.path.join(HERE, prefix + '_inset.csv'), 'w', newline='',
              encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['output', 'rank_by_ST', 'K parameter', 'role in ODE',
                    'S1', 'S1_95CI', 'ST', 'ST_95CI', 'exchangeable with'])
        w.writerows(irows)

    print('saved %s.png/.pdf/.csv/_inset.csv, %s_Kfamily.png/.pdf, '
          '%s_Htrace.png/.pdf' % (prefix, prefix, prefix))
    for key, grp, glabel in PANELS:
        print('  %-22s %s'
              % (key, rank_text(d['within_results'][key], sym=SYM[key])))


def main():
    ap = argparse.ArgumentParser(
        description='Fig.4 final: Sobol sensitivity analysis of the '
                    'four-variable Volvox model')
    ap.add_argument('--n', type=int, default=4096, help='Saltelli base sample size')
    ap.add_argument('--nproc', type=int, default=8, help='number of worker processes')
    ap.add_argument('--serial', action='store_true', help='single process')
    ap.add_argument('--seed', type=int, default=20241011)
    ap.add_argument('--force', action='store_true', help='ignore the cache and recompute')
    ap.add_argument('--plot-only', action='store_true',
                    help='plot only, reusing the existing fig4-final.json')
    ap.add_argument('--figsize', type=float, nargs=2, default=(13.5, 10.1),
                    metavar=('W', 'H'), help='main-figure size (inches)')
    ap.add_argument('--dpi', type=int, default=200)
    ap.add_argument('--prefix', default='fig4-final')
    args = ap.parse_args()

    jpath = os.path.join(HERE, 'fig4-final.json')
    if args.plot_only:
        with open(jpath, encoding='utf-8') as f:
            d = json.load(f)
        print('read %s (N=%s)' % (os.path.basename(jpath),
                                  format(d['meta']['N_base'], ',')))
    else:
        nproc = 0 if args.serial else args.nproc
        d = compute(args.n, nproc, args.seed, args.force)

    plot_all(d, figsize=tuple(args.figsize), dpi=args.dpi,
             prefix=args.prefix)


if __name__ == '__main__':
    main()
