import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', 'fig4')))

import fig4_final as F

N = 512
SEED = 20241011
S_FIX = 0.6
NAMES = F.LR_NAMES
WHAT = ['S_th', 'log10(L_high/R_low)', 'log10(R_high/R_low)']


def ratio_at(S, p):
    st = F.stable_states(S, p)
    if len(st) < 2:
        return None
    iL = max(range(len(st)), key=lambda k: st[k][0] - st[k][1])
    iR = max(range(len(st)), key=lambda k: st[k][1] - st[k][0])
    if iL == iR:
        return None
    Lh, Rl = st[iL]
    Ll, Rh = st[iR]
    return (float(np.log10(Lh / max(Rl, 1e-12))),
            float(np.log10(Rh / max(Ll, 1e-12))))


def row_A(r):
    p = F.build_p(F.expand(r, NAMES), F.FULL)
    out = F.lr_outputs(p)
    return [np.nan, np.nan, np.nan] if out is None else list(out)


def row_B(r, S=S_FIX):
    p = F.build_p(F.expand(r, NAMES), F.FULL)
    rr = ratio_at(S, p)
    return [np.nan, np.nan] if rr is None else list(rr)


def ev(fn, rows, nproc=8):
    from multiprocessing import Pool
    with Pool(nproc) as pool:
        return np.array(pool.map(fn, list(rows), chunksize=32), dtype=float)


def row_B0(r):
    return row_B(r)


def problem():
    return {'num_vars': len(NAMES), 'names': NAMES,
            'bounds': [F.bounds_of(n) for n in NAMES]}


def Si(prob, y, seed=SEED):
    from SALib.analyze import sobol
    try:
        return sobol.analyze(prob, y, calc_second_order=False, seed=seed)
    except TypeError:
        np.random.seed(seed)
        return sobol.analyze(prob, y, calc_second_order=False)

def main():
    from SALib.sample import saltelli
    np.random.seed(SEED)
    X0 = saltelli.sample(problem(), N, calc_second_order=False)
    print('Saltelli design: N=%d, D=%d -> %d rows' % (N, len(NAMES), len(X0)))

    t0 = time.time()
    YA = ev(row_A, X0)
    print('rule A (per-sample S_ref) took %.0f s, NaN rows = %d'
          % (time.time() - t0, int(np.isnan(YA[:, 0]).sum())))

    t0 = time.time()
    YB = ev(row_B0, X0)
    n_bad0 = int(np.any(np.isnan(YB), axis=1).sum())
    print('rule B (fixed S=%.1f) took %.0f s' % (S_FIX, time.time() - t0))
    print('  -> rows failing directly in the raw design: %d / %d = %.1f%%'
          % (n_bad0, len(X0), 100.0 * n_bad0 / len(X0)))

    ok = np.isfinite(YA[:, 1]) & np.isfinite(YB[:, 1])
    print('\n=== y_B vs y_A on the %d rows where both are defined ===' % int(ok.sum()))
    for j, bj, nm in ((1, 0, 'log10(L_high/R_low)'),
                      (2, 1, 'log10(R_high/R_low)')):
        a, b = YA[ok, j], YB[ok, bj]
        print('  %-22s mean %.4f vs %.4f   Δ=%.4f   Pearson r=%.4f'
              % (nm, a.mean(), b.mean(), (b - a).mean(),
                 np.corrcoef(a, b)[0, 1]))
    sth_ok = np.log10(YA[ok, 0])
    print('  corr(y_A, log10 S_th) = %+.3f' % np.corrcoef(YA[ok, 1], sth_ok)[0, 1])
    print('  corr(y_B, log10 S_th) = %+.3f   <- with S fixed, y degenerates into a proxy for S_th'
          % np.corrcoef(YB[ok, 0], sth_ok)[0, 1])

    XB = X0.copy()
    base_vec = np.array([F.base_of(n) for n in NAMES], dtype=float)
    halves = np.zeros(len(XB), dtype=int)
    for _ in range(6):
        bad = np.any(np.isnan(YB), axis=1)
        if not bad.any():
            break
        XB[bad] = 0.5 * XB[bad] + 0.5 * base_vec
        halves[bad] += 1
        YB[bad] = ev(row_B0, XB[bad])
    print('\n=== after repair (halving towards BASE, at most 6 times) ===')
    print('  rows moved: %d / %d = %.1f%%'
          % (int((halves > 0).sum()), len(XB), 100.0 * (halves > 0).mean()))
    print('  distribution of halving counts:', {int(k): int(v) for k, v in
                              zip(*np.unique(halves, return_counts=True))})
    d0 = np.abs(X0 - base_vec).mean(axis=1)
    d1 = np.abs(XB - base_vec).mean(axis=1)
    print('  remaining NaN: %d' % int(np.any(np.isnan(YB), axis=1).sum()))
    print('  design compressed: mean row displacement |X-BASE| keeps only %.1f%% of the original'
          % (100.0 * d1.mean() / d0.mean()))

    prob = problem()
    print('\n=== Sobol total-order index ST (output: log10(L_high/R_low)) ===')
    print('%-8s %-12s %-12s %s' % ('param', 'A: per-sample', 'B: fixed 0.6', 'Δ'))
    sa, sb = Si(prob, YA[:, 1]), Si(prob, YB[:, 1])
    for i, nm in enumerate(NAMES):
        print('%-8s %-12.4f %-12.4f %+.4f'
              % (nm, sa['ST'][i], sb['ST'][i], sb['ST'][i] - sa['ST'][i]))
    oa = list(np.argsort(-np.array(sa['ST'])))
    ob = list(np.argsort(-np.array(sb['ST'])))
    print('  ranking A: %s' % ' > '.join(NAMES[i] for i in oa))
    print('  ranking B: %s' % ' > '.join(NAMES[i] for i in ob))
    print('  ΣST   A = %.3f   B = %.3f'
          % (float(np.sum(sa['ST'])), float(np.sum(sb['ST']))))

    sth = Si(prob, YA[:, 0])
    print('\n=== key control: ST of the output S_th (identical under both rules) ===')
    print('%-8s %-12s %-12s %s' % ('param', 'ST of S_th', 'ST of B', 'ratio'))
    for i, nm in enumerate(NAMES):
        print('%-8s %-12.4f %-12.4f %.2f'
              % (nm, sth['ST'][i], sb['ST'][i],
                 sb['ST'][i] / sth['ST'][i] if sth['ST'][i] > 1e-9 else np.nan))
    va, vb, vs = (np.array(sa['ST']), np.array(sb['ST']), np.array(sth['ST']))
    print('  corr(ST_A, ST_Sth) = %+.3f' % np.corrcoef(va, vs)[0, 1])
    print('  corr(ST_B, ST_Sth) = %+.3f  <- with S fixed this is close to a copy of S_th'
          % np.corrcoef(vb, vs)[0, 1])
    print('\n(the S_th and H peak/plateau outputs do not depend on where S is read, so both rules give identical values)')
if __name__ == '__main__':
    main()
