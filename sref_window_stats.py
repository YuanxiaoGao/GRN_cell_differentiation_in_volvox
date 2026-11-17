import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
F4DIR = os.path.abspath(os.path.join(HERE, '..', 'fig4'))
sys.path.insert(0, F4DIR)

import fig4_final as F
A_REF, B_REF = 3.0, 0.6
N_SAMP = 150
SEED = 7
CSV_OUT = os.path.join(HERE, 'sref_window_stats.csv')


def sref_of(p, a=A_REF, b=B_REF):
    """Return (S_th, S_hi, S_ref) for one parameter set, or None."""
    w = F.bistable_window(p)
    if w is None:
        return None
    S_th, S_hi = w
    v = min(a * S_th, b * S_hi) if S_hi > 0 else a * S_th
    return S_th, S_hi, float(np.clip(v, S_th * 1.02, S_hi * 0.98))


def _one(item):
    i, p = item
    r = sref_of(p)
    if r is None:
        return (i, np.nan, np.nan, np.nan)
    return (i,) + r


def draw_params(n=N_SAMP, seed=SEED):
    rng = np.random.default_rng(seed)
    keys = list(F.BOUNDS.keys())
    out = []
    for _ in range(n):
        p = dict(F.BASE)
        for k in keys:
            lo, hi = F.BOUNDS[k]
            p[k] = float(lo + rng.random() * (hi - lo))
        out.append(p)
    return out


def run_stats(n=N_SAMP):
    ps = draw_params(n)
    import multiprocessing as mp
    nproc = max(1, min(8, os.cpu_count() or 1))
    with mp.Pool(nproc) as pool:
        rows = pool.map(_one, list(enumerate(ps)), chunksize=1)
    rows = [r for r in rows if np.isfinite(r[1])]
    arr = np.array([[r[1], r[2], r[3]] for r in rows])

    with open(CSV_OUT, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['sample', 'S_th', 'S_hi', 'S_ref'])
        for r in rows:
            w.writerow(['%d' % r[0], '%.6f' % r[1], '%.4f' % r[2], '%.6f' % r[3]])

    base = sref_of(dict(F.BASE))
    S0 = base[2]
    n_out = int((arr[:, 0] > S0).sum())
    print('samples with a bistable window : %d / %d' % (len(arr), n))
    print('S_th   min=%.4f  p5=%.4f  median=%.4f  p95=%.4f  max=%.4f'
          % (arr[:, 0].min(), np.percentile(arr[:, 0], 5), np.median(arr[:, 0]),
             np.percentile(arr[:, 0], 95), arr[:, 0].max()))
    print('S_ref  min=%.4f  p5=%.4f  median=%.4f  p95=%.4f  max=%.4f  (span %.1fx)'
          % (arr[:, 2].min(), np.percentile(arr[:, 2], 5), np.median(arr[:, 2]),
             np.percentile(arr[:, 2], 95), arr[:, 2].max(),
             arr[:, 2].max() / arr[:, 2].min()))
    print('S_hi pinned at the grid edge (60)  : %d / %d (%.1f%%)'
          % (int(np.isclose(arr[:, 1], 60.0).sum()), len(arr),
             100 * np.isclose(arr[:, 1], 60.0).mean()))
    print('reference S_ref (BASE)             : %.4f' % S0)
    print('samples whose window excludes S0   : %d / %d (%.1f%%)'
          % (n_out, len(arr), 100 * n_out / len(arr)))
    print('upper branch b*S_hi in use         : %d / %d (%.2f%%)'
          % (int((arr[:, 2] < A_REF * arr[:, 0] - 1e-9).sum()), len(arr),
             100 * (arr[:, 2] < A_REF * arr[:, 0] - 1e-9).mean()))
    print('CSV written to %s' % CSV_OUT)
    return arr, S0


if __name__ == '__main__':
    run_stats()
