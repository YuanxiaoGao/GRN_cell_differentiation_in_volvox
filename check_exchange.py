import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model_v7 as M  # noqa: E402

rng = np.random.default_rng(2026)
NAMES = ['aL', 'eta', 'beta', 'K_SL', 'K_SR', 'K_RL', 'K_LR', 'K_L', 'K_R',
         'p', 'n', 'q']
J = {n: i for i, n in enumerate(NAMES)}

diffs_S, diffs_Y = [], []
bad = 0
for t in range(60):
    v = [float(rng.uniform(*M.BOUNDS[n])) for n in NAMES]
    w = list(v)
    for a, b in [('K_SL', 'K_SR'), ('K_RL', 'K_LR'), ('K_L', 'K_R')]:
        w[J[a]], w[J[b]] = v[J[b]], v[J[a]]
    o1 = M.model_output(v, compute_dyn=False, names=NAMES)
    o2 = M.model_output(w, compute_dyn=False, names=NAMES)
    if not (np.all(np.isfinite(o1[:2])) and np.all(np.isfinite(o2[:2]))):
        bad += 1
        continue
    diffs_S.append(abs(o1[0] - o2[0]) / max(abs(o1[0]), 1e-12))
    diffs_Y.append(abs(o1[1] - o2[1]) / max(abs(o1[1]), 1e-12))

print('tested %d pairs (skipped %d non-finite)' % (len(diffs_S), bad))
print('  S_th      : median rel diff %.3e   max %.3e'
      % (np.median(diffs_S), np.max(diffs_S)))
print('  |log L/R| : median rel diff %.3e   max %.3e'
      % (np.median(diffs_Y), np.max(diffs_Y)))
