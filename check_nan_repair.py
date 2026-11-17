import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
F4DIR = os.path.abspath(os.path.join(HERE, '..', 'fig4'))
sys.path.insert(0, F4DIR)

import fig4_final as F
from SALib.sample import saltelli

SEED = 20241011
N = 4096

DESIGNS = [
    ('main',   'lr', F.LR_NAMES),
    ('main',   'hs', F.HS_NAMES),
    ('within', 'lr', F.K7),
    ('within', 'hs', F.K7),
]


def main():
    total_bad = 0
    for prefix, tag, names in DESIGNS:
        problem = F.make_problem(names)
        np.random.seed(SEED)
        X_fresh = saltelli.sample(problem, N, calc_second_order=False)
        X_cache = np.load(F.cache_path(N, prefix, tag))['X']
        assert X_fresh.shape == X_cache.shape, (prefix, tag)
        bad = np.any(X_fresh != X_cache, axis=1)
        total_bad += int(bad.sum())
        print('%-6s %-3s rows=%5d  repaired=%d  max|dX|=%.3g'
              % (prefix, tag, len(X_cache), bad.sum(),
                 np.abs(X_fresh - X_cache).max()))
    print('total repaired rows: %d  -> NaN repair %s in all production '
          'designs' % (total_bad, 'never triggered'
                       if total_bad == 0 else 'WAS triggered'))
    return total_bad


if __name__ == '__main__':
    sys.exit(0 if main() == 0 else 1)
