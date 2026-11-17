# -*- coding: utf-8 -*-
import numpy as np
from scipy.optimize import root, brentq
from scipy.integrate import solve_ivp

GAMMA = 0.35
BASE = {
    'aH': 3.5, 'aL': 3.5, 'eta': 0.7, 'beta': 0.0035,
    'K_SH': 0.5, 'K_SL': 0.5, 'K_SR': 0.5,
    'K_RL': 0.5, 'K_LR': 0.5, 'K_L': 0.5, 'K_R': 0.5,
    'Kc': 1.5, 'delta': 7.0, 'p': 3, 'n': 3, 'm': 3, 'q': 2,
    'b0': 0.055, 'bH': 0.9, 'Khs': 0.5, 'gS': 0.1,
}

NAMES = ['aH', 'aL', 'eta', 'beta',
         'K_SH', 'K_SL', 'K_SR', 'K_RL', 'K_LR', 'K_L', 'K_R',
         'Kc', 'delta', 'p', 'n', 'm', 'q', 'b0', 'bH', 'Khs', 'gS']

GREEK = {
    'aH': r'$\alpha_H$', 'aL': r'$\alpha_L$', 'eta': r'$\eta$',
    'beta': r'$\beta$',
    'K_SH': r'$K_{SH}$', 'K_SL': r'$K_{SL}$', 'K_SR': r'$K_{SR}$',
    'K_RL': r'$K_{RL}$', 'K_LR': r'$K_{LR}$',
    'K_L': r'$K_{L}$', 'K_R': r'$K_{R}$',
    'Kc': r'$K_c$', 'delta': r'$\delta$',
    'p': r'$p$', 'n': r'$n$', 'm': r'$m$', 'q': r'$q$',
    'b0': r'$\beta_0$', 'bH': r'$\beta_H$', 'Khs': r'$K_{HS}$',
    'gS': r'$\gamma_S$',
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

OUTPUT_NAMES = ['S_th', 'log10(L_high/R_low)', 'H peak/plateau']
OUTPUT_TITLES = [r'$S_{\rm th}$',
                 r'$\log_{10}(L^*_{\rm high}/R^*_{\rm low})$',
                 r'$H^*_{\rm peak}/H^*_{\rm plateau}$']

K_FAMILY = ['K_SH', 'K_SL', 'K_SR', 'K_RL', 'K_LR', 'K_L', 'K_R']


def build_p(values, names=None):
    p = dict(BASE)
    if names is None:
        names = NAMES
    for nm, val in zip(names, values):
        p[nm] = float(val)
    return p

def lr_rhs(L, R, S, p):
    L = max(L, 0.0)
    R = max(R, 0.0)
    n, pe, q = p['n'], p['p'], p['q']
    FL = S**n / (p['K_SL']**n + S**n)
    FR = S**n / (p['K_SR']**n + S**n)
    rR = p['K_RL']**pe / (p['K_RL']**pe + R**pe)  # R -| L
    rL = p['K_LR']**pe / (p['K_LR']**pe + L**pe)  # L -| R
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
    J = np.array([
        [FL * p['eta'] * dhL * rR - GAMMA,
         FL * (p['aL'] + p['eta'] * hL) * dgR],
        [FR * (p['aL'] + p['eta'] * hR) * dgL,
         FR * p['eta'] * dhR * rL - GAMMA],
    ])
    return J


GUESS = [[10.0, 0.01], [0.01, 10.0], [1.0, 1.0],
         [5.0, 5.0], [0.1, 8.0], [8.0, 0.1]]


def stable_states(S, p, guesses=None):
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


def num_stable(S, p):
    return len(stable_states(S, p))


def _bisect_flag(f, lo, hi, iters):
    flo = f(lo)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if f(mid) == flo:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


_SGRID = np.geomspace(0.03, 60.0, 40)


def bistable_window(p, sgrid=None, iters=20):
    """Generic bistable window [S_lo, S_hi]: the interval in which the number of
    stable equilibria goes 1 -> >=2 -> 1.

    Does not rely on L/R symmetry (K_SL and K_SR may be perturbed
    independently).  Returns (S_lo, S_hi) or None."""
    if sgrid is None:
        sgrid = _SGRID
    cnt = [num_stable(S, p) for S in sgrid]
    idx2 = [i for i, c in enumerate(cnt) if c >= 2]
    if not idx2:
        return None
    i0, i1 = idx2[0], idx2[-1]
    f = lambda z: num_stable(z, p) >= 2           # noqa: E731
    lo = sgrid[0] if i0 == 0 else _bisect_flag(f, sgrid[i0 - 1], sgrid[i0],
                                              iters)
    hi = (sgrid[-1] if i1 == len(sgrid) - 1
          else _bisect_flag(f, sgrid[i1], sgrid[i1 + 1], iters))
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
        return None
    return float(lo), float(hi)


def analyze_lr(p, Smax=60.0):
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
    L_high, R_low = st[i_L]
    ratios = [max(x[0] / max(x[1], 1e-12), x[1] / max(x[0], 1e-12))
              for x in st]
    y2 = float(np.log10(max(ratios)))
    return {'S_th': float(S_th), 'S_hi': float(S_hi),
            'S_ref': float(S_ref), 'y2': y2,
            'L_high': float(L_high), 'R_low': float(R_low),
            'n_stable': len(st)}


def full_dynamics_ratio(p, T_MAX=36.0, rtol=1e-9, atol=1e-11, max_step=0.1):
    def rhs(t, y):
        H, S, L, R = y
        L = max(L, 0.0)
        R = max(R, 0.0)
        n, pe, q = p['n'], p['p'], p['q']
        FL = S**n / (p['K_SL']**n + S**n)          # S -> L
        FR = S**n / (p['K_SR']**n + S**n)          # S -> R
        FH = S**n / (p['K_SH']**n + S**n)          # S -> H
        G = S**p['m'] / (p['Kc']**p['m'] + S**p['m'])
        dH = p['aH'] * FH - GAMMA * H - p['delta'] * G * H
        dS = (p['b0'] + p['bH'] * H**n / (p['Khs']**n + H**n)
              - p['gS'] * S)
        rR = p['K_RL']**pe / (p['K_RL']**pe + R**pe)
        rL = p['K_LR']**pe / (p['K_LR']**pe + L**pe)
        hL = L**q / (p['K_L']**q + L**q)
        hR = R**q / (p['K_R']**q + R**q)
        dL = (p['beta'] + FL * (p['aL'] + p['eta'] * hL) * rR - GAMMA * L)
        dR = (p['beta'] + FR * (p['aL'] + p['eta'] * hR) * rL - GAMMA * R)
        return [dH, dS, dL, dR]

    t_pre = np.arange(0, 3.000001, 0.25)
    t_post = np.arange(3.0, T_MAX + 0.000001, 0.25)
    y0 = [0.01, 0.05, 0.05, 0.05]
    pre = solve_ivp(rhs, [0, 3.0], y0, t_eval=t_pre, method='LSODA',
                    max_step=max_step, rtol=rtol, atol=atol)
    y3 = pre.y[:, -1].copy()
    y3[2] += 0.02
    y3[3] -= 0.02
    post = solve_ivp(rhs, [3.0, T_MAX], y3, t_eval=t_post,
                     method='LSODA', max_step=max_step, rtol=rtol, atol=atol)
    H = np.hstack([pre.y[0], post.y[0][1:]])
    return float(H.max() / max(H[-1], 1e-12))


def model_output(x, compute_dyn=True, names=None, need_lr=True):
    p = build_p(x, names)
    if need_lr:
        lr = analyze_lr(p)
        if lr is None:
            return [np.nan, np.nan, np.nan]
        out = [lr['S_th'], lr['y2'], np.nan]
    else:
        out = [np.nan, np.nan, np.nan]
    if compute_dyn:
        try:
            out[2] = full_dynamics_ratio(p)
        except Exception:
            out[2] = np.nan
    return out

if __name__ == '__main__':
    import os
    import sys
    import time

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        import model_v6 as M6
        o6 = M6.model_output([M6.BASE[n] for n in M6.NAMES], compute_dyn=True)
        o7 = model_output([BASE[n] for n in NAMES], compute_dyn=True)
        print('=== BASE equivalence (model_v6 symmetric vs model_v7 split) ===')
        print('  v6: S_th=%.10f  log10(L/R)=%.10f  H=%.10f' % tuple(o6))
        print('  v7: S_th=%.10f  log10(L/R)=%.10f  H=%.10f' % tuple(o7))
        for i, nm in enumerate(['S_th', 'log10(L/R)', 'H']):
            a, b = o6[i], o7[i]
            if np.isfinite(a) and np.isfinite(b) and abs(a) > 1e-14:
                print('   %-10s rel diff = %.3e' % (nm, abs(a - b) / abs(a)))
            else:
                print('   %-10s abs diff = %.3e' % (nm, abs(a - b)))
    except Exception as e:
        print('[skip] model_v6 comparison:', e)

    print('\n=== exchangeability check (pairwise K swap) ===')
    rng = np.random.default_rng(7)
    diffs = []
    for _ in range(16):
        names = ['aL', 'eta', 'beta', 'K_SL', 'K_SR', 'K_RL', 'K_LR',
                'K_L', 'K_R', 'p', 'n', 'q']
        v = [float(rng.uniform(*BOUNDS[n])) for n in names]
        j = {n: i for i, n in enumerate(names)}
        for nm in ['K_SL', 'K_SR', 'K_RL', 'K_LR', 'K_L', 'K_R']:
            v[j[nm]] = float(rng.uniform(0.45, 1.35))
        w = list(v)
        for a, b in [('K_SL', 'K_SR'), ('K_RL', 'K_LR'), ('K_L', 'K_R')]:
            w[j[a]], w[j[b]] = v[j[b]], v[j[a]]
        o1 = model_output(v, compute_dyn=False, names=names)
        o2 = model_output(w, compute_dyn=False, names=names)
        if all(np.isfinite(o1)) and all(np.isfinite(o2)):
            diffs.append(max(abs(o1[0] - o2[0]), abs(o1[1] - o2[1])))
    print('   pairs tested = %d, max |output diff| = %.3e'
          % (len(diffs), max(diffs) if diffs else float('nan')))

    rng = np.random.default_rng(1)
    t0 = time.time()
    nan = 0
    NT = 30
    for _ in range(NT):
        x = [BOUNDS[n][0] + (BOUNDS[n][1] - BOUNDS[n][0]) * rng.random()
             for n in NAMES]
        o = model_output(x, compute_dyn=False)
        if np.any(np.isnan(o)):
            nan += 1
    dt = (time.time() - t0) / NT
    print('\nserial: %.1f ms/sample (no dynamics); NaN rate %d/%d'
          % (dt * 1e3, nan, NT))
