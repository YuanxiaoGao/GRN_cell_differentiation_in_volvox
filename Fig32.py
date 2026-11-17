import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, root
from SALib.sample import morris as morris_sample
from SALib.analyze import morris as morris_analyze

OUT = os.path.dirname(os.path.abspath(__file__))

GAMMA = 0.35

NAMES = [
    'aH', 'aL', 'eta', 'beta', 'K', 'Kc', 'delta',
    'p', 'n', 'm', 'q', 'b0', 'bH', 'Khs', 'gS',
]
GREEK_NAMES = [
    r'$\alpha_H$', r'$\alpha_L$', r'$\eta$', r'$\beta$', r'$K$',
    r'$K_c$', r'$\delta$', r'$p$', r'$n$', r'$m$', r'$q$',
    r'$\beta_0$', r'$\beta_H$', r'$K_{HS}$', r'$\gamma_S$',
]
BOUNDS = [
    [1.75, 5.25], [1.75, 5.25], [0.35, 1.05], [0.00175, 0.00525],
    [0.5, 1.5], [1.0, 2.0], [3.5, 10.5],
    [2, 4], [2, 4], [2, 4], [1, 3],
    [0.03, 0.08], [0.5, 1.3], [0.3, 0.8], [0.05, 0.2],
]

BASE = {
    'aH': 3.5, 'aL': 3.5, 'eta': 0.7, 'beta': 0.0035, 'K': 0.5,
    'Kc': 1.5, 'delta': 7.0, 'p': 3, 'n': 3, 'm': 3, 'q': 2,
    'b0': 0.055, 'bH': 0.9, 'Khs': 0.5, 'gS': 0.1,
}

OUTPUT_NAMES = [
    'S_th', 'log10(L_high/R_low)', 'H peak/plateau',
    'log10(R_high/R_low)',
]
OUTPUT_TITLES = [
    r'$S_{\rm th}$',
    r'$\log_{10}(L^*_{\rm high}/R^*_{\rm low})$',
    r'$H^*_{\rm peak}/H^*_{\rm plateau}$',
    r'$\log_{10}(R^*_{\rm high}/R^*_{\rm low})$',
]
COLORS = ['#377eb8', '#35978f', '#a65628', '#984ea3']


def build_p(x):
    p = dict(BASE)
    for name, value in zip(NAMES, x):
        p[name] = float(value)
    return p


def lr_rhs(L, R, S, p):
    L = max(L, 0.0)
    R = max(R, 0.0)
    n = p['n']
    F = S**n / (p['K']**n + S**n)
    rR = p['K']**p['p'] / (p['K']**p['p'] + R**p['p'])
    rL = p['K']**p['p'] / (p['K']**p['p'] + L**p['p'])
    hL = L**p['q'] / (p['K']**p['q'] + L**p['q'])
    hR = R**p['q'] / (p['K']**p['q'] + R**p['q'])
    dL = (p['beta'] + F * (p['aL'] + p['eta'] * hL) * rR - GAMMA * L)
    dR = (p['beta'] + F * (p['aL'] + p['eta'] * hR) * rL - GAMMA * R)
    return dL, dR


def sym_scalar(x, S, p):
    n = p['n']
    F = S**n / (p['K']**n + S**n)
    g = p['K']**p['p'] / (p['K']**p['p'] + x**p['p'])
    h = x**p['q'] / (p['K']**p['q'] + x**p['q'])
    return p['beta'] + F * (p['aL'] + p['eta'] * h) * g - GAMMA * x


def sym_antisym_eig(S, p):
    try:
        x = brentq(lambda z: sym_scalar(z, S, p), 1e-6, 60.0,
                   xtol=1e-12, rtol=1e-12)
    except ValueError:
        return None, None
    n = p['n']
    F = S**n / (p['K']**n + S**n)
    g = p['K']**p['p'] / (p['K']**p['p'] + x**p['p'])
    h = x**p['q'] / (p['K']**p['q'] + x**p['q'])
    dg = (-p['p'] * p['K']**p['p'] * x**(p['p'] - 1)
          / (p['K']**p['p'] + x**p['p'])**2)
    dh = (p['q'] * p['K']**p['q'] * x**(p['q'] - 1)
          / (p['K']**p['q'] + x**p['q'])**2)
    a = F * p['eta'] * dh * g - GAMMA
    b = F * (p['aL'] + p['eta'] * h) * dg
    return a - b, x


def s_threshold(p, S_max=10.0):
    Svals = np.geomspace(0.01, S_max, 60)
    prev = None
    prev_eig = None
    for S in Svals:
        eig, _ = sym_antisym_eig(S, p)
        if eig is None:
            continue
        if eig > 0 and prev_eig is not None and prev_eig <= 0:
            try:
                thr = brentq(lambda z: sym_antisym_eig(z, p)[0],
                             prev, S, xtol=1e-10)
                return float(thr)
            except ValueError:
                return float(S)
        prev, prev_eig = S, eig
    return float(S_max)


def stable_states_at_s5(p):
    S = 5.0
    sols = []
    for guess in ([10.0, 0.01], [0.01, 10.0], [1.0, 1.0]):
        r = root(lambda xy: np.array(lr_rhs(xy[0], xy[1], S, p)),
                 guess, method='hybr', options={'xtol': 1e-12})
        if not r.success:
            continue
        x = r.x
        if x[0] < 0 or x[1] < 0:
            continue
        if any(np.max(np.abs(x - s)) < 1e-4 for s in sols):
            continue
        # Jacobian eigenvalues for stability
        n = p['n']
        F = S**n / (p['K']**n + S**n)
        L, R = x
        rR = p['K']**p['p'] / (p['K']**p['p'] + R**p['p'])
        rL = p['K']**p['p'] / (p['K']**p['p'] + L**p['p'])
        hL = L**p['q'] / (p['K']**p['q'] + L**p['q'])
        hR = R**p['q'] / (p['K']**p['q'] + R**p['q'])
        dhL = (p['q'] * p['K']**p['q'] * L**(p['q'] - 1)
               / (p['K']**p['q'] + L**p['q'])**2)
        dhR = (p['q'] * p['K']**p['q'] * R**(p['q'] - 1)
               / (p['K']**p['q'] + R**p['q'])**2)
        dgR = (-p['p'] * p['K']**p['p'] * R**(p['p'] - 1)
               / (p['K']**p['p'] + R**p['p'])**2)
        dgL = (-p['p'] * p['K']**p['p'] * L**(p['p'] - 1)
               / (p['K']**p['p'] + L**p['p'])**2)
        J = np.array([
            [F * p['eta'] * dhL * rR - GAMMA,
             F * (p['aL'] + p['eta'] * hL) * dgR],
            [F * (p['aL'] + p['eta'] * hR) * dgL,
             F * p['eta'] * dhR * rL - GAMMA],
        ])
        ev = np.linalg.eigvals(J)
        if np.all(np.real(ev) < 0):
            sols.append(x)
    return sols


def full_dynamics_ratio(p):
    def rhs(t, y):
        H, S, L, R = y
        L = max(L, 0.0)
        R = max(R, 0.0)
        F = S**p['n'] / (p['K']**p['n'] + S**p['n'])
        G = S**p['m'] / (p['Kc']**p['m'] + S**p['m'])
        dH = p['aH'] * F - GAMMA * H - p['delta'] * G * H
        dS = (p['b0'] + p['bH'] * H**p['n']
              / (p['Khs']**p['n'] + H**p['n']) - p['gS'] * S)
        rR = p['K']**p['p'] / (p['K']**p['p'] + R**p['p'])
        rL = p['K']**p['p'] / (p['K']**p['p'] + L**p['p'])
        dL = (p['beta'] + F * (p['aL'] + p['eta'] * L**p['q']
              / (p['K']**p['q'] + L**p['q'])) * rR - GAMMA * L)
        dR = (p['beta'] + F * (p['aL'] + p['eta'] * R**p['q']
              / (p['K']**p['q'] + R**p['q'])) * rL - GAMMA * R)
        return [dH, dS, dL, dR]

    T_MAX = 36.0
    t_pre = np.arange(0, 3.000001, 0.25)
    t_post = np.arange(3.0, T_MAX + 0.000001, 0.25)
    y0 = [0.01, 0.05, 0.05, 0.05]
    pre = solve_ivp(rhs, [0, 3.0], y0, t_eval=t_pre, method='LSODA',
                    max_step=0.1, rtol=1e-9, atol=1e-11)
    y3 = pre.y[:, -1].copy()
    y3[2] += 0.02
    y3[3] -= 0.02
    post = solve_ivp(rhs, [3.0, T_MAX], y3, t_eval=t_post,
                     method='LSODA', max_step=0.1, rtol=1e-9, atol=1e-11)
    H = np.hstack([pre.y[0], post.y[0][1:]])
    return float(H.max() / max(H[-1], 1e-12))


def model_output(x, compute_dyn=True):
    p = build_p(x)
    y1 = s_threshold(p)
    states = stable_states_at_s5(p)
    if len(states) >= 2:
        ld = max(states, key=lambda s: s[0])
        rd = max(states, key=lambda s: s[1])
        L_high, R_low = ld
        L_low, R_high = rd
        y2 = float(np.log10(max(L_high, 1e-12) / max(R_low, 1e-12)))
        y4 = float(np.log10(max(R_high, 1e-12) / max(L_low, 1e-12)))
        y5 = float(np.log10(max(L_high, 1e-12) / max(L_low, 1e-12)))
    else:
        y2 = y4 = y5 = 0.0
    y3 = full_dynamics_ratio(p) if compute_dyn else np.nan
    return [y1, y2, y3, y4, y5]


def plot_morris(results, problem, fig_name):
    fig, ax = plt.subplots(figsize=(14, 9))
    mu_by_out = []
    sigma_by_out = []
    markers = ['o', 'o', 'o', 'o']
    sizes = [2600, 2200, 2600, 3400]
    edge_colors = ['black', 'black', 'black', COLORS[3]]
    face_colors = [COLORS[0], COLORS[1], COLORS[2], 'none']
    lws = [1.0, 1.0, 1.0, 3.0]
    for j, name in enumerate(OUTPUT_NAMES):
        mu_by_out.append(np.array(results[name]['mu_star']))
        sigma_by_out.append(np.array(results[name]['sigma']))
    for j in [3, 2, 1, 0]:
        mu = mu_by_out[j]
        sigma = sigma_by_out[j]
        ax.scatter(mu, sigma, s=sizes[j], c=face_colors[j], alpha=0.9,
                   edgecolors=edge_colors[j], linewidths=lws[j], zorder=2,
                   marker=markers[j])
        for i, pname in enumerate(problem['names']):
            ax.text(mu[i], sigma[i],
                    GREEK_NAMES[NAMES.index(pname)],
                    ha='center', va='center', color='white',
                    fontsize=16, zorder=3)
    ax.set_xlim(-0.1, 2.0)
    ax.set_ylim(-0.25, 3.15)
    ax.set_xlabel('Mean absolute elementary effect ($\\mu^*$)', fontsize=16)
    ax.set_ylabel('Standard deviation of elementary effects ($\\sigma$)',
                  fontsize=16)
    legend_order = [0, 2, 1, 3]
    handles = [
        Line2D([0], [0], marker=markers[j], linestyle='', markersize=16,
               markerfacecolor=face_colors[j],
               markeredgecolor=edge_colors[j],
               label=OUTPUT_TITLES[j])
        for j in legend_order
    ]
    ax.legend(handles=handles, loc='upper left', fontsize=14,
              framealpha=0.95)
    ax.tick_params(axis='both', labelsize=14)

    ax_ins = fig.add_axes([0.70, 0.16, 0.26, 0.26])
    ins_sizes = [1100, 900, 1100, 1400]
    ins_lws = [0.6, 0.6, 0.6, 1.8]
    for j in [3, 2, 1, 0]:
        mu = mu_by_out[j]
        sigma = sigma_by_out[j]
        mask = ((mu <= 0.06) & (sigma <= 0.1)
                & ((mu > 1e-9) | (sigma > 1e-9)))
        ax_ins.scatter(mu[mask], sigma[mask], s=ins_sizes[j],
                       c=face_colors[j], alpha=0.9,
                       edgecolors=edge_colors[j], linewidths=ins_lws[j],
                       zorder=2, marker=markers[j])
        for i in np.where(mask)[0]:
            ax_ins.text(mu[i], sigma[i],
                        GREEK_NAMES[NAMES.index(problem['names'][i])],
                        ha='center', va='center', color='white',
                        fontsize=12, zorder=3)
    near = set()
    for j in range(len(OUTPUT_NAMES)):
        mu = mu_by_out[j]
        sigma = sigma_by_out[j]
        for i in np.where((mu <= 0.005) & (sigma <= 0.005))[0]:
            near.add(problem['names'][i])
    near_syms = ', '.join(
        GREEK_NAMES[NAMES.index(nm)]
        for nm in sorted(near, key=NAMES.index))
    ax_ins.text(0.02, 0.98, 'near origin:\n' + near_syms,
                transform=ax_ins.transAxes, va='top', ha='left',
                fontsize=9)
    ax_ins.set_xlim(-0.01, 0.06)
    ax_ins.set_ylim(-0.02, 0.1)
    ax_ins.set_xlabel(r'$\mu^*$', fontsize=11)
    ax_ins.set_ylabel(r'$\sigma$', fontsize=11)
    ax_ins.tick_params(labelsize=10)

    fig.subplots_adjust(left=0.07, right=0.98, top=0.97, bottom=0.08)
    ax.text(0.99, 0.99,
            'Because $L$ and $R$ are symmetric,\n'
            r'$\log_{10}(R^*_{\rm high}/L^*_{\rm low}) = '
            r'\log_{10}(L^*_{\rm high}/R^*_{\rm low})$',
            transform=ax.transAxes, ha='right', va='top', fontsize=12)
    fig.savefig(fig_name, dpi=200, bbox_inches='tight')
    plt.close(fig)


def run_morris(N, levels):
    problem = {'num_vars': len(NAMES), 'names': NAMES, 'bounds': BOUNDS}
    X = morris_sample.sample(problem, N, num_levels=levels)
    Y = np.array([model_output(xi) for xi in X])
    results = {}
    for j, name in enumerate(OUTPUT_NAMES):
        Si = morris_analyze.analyze(problem, X, Y[:, j])
        results[name] = {
            'mu_star': [float(v) for v in Si['mu_star']],
            'sigma': [float(v) for v in Si['sigma']],
        }

    fig_name = os.path.join(OUT, 'fig32_morris.png')
    plot_morris(results, problem, fig_name)

    # Keep a parameter if its mu* is above 10% of the max mu* for any output.
    max_mu = {name: max(results[name]['mu_star']) for name in OUTPUT_NAMES}
    lr_outputs = ['S_th', 'log10(L_high/R_low)',
                  'log10(R_high/R_low)']
    hs_outputs = ['H peak/plateau']

    def keep_for(outs):
        kept = []
        for i, pname in enumerate(NAMES):
            if any(results[out]['mu_star'][i] > 0.05 * max_mu[out]
                   for out in outs):
                kept.append(pname)
        return kept

    lr_keep = keep_for(lr_outputs)
    hs_keep = keep_for(hs_outputs)

    out_json = os.path.join(OUT, 'fig32_morris.json')
    with open(out_json, 'w') as f:
        json.dump({'levels': levels, 'N': N,
                   'results': results, 'lr_keep': lr_keep,
                   'hs_keep': hs_keep, 'names': NAMES,
                   'bounds': BOUNDS}, f, indent=2)

    print('Morris screening done: N={}, p={}'.format(N, levels))
    print('L-R output parameters:', lr_keep)
    print('H output parameters:', hs_keep)
    print('Saved', fig_name)
    print('Saved', out_json)
    for name in OUTPUT_NAMES:
        order = np.argsort(results[name]['mu_star'])[::-1]
        top = [(NAMES[i], round(results[name]['mu_star'][i], 4))
               for i in order[:5]]
        print('{} top: {}'.format(name, top))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=120,
                        help='Morris trajectories')
    parser.add_argument('--levels', type=int, default=4,
                        help='number of levels p')
    parser.add_argument('--plot-only', action='store_true',
                        help='regenerate figure from saved JSON')
    args = parser.parse_args()
    if args.plot_only:
        with open(os.path.join(OUT, 'fig32_morris.json'), 'r') as f:
            data = json.load(f)
        problem = {'num_vars': len(NAMES), 'names': NAMES,
                   'bounds': BOUNDS}
        fig_name = os.path.join(OUT, 'fig32_morris.png')
        plot_morris(data['results'], problem, fig_name)
        print('Plot-only saved', fig_name)
    else:
        run_morris(args.n, args.levels)
