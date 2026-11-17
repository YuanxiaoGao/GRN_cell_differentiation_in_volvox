import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, root

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, '_fig5_v3_cache.npz')
CACHE_V2 = os.path.join(HERE, '_fig5_v2_cache.npz')

S_FIN = 5.042
GAMMA = 0.35
SEED = 20241012
N_CURVE = 1000
N_HIST = 3000

SYM = {
    'aH': 3.5, 'aL': 3.5, 'aR': 3.5, 'K': 0.5, 'Kc': 2.275,
    'delta': 7.599, 'n': 3, 'm': 3.874, 'q': 2, 'p': 3,
    'b0': 0.0523, 'bH': 0.634, 'Khs': 0.264, 'gS': 0.116,
    'etaL': 0.7, 'etaR': 0.7, 'bL': 0.003944, 'bR': 0.003944,
    'KL': 0.5, 'KR': 0.5, 'KSL': 0.5, 'KSR': 0.5,
    'KRL': 0.5, 'KLR': 0.5,
}

MECH = ['alpha', 'eta', 'K']
MECH_LABEL = {
    'alpha': r'$\alpha_R/\alpha_L$',
    'eta': r'$\eta_R/\eta_L$',
    'K': r'$K_L/K_R$',
}
STRENGTH_RANGES = {'alpha': (1.0, 3.0), 'eta': (1.0, 5.0), 'K': (1.0, 3.0)}
EFFECT_SCALE = {'alpha': '~22%', 'eta': '~7%', 'K': '<0.1%'}

def full_rhs(t, y, p):
    H, S, L, R = y
    L = max(L, 0.0)
    R = max(R, 0.0)
    FH = S**p['n'] / (p['K']**p['n'] + S**p['n'])
    GH = S**p['m'] / (p['Kc']**p['m'] + S**p['m'])
    FL = S**p['n'] / (p['KSL']**p['n'] + S**p['n'])
    FR = S**p['n'] / (p['KSR']**p['n'] + S**p['n'])
    rR = p['KRL']**p['p'] / (p['KRL']**p['p'] + R**p['p'])
    rL = p['KLR']**p['p'] / (p['KLR']**p['p'] + L**p['p'])
    hL = L**p['q'] / (p['KL']**p['q'] + L**p['q'])
    hR = R**p['q'] / (p['KR']**p['q'] + R**p['q'])
    dH = p['aH'] * FH - GAMMA * H - p['delta'] * GH * H
    dS = (p['b0'] + p['bH'] * H**p['n']
          / (p['Khs']**p['n'] + H**p['n']) - p['gS'] * S)
    dL = p['bL'] + FL * (p['aL'] + p['etaL'] * hL) * rR - GAMMA * L
    dR = p['bR'] + FR * (p['aR'] + p['etaR'] * hR) * rL - GAMMA * R
    return [dH, dS, dL, dR]


def run_full(p, T=40.0):
    t_pre = np.arange(0, 3.000001, 0.25)
    t_post = np.arange(3.0, T + 0.000001, 0.25)
    y0 = [0.01, 0.05, 0.05, 0.05]
    pre = solve_ivp(lambda tt, y: full_rhs(tt, y, p), [0, 3.0], y0,
                    t_eval=t_pre, method='LSODA', max_step=0.1,
                    rtol=1e-9, atol=1e-11)
    y3 = pre.y[:, -1].copy()

    y3[2] += 0.02
    y3[3] -= 0.02
    post = solve_ivp(lambda tt, y: full_rhs(tt, y, p), [3.0, T], y3,
                     t_eval=t_post, method='LSODA', max_step=0.1,
                     rtol=1e-9, atol=1e-11)
    t = np.concatenate([pre.t, post.t[1:]])
    y = np.hstack([pre.y, post.y[:, 1:]])
    return t, y


def lr_stable(S, p):
    """Stable L/R steady state at S.  P3: applies the same non-negativity guard as
    full_rhs, so that a fractional q never raises a negative base to a power
    and produces NaN."""
    def lr(xy):
        L = max(xy[0], 0.0)
        R = max(xy[1], 0.0)
        FL = S**p['n'] / (p['KSL']**p['n'] + S**p['n'])
        FR = S**p['n'] / (p['KSR']**p['n'] + S**p['n'])
        rR = p['KRL']**p['p'] / (p['KRL']**p['p'] + R**p['p'])
        rL = p['KLR']**p['p'] / (p['KLR']**p['p'] + L**p['p'])
        hL = L**p['q'] / (p['KL']**p['q'] + L**p['q'])
        hR = R**p['q'] / (p['KR']**p['q'] + R**p['q'])
        dL = p['bL'] + FL * (p['aL'] + p['etaL'] * hL) * rR - GAMMA * L
        dR = p['bR'] + FR * (p['aR'] + p['etaR'] * hR) * rL - GAMMA * R
        return [dL, dR]

    sols = []
    for guess in ([10.0, 0.01], [0.01, 10.0], [1.0, 1.0]):
        with np.errstate(all='ignore'):
            r = root(lr, guess, method='hybr', options={'xtol': 1e-12})
        if r.success and r.x[0] > 0 and r.x[1] > 0:
            if not any(np.max(np.abs(r.x - s)) < 1e-4 for s in sols):
                sols.append(r.x)
    return sols


def ratios_at(S, p):
    sols = lr_stable(S, p)
    if len(sols) < 2:
        return None, None
    ld = max(sols, key=lambda s: s[0])
    rd = max(sols, key=lambda s: s[1])
    L_high, R_low = ld
    L_low, R_high = rd
    return (np.log10(R_high / max(R_low, 1e-12)),
            np.log10(L_high / max(L_low, 1e-12)))


def tune_beta(p):
    def f(b):
        pp = dict(p)
        pp['bL'] = b
        pp['bR'] = b
        rr, _ = ratios_at(S_FIN, pp)
        return rr if rr is not None else 10.0
    return brentq(lambda x: f(x) - 3.0, 0.0005, 0.02, xtol=1e-10)


def build_param(mech, strength, rng):
    p = dict(SYM)
    if mech == 'alpha':
        p['aL'] = 7.0 / (1.0 + strength)
        p['aR'] = 7.0 * strength / (1.0 + strength)
        eta = rng.uniform(0.5, 0.9)
        p['etaL'] = eta
        p['etaR'] = eta
        p['K'] = rng.uniform(0.4, 0.6)
    elif mech == 'eta':
        p['etaL'] = 0.7
        p['etaR'] = 0.7 * strength
        p['K'] = rng.uniform(0.4, 0.6)
    elif mech == 'K':
        p['KL'] = 0.5
        p['KR'] = 0.5 / strength
        eta = rng.uniform(0.5, 0.9)
        p['etaL'] = eta
        p['etaR'] = eta
    p['p'] = rng.uniform(2.5, 3.5)
    p['q'] = rng.uniform(1.5, 2.5)
    return p


def sample_dist(mech, strength, N, seed=SEED):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(N):
        p = build_param(mech, strength, rng)
        try:
            beta = tune_beta(p)
        except ValueError:
            continue
        p['bL'] = beta
        p['bR'] = beta
        rr, lr = ratios_at(S_FIN, p)
        if rr is None or abs(rr - 3.0) > 0.01:
            continue
        vals.append(lr)
    return np.array(vals)


def summary(vals):
    med = np.median(vals)
    lo, hi = np.percentile(vals, [5, 95])
    return 10**med, 10**lo, 10**hi


def symmetric_ratio():
    p = dict(SYM)
    beta = tune_beta(p)
    p['bL'] = beta
    p['bR'] = beta
    rr, lr = ratios_at(S_FIN, p)
    return 10**rr, 10**lr

def compute():
    out = {}
    t0 = time.time()
    t, y = run_full(SYM, T=40.0)
    out['t'] = t
    out['H'] = y[0]
    out['S'] = y[1]
    out['h_ratio'] = y[0].max() / max(y[0][-1], 1e-12)

    sym_R, sym_L = symmetric_ratio()
    out['sym_R'] = sym_R
    out['sym_L'] = sym_L

    dists = {}
    for mech in MECH:
        dists[mech] = sample_dist(mech, 1.5, N_HIST)
        print('  [%5.1fs] %s @1.5: n=%d' % (time.time() - t0, mech,
                                            len(dists[mech])))
        out['dist_' + mech] = dists[mech]

    curves = {}
    for mech in MECH:
        lo_s, hi_s = STRENGTH_RANGES[mech]
        strengths = np.linspace(lo_s, hi_s, 51)
        meds, los, his = [], [], []
        for s in strengths:
            vals = sample_dist(mech, float(s), N_CURVE)
            if len(vals) == 0:
                meds.append(np.nan)
                los.append(np.nan)
                his.append(np.nan)
                continue
            m, lo, hi = summary(vals)
            meds.append(m)
            los.append(lo)
            his.append(hi)
        curves[mech] = (strengths, np.array(meds), np.array(los),
                        np.array(his))
        print('  [%5.1fs] curve %s done' % (time.time() - t0, mech))
    for mech in MECH:
        s, m, lo, hi = curves[mech]
        out['cur_' + mech + '_s'] = s
        out['cur_' + mech + '_m'] = m
        out['cur_' + mech + '_lo'] = lo
        out['cur_' + mech + '_hi'] = hi
    np.savez(CACHE, **out)
    print('  cache -> %s (%.1f min total)' % (CACHE, (time.time() - t0) / 60))
    return out


def load_cache():
    src = CACHE if os.path.exists(CACHE) else CACHE_V2
    z = np.load(src)
    print('  cache <- %s' % src)
    return {k: z[k] for k in z.files}


ROW2_LEG = {'alpha': 'upper left', 'eta': 'lower left', 'K': 'upper left'}

E_YWIN = (990.0, 1010.0)


def _row2_ylim(mech, lo, hi, pad_frac=0.06):
    if mech == 'K':
        return E_YWIN
    vals = np.concatenate([lo[np.isfinite(lo)], hi[np.isfinite(hi)]])
    lo_m, hi_m = vals.min(), vals.max()
    span = max(hi_m - lo_m, 1e-9)
    return lo_m - pad_frac * span, hi_m + pad_frac * span


def _row2_title(mech):
    return 'Asymmetry strength (effect {})'.format(EFFECT_SCALE[mech])


def _row3_title(mech):
    """Titles of panels F/G/H"""
    return 'Robustness ({}=1.5, effect {})'.format(
        MECH_LABEL[mech], EFFECT_SCALE[mech])


def main(plot_only=False):
    do_compute = ('--recompute' in sys.argv)
    data = compute() if do_compute else load_cache()

    fig = plt.figure(figsize=(17, 14))
    gs = fig.add_gridspec(3, 3, hspace=0.44, wspace=0.24)
    gs_top = gs[0, :].subgridspec(1, 2, width_ratios=[1.15, 1])

    ax_h = fig.add_subplot(gs_top[0, 0])
    ax_h.plot(data['t'], data['S'], color='#111111', lw=3, label='S')
    ax_h.plot(data['t'], data['H'], color='#a65628', lw=3.5,
              label='H (HSP70A)')
    ax_h.axvline(3, color='gray', ls='--', lw=1.2)
    ax_h.annotate('symmetry-breaking\nperturbation at $t=3$ h',
                  xy=(3.0, 4.30), xytext=(3.9, 4.95), fontsize=9.5,
                  color='#555555', ha='left', va='bottom',
                  arrowprops=dict(arrowstyle='->', color='#555555', lw=0.9))

    ax_h.annotate('H peak/plateau\n$\\approx$ {:.2f}'.format(data['h_ratio']),
                  xy=(5.55, 2.82), xytext=(3.9, 3.62), fontsize=10,
                  color='#a65628', ha='left', va='bottom',
                  arrowprops=dict(arrowstyle='->', color='#a65628', lw=0.9))
    ax_h.set_xlabel('Time (h)', fontsize=13)
    ax_h.set_ylabel('Expression level (S, H)', fontsize=13)
    ax_h.set_xlim(0, 40)
    ax_h.set_ylim(0, 6)
    ax_h.legend(loc='center right', fontsize=11, frameon=False)

    ax_h.set_title('Full dynamics with H ratio $\\approx$ {:.0f}'.format(
        data['h_ratio']), fontsize=12)

    ax_bar = fig.add_subplot(gs_top[0, 1])
    labels = ['R ratio\n(symmetric)', 'L ratio\n(symmetric)',
              'L ratio\n($\\alpha_R/\\alpha_L=1.5$)',
              'L ratio\n($\\eta_R/\\eta_L=1.5$)',
              'L ratio\n($K_L/K_R=1.5$)']
    meds = [data['sym_R'], data['sym_L']]
    los = list(meds)
    his = list(meds)
    for mech in MECH:
        m, lo, hi = summary(data['dist_' + mech])
        meds.append(m)
        los.append(lo)
        his.append(hi)
    colors = ['#377eb8', '#35978f', '#984ea3', '#ff7f00', '#999999']
    yerr = [[med - lo for med, lo in zip(meds, los)],
            [hi - med for med, hi in zip(meds, his)]]
    bars = ax_bar.bar(range(5), meds, yerr=yerr, capsize=6,
                      color=colors, alpha=0.9, edgecolor='black',
                      linewidth=0.8)
    ax_bar.axhline(1000, color='gray', ls='--', lw=1.2)
    ax_bar.set_xticks(range(5))
    ax_bar.set_xticklabels(labels, fontsize=9)
    ax_bar.set_ylabel(r'$L_{\rm high}/L_{\rm low}$', fontsize=13)
    ax_bar.set_ylim(0, 1400)
    for bar, med, hi in zip(bars, meds, his):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, hi + 40,
                    '{:.0f}'.format(med), ha='center', fontsize=9)
    ax_bar.set_title('Predicted lag expression ratio', fontsize=12)

    row2_axes = []
    for j, mech in enumerate(MECH):
        ax = fig.add_subplot(gs[1, j])
        row2_axes.append(ax)
        s = data['cur_' + mech + '_s']
        m = data['cur_' + mech + '_m']
        lo = data['cur_' + mech + '_lo']
        hi = data['cur_' + mech + '_hi']
        ax.fill_between(s, lo, hi, color='#984ea3', alpha=0.25,
                        label='5%-95%')
        if mech == 'K':
            ax.plot(s, lo, color='#984ea3', lw=1.0, alpha=0.8)
            ax.plot(s, hi, color='#984ea3', lw=1.0, alpha=0.8)
        ax.plot(s, m, color='#984ea3', lw=3, label='median')
        ax.axhline(1000, color='#377eb8', ls='--', lw=2)
        ax.set_xlabel(MECH_LABEL[mech], fontsize=12)
        ax.set_ylabel(r'$L_{\rm high}/L_{\rm low}$', fontsize=12)
        y0, y1 = _row2_ylim(mech, lo, hi)
        ax.set_ylim(y0, y1)
        if mech == 'K':
            ax.set_yticks([990, 995, 1000, 1005, 1010])
        ax.ticklabel_format(axis='y', style='plain', useOffset=False)
        ax.legend(fontsize=9, loc=ROW2_LEG[mech])
        ax.set_title(_row2_title(mech), fontsize=11.5)

    row3_axes = []
    for j, mech in enumerate(MECH):
        ax = fig.add_subplot(gs[2, j])
        row3_axes.append(ax)
        vals = data['dist_' + mech]
        data_lin = 10**vals
        m, lo, hi = summary(vals)
        ax.hist(data_lin, bins=60, density=True, color='#377eb8',
                alpha=0.7, edgecolor='black', linewidth=0.5)
        ax.axvline(m, color='#d62728', lw=2, label='median {:.0f}'.format(m))
        ax.axvspan(lo, hi, color='gray', alpha=0.2, label='5%-95%')
        ax.axvline(1000, color='#984ea3', ls='--', lw=2, label='1000')
        q = np.percentile(data_lin, [0.3, 99.7])
        span = q[1] - q[0]
        pad = 0.06 * span
        xlo, xhi = q[0] - pad, q[1] + pad
        if 1000.0 > xhi:
            xhi = 1000.0 + 0.035 * (xhi - xlo)
        ax.set_xlim(xlo, xhi)
        if span < 3.0:
            ax.ticklabel_format(axis='x', style='plain', useOffset=False)
        ax.set_xlabel(r'$L_{\rm high}/L_{\rm low}$', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.legend(fontsize=9, loc='best')
        ax.set_title(_row3_title(mech), fontsize=11)

    fig.subplots_adjust(left=0.07, right=0.98, top=0.95, bottom=0.06)

    labels_letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    axes_list = [ax_h, ax_bar] + row2_axes + row3_axes
    for ax, label in zip(axes_list, labels_letters):
        bb = ax.get_position()
        fig.text(bb.x0 - 0.035, bb.y1 + 0.010, label,
                 ha='left', va='bottom', fontsize=18, fontweight='bold')

    png = os.path.join(HERE, 'fig5-v4.png')
    pdf = os.path.join(HERE, 'fig5-v4.pdf')
    fig.savefig(png, dpi=200)
    fig.savefig(pdf)
    plt.close(fig)
    print('Saved', png)
    print('Saved', pdf)


if __name__ == '__main__':
    main(plot_only=('--plot-only' in sys.argv))
