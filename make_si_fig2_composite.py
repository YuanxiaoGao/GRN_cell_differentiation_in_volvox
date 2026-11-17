import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
F4DIR = os.path.abspath(os.path.join(HERE, '..', 'fig4'))
sys.path.insert(0, F4DIR)

import fig4_final as F
A_REF, B_REF = 3.0, 0.6
STATS_CSV = os.path.join(HERE, 'sref_window_stats.csv')
ST_CSV = os.path.join(HERE, 'sref_scan', 'sref_ab_ST.csv')
CACHE = os.path.join(HERE, 'sref_scan', '_cache')
SEED = 20241011
N_BAR = 512

BAR_CONFIGS = [
    (r'$a=2.0,\,b=0.50$', 'a20_b050'),
    (r'$a=2.5,\,b=0.55$', 'a25_b055'),
    (r'$a=3.0,\,b=0.60$', 'a30_b060'),      # published rule -> brown
    (r'$a=4.0,\,b=0.70$', 'a40_b070'),
    (r'$a=5.0,\,b=0.80$', 'a50_b080'),
    (r'$a=8.0,\,b=0.90$', 'a80_b090'),
]
REF_IDX = 2

GREEK = {'aL': r'$\alpha_L$', 'eta': r'$\eta$', 'beta': r'$\beta$',
         'K': r'$K$', 'p': r'$p$', 'n': r'$n$', 'q': r'$q$'}

_memo = {}

TS = 8.4 
LS = 12.0 


def load_stats():
    d = np.genfromtxt(STATS_CSV, delimiter=',', names=True)
    arr = np.column_stack([d['S_th'], d['S_hi'], d['S_ref']])
    w = F.bistable_window(dict(F.BASE))
    v = min(A_REF * w[0], B_REF * w[1])
    S0 = float(np.clip(v, w[0] * 1.02, w[1] * 0.98))
    return arr, S0


ST_CACHE = os.path.join(HERE, '_bar_st_cache.npz')


def bar_st(tag):
    if os.path.exists(ST_CACHE):
        z = np.load(ST_CACHE)
        if tag + '_L' in z.files and tag + '_R' in z.files:
            return [z[tag + '_L'], z[tag + '_R']]
    Y = np.load(os.path.join(CACHE, '_srefab_%s_n%d.npz' % (tag, N_BAR)))['Y']
    problem = F.make_problem(F.LR_NAMES)
    out = [np.array(F.sobol_analyze(problem, Y[:, c], SEED)['ST'], float)
           for c in (2, 3)]          # col 2 = log10(L_hi/R_lo), col 3 = R_hi/L_lo
    _memo[tag] = out
    return out


def plain_log_axis(ax, ticks, which='x'):
    from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
    a = ax.xaxis if which == 'x' else ax.yaxis
    a.set_major_locator(FixedLocator(ticks))
    a.set_major_formatter(FixedFormatter(['%g' % t for t in ticks]))
    a.set_minor_locator(NullLocator())


def panel_letter(ax, letter):
    ax.text(-0.155, 1.045, letter, transform=ax.transAxes,
            ha='left', va='bottom', fontsize=LS, fontweight='bold',
            family='DejaVu Sans', color='black', clip_on=False)


def panel_title(ax, text, size=None):
    """Descriptive title on its own line, centred over the axes (Fig. S1)."""
    ax.set_title(text, fontsize=TS if size is None else size, pad=3.0)


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'mathtext.fontset': 'stix',
        'font.size': 8.2,
        'axes.linewidth': 0.8,
        'xtick.direction': 'in', 'ytick.direction': 'in',
        'xtick.top': True, 'ytick.right': True,
        'legend.frameon': False,
        'savefig.bbox': 'tight', 'savefig.pad_inches': 0.03,
    })

    arr, S0 = load_stats()
    # rows are (label, tag, ST[y=L_hi/R_lo], ST[y=R_hi/L_lo]) -> data cols 2, 3
    bars = [(lab, tag) + tuple(bar_st(tag)) for lab, tag in BAR_CONFIGS]
    if _memo:                                   # first run: persist for reuse
        np.savez(ST_CACHE,
                 **{k + suf: v[i] for k, v in _memo.items()
                    for i, suf in enumerate(('_L', '_R'))})
        print('cached ST estimates -> _bar_st_cache.npz')
    asym = max(float(np.abs(b[2] - b[3]).max()) for b in bars)
    print('max |ST(L_hi/R_lo) - ST(R_hi/L_lo)| over 6 designs x 7 params: %.2e'
          % asym)

    fig, axs = plt.subplots(2, 2, figsize=(7.0, 5.55))
    (axA, axB), (axC, axD) = axs

    frac = 100 * (arr[:, 0] > S0).mean()
    print('a fixed read-out misses %.1f%% of the draws (S_ref=%.3f)'
          % (frac, S0))
    bins = np.geomspace(0.15, 1.65, 22)
    axA.hist(arr[:, 0], bins=bins, color='#9ecae1', edgecolor='#3182bd',
             linewidth=0.6)
    axA.hist(arr[:, 0][arr[:, 0] > S0], bins=bins, color='#f4a582',
             edgecolor='#c0392b', linewidth=0.6)
    axA.axvline(S0, color='#a61e22', lw=1.4, ls='--')
    axA.set_xscale('log')
    plain_log_axis(axA, [0.3, 0.5, 0.7, 1.0, 1.5])
    axA.set_xlabel(r'lower fold point  $S_{\rm th}$')
    axA.set_ylabel('samples')
    axA.set_ylim(0, 30)
    axA.annotate(r'$S_{\rm ref}=%.3f$' % S0, xy=(S0, 26.5),
                 xytext=(0.175, 28.2), fontsize=7.4, color='#a61e22',
                 va='center',
                 arrowprops=dict(arrowstyle='-', color='#a61e22', lw=0.7,
                                 shrinkA=2, shrinkB=2))

    ok = arr[:, 2] > 0
    axB.scatter(arr[ok, 0], arr[ok, 2], s=13, color='#3182bd', alpha=0.75,
                edgecolor='none', zorder=3,
                label=r'$S_{\rm ref}=\min\{3S_{\rm th},\,0.6S_{\rm hi}\}$')
    xx = np.geomspace(arr[:, 0].min() * 0.95, arr[:, 0].max() * 1.05, 50)
    axB.plot(xx, A_REF * xx, color='#333333', lw=0.9, ls='--', zorder=2,
             label=r'$3\,S_{\rm th}$')
    off = np.abs(arr[:, 2] / (A_REF * arr[:, 0]) - 1) > 1e-3
    if off.any():
        axB.scatter(arr[off, 0], arr[off, 2], s=30, facecolor='none',
                    edgecolor='#c0392b', linewidth=1.0, zorder=4,
                    label=r'$0.6\,S_{\rm hi}$ branch')
    axB.set_xscale('log')
    axB.set_yscale('log')
    plain_log_axis(axB, [0.3, 0.5, 0.7, 1.0, 1.5])
    plain_log_axis(axB, [1, 2, 3, 4], 'y')
    axB.set_xlabel(r'lower fold point  $S_{\rm th}$')
    axB.set_ylabel(r'read-out size  $S_{\rm ref}$')
    axB.legend(fontsize=7.0, loc='upper left', handletextpad=0.4,
               borderaxespad=0.3, labelspacing=0.35)

    p = dict(F.BASE)
    S_th = F.bistable_window(p)[0]
    ms = np.geomspace(1.02, 30.0, 40)
    ys = []
    for m in ms:
        st = F.stable_states(float(m * S_th), p)
        if len(st) < 2:
            ys.append(np.nan)
            continue
        iL = max(range(len(st)), key=lambda k: st[k][0] - st[k][1])
        Lh, Rl = st[iL]
        ys.append(np.log10(Lh / max(Rl, 1e-12)))
    ys = np.array(ys)
    axC.plot(ms, ys, color='#1f4e79', lw=1.5)
    axC.axhline(np.nanmax(ys), color='#888888', lw=0.8, ls=':')
    for a, c, lw in ((2.0, '#c0392b', 0.9), (3.0, '#a61e22', 1.3),
                     (5.0, '#2e8b57', 0.9), (8.0, '#8e44ad', 0.9)):
        axC.plot([a, a], [0, 3.06], color=c, lw=lw, ls='--' if a != 3 else '-')
        j = int(np.nanargmin(np.abs(ms - a)))
        if np.isfinite(ys[j]):
            axC.plot([a], [ys[j]], 'o', ms=4, color=c, zorder=5)
        axC.text(a, 3.12, '%g' % a, fontsize=7.0, color=c, ha='center',
                 va='bottom')
    axC.set_xscale('log')
    plain_log_axis(axC, [1, 1.5, 2, 3, 5, 10, 20])
    plain_log_axis(axC, [0, 1, 2, 3], 'y')
    axC.set_xlabel(r'$S_{\rm ref}/S_{\rm th}=a$', labelpad=2)
    axC.set_ylabel(r'$y=\log_{10}(L_{\rm high}/R_{\rm low})$')
    axC.set_ylim(0, 3.62)

    cmap = plt.get_cmap('Blues')
    bcols = ['#a65628' if i == REF_IDX else cmap(0.30 + 0.62 * i / (len(bars) - 1))
             for i in range(len(bars))]
    x = np.arange(len(F.LR_NAMES))
    wbar = 0.80 / len(bars)
    bar_handles = []
    for i, row in enumerate(bars):
        h = axD.bar(x + (i - (len(bars) - 1) / 2.0) * wbar, row[2], wbar,
                    color=bcols[i], edgecolor='#6e6e6e', linewidth=0.22,
                    zorder=3, label=row[0])
        bar_handles.append(h)
    axD.set_xticks(x)
    axD.set_xticklabels([GREEK[nm] for nm in F.LR_NAMES])
    axD.set_xlim(-0.62, len(F.LR_NAMES) - 0.38)
    axD.set_ylim(0, 0.60)
    axD.axhline(0, color='k', lw=0.7)
    axD.grid(axis='y', alpha=0.22, lw=0.5, zorder=0)
    axD.set_ylabel(r'total-order index  $S_T$')

    for ax, letter in zip((axA, axB, axC, axD), 'ABCD'):
        panel_letter(ax, letter)

    fig.tight_layout(h_pad=2.0, w_pad=1.3, rect=(0, 0.085, 1, 1))
    lab = [h.get_label() for h in bar_handles]
    fig.legend(bar_handles, lab, loc='lower center', ncol=6, fontsize=7.0,
               handlelength=0.9, handletextpad=0.35, columnspacing=0.85,
               bbox_to_anchor=(0.5, 0.022))

    for ext in ('.pdf', '.png'):
        fig.savefig(os.path.join(HERE, 'figS2_composite' + ext), dpi=400)
    plt.close(fig)
    print('wrote figS2_composite.pdf/.png')


if __name__ == '__main__':
    main()
