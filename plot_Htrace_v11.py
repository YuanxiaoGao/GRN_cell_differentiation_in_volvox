import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import model_v7 as M

OUTDIR = os.path.dirname(os.path.abspath(__file__))
T_MAX = 36.0


def traj(p, T_MAX=T_MAX):
    def rhs(t, y):
        H, S, L, R = y
        L = max(L, 0.0)
        R = max(R, 0.0)
        n, pe, q = p['n'], p['p'], p['q']
        FL = S**n / (p['K_SL']**n + S**n)
        FR = S**n / (p['K_SR']**n + S**n)
        FH = S**n / (p['K_SH']**n + S**n)
        G = S**p['m'] / (p['Kc']**p['m'] + S**p['m'])
        dH = p['aH'] * FH - M.GAMMA * H - p['delta'] * G * H
        dS = (p['b0'] + p['bH'] * H**n / (p['Khs']**n + H**n)
              - p['gS'] * S)
        rR = p['K_RL']**pe / (p['K_RL']**pe + R**pe)
        rL = p['K_LR']**pe / (p['K_LR']**pe + L**pe)
        hL = L**q / (p['K_L']**q + L**q)
        hR = R**q / (p['K_R']**q + R**q)
        dL = p['beta'] + FL * (p['aL'] + p['eta'] * hL) * rR - M.GAMMA * L
        dR = p['beta'] + FR * (p['aL'] + p['eta'] * hR) * rL - M.GAMMA * R
        return [dH, dS, dL, dR]

    from scipy.integrate import solve_ivp
    t_pre = np.arange(0, 3.000001, 0.25)
    t_post = np.arange(3.0, T_MAX + 1e-6, 0.25)
    y0 = [0.01, 0.05, 0.05, 0.05]
    pre = solve_ivp(rhs, [0, 3.0], y0, t_eval=t_pre, method='LSODA',
                    max_step=0.1, rtol=1e-9, atol=1e-11)
    y3 = pre.y[:, -1].copy()
    y3[2] += 0.02
    y3[3] -= 0.02
    post = solve_ivp(rhs, [3.0, T_MAX], y3, t_eval=t_post, method='LSODA',
                     max_step=0.1, rtol=1e-9, atol=1e-11)
    t = np.hstack([pre.t, post.t[1:]])
    Y = np.hstack([pre.y, post.y[:, 1:]])
    return t, Y


def main():
    p = dict(M.BASE)
    t, Y = traj(p)
    H, S = Y[0], Y[1]
    i_pk = int(np.argmax(H))
    ratio = H.max() / H[-1]

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
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
    fig.tight_layout()

    for ext in ('.png', '.pdf'):
        fig.savefig(os.path.join(OUTDIR, 'fig4-v11_Htrace' + ext), dpi=200,
                    bbox_inches='tight')
    plt.close(fig)

    t2, Y2 = traj(p, T_MAX=120.0)
    print('full-dynamics H trace at the BASE reference parameters')
    print('  peak      H = %.6f  (t = %.2f h)' % (H[i_pk], t[i_pk]))
    print('  plateau   H(36 h) = %.6f' % H[-1])
    print('  ratio (36 h) = %.6f' % ratio)
    print('  cross-check: H(120 h) = %.6f, ratio(120 h) = %.6f, relative difference %.3f%%'
          % (Y2[0][-1], Y2[0].max() / Y2[0][-1],
             100 * abs(ratio - Y2[0].max() / Y2[0][-1]) / (Y2[0].max() / Y2[0][-1])))
    print('saved fig4-v11_Htrace.png/.pdf')


if __name__ == '__main__':
    main()
