# fig31_3x3.py

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from scipy.integrate import solve_ivp
from scipy.optimize import root, brentq

OUT = os.path.dirname(os.path.abspath(__file__))
CMAP_BR = LinearSegmentedColormap.from_list(
    'twilight_shifted_br',
    plt.get_cmap('twilight_shifted')(np.linspace(0.13, 0.72, 256)))

P = {
    'n': 3, 'm': 3, 'q': 2, 'p': 3,
    'aH': 3.5, 'Ksh': 0.5, 'gH': 0.35, 'd': 7.0, 'Kc': 1.5,
    'b0': 0.055, 'bH': 0.9, 'Khs': 0.5, 'gS': 0.1,
    'aL': 3.5, 'aR': 3.5, 'Ksl': 0.5, 'Ksr': 0.5,
    'Krl': 0.5, 'Klr': 0.5, 'pm': 3, 'gL': 0.35, 'gR': 0.35,
    'bL': 0.0035, 'bR': 0.0035, 'etaL': 0.7, 'etaR': 0.7,
    'KL': 0.5, 'KR': 0.5, 'nH': 3,
}


def rhs_vec(L, R, S, p):
    n = p['n']
    AL = S**n / (p['Ksl']**n + S**n)
    AR = S**n / (p['Ksr']**n + S**n)
    rR = p['Krl']**p['pm'] / (p['Krl']**p['pm'] + R**p['pm'])
    rL = p['Klr']**p['pm'] / (p['Klr']**p['pm'] + L**p['pm'])
    dL = (p['bL'] + (p['aL']*AL + p['etaL']*AL*(L**p['q']/(p['KL']**p['q'] + L**p['q'])))*rR
          - p['gL']*L)
    dR = (p['bR'] + (p['aR']*AR + p['etaR']*AR*(R**p['q']/(p['KR']**p['q'] + R**p['q'])))*rL
          - p['gR']*R)
    return dL, dR


def jac(L, R, S, p):
    n, q, KL, KR = p['n'], p['q'], p['KL'], p['KR']
    AL = S**n / (p['Ksl']**n + S**n)
    AR = S**n / (p['Ksr']**n + S**n)
    rR = p['Krl']**p['pm'] / (p['Krl']**p['pm'] + R**p['pm'])
    rL = p['Klr']**p['pm'] / (p['Klr']**p['pm'] + L**p['pm'])
    selfL = p['etaL']*AL*q*KL**q*L**(q-1) / (KL**q + L**q)**2
    selfR = p['etaR']*AR*q*KR**q*R**(q-1) / (KR**q + R**q)**2
    prodL = p['aL']*AL + p['etaL']*AL*(L**q/(KL**q + L**q))
    prodR = p['aR']*AR + p['etaR']*AR*(R**q/(KR**q + R**q))
    return np.array([
        [selfL*rR - p['gL'],
         -prodL*p['pm']*p['Krl']**p['pm']*R**(p['pm']-1)/(p['Krl']**p['pm'] + R**p['pm'])**2],
        [-prodR*p['pm']*p['Klr']**p['pm']*L**(p['pm']-1)/(p['Klr']**p['pm'] + L**p['pm'])**2,
         selfR*rL - p['gR']],
    ])


def find_fixed_points(S, p, starts):
    sols = []
    for st in starts:
        r = root(lambda xy: np.array(rhs_vec(xy[0], xy[1], S, p)), st,
                 jac=lambda xy: jac(xy[0], xy[1], S, p),
                 method='hybr', options={'xtol': 1e-12, 'maxfev': 5000})
        if not r.success:
            continue
        x = r.x
        if x[0] < 0 or x[1] < 0:
            continue
        if not any(np.max(np.abs(x - s)) < 1e-4 for s in sols):
            sols.append(x)
    stable, saddles = [], []
    for x in sols:
        ev = np.linalg.eigvals(jac(x[0], x[1], S, p))
        if np.all(np.real(ev) < 0):
            stable.append(x)
        elif np.prod(np.real(ev)) < 0:
            saddles.append(x)
    return stable, saddles


def Hstar(S, p):
    aH, Ksh, nH, gH, d, Kc, m = (p.get(k, v) for k, v in [
        ('aH', 0.15), ('Ksh', 0.5), ('nH', 3), ('gH', 0.015),
        ('d', 0.12), ('Kc', 1.5), ('m', 3)])
    return aH*S**nH/(Ksh**nH + S**nH) / (gH + d*S**m/(Kc**m + S**m))


def coarse_starts():
    return [[L0, R0] for L0 in np.logspace(-2, 1.6, 14)
            for R0 in np.logspace(-2, 1.6, 14)]


def dense_starts():
    return [[L0, R0] for L0 in np.logspace(-4, 1.7, 40)
            for R0 in np.logspace(-4, 1.7, 40)]


def full_rhs(t, y, p):
    H, S, L, R = y
    F = S**p['n'] / (p['Ksh']**p['n'] + S**p['n'])
    G = S**p['m'] / (p['Kc']**p['m'] + S**p['m'])
    dH = p['aH'] * F - p['gH'] * H - p['d'] * G * H
    dS = (p['b0'] + p['bH'] * H**p['n'] / (p['Khs']**p['n'] + H**p['n'])
          - p['gS'] * S)
    rR = p['Krl']**p['p'] / (p['Krl']**p['p'] + R**p['p'])
    rL = p['Klr']**p['p'] / (p['Klr']**p['p'] + L**p['p'])
    dL = (p['bL'] + F * (p['aL'] + p['etaL'] * L**p['q'] /
                         (p['KL']**p['q'] + L**p['q'])) * rR
          - p['gL'] * L)
    dR = (p['bR'] + F * (p['aR'] + p['etaR'] * R**p['q'] /
                         (p['KR']**p['q'] + R**p['q'])) * rL
          - p['gR'] * R)
    return [dH, dS, dL, dR]


def lr_rhs(xy, S):
    L, R = xy
    F = S**P['n'] / (P['Ksl']**P['n'] + S**P['n'])
    rR = P['Krl']**P['p'] / (P['Krl']**P['p'] + R**P['p'])
    rL = P['Klr']**P['p'] / (P['Klr']**P['p'] + L**P['p'])
    dL = (P['bL'] + F * (P['aL'] + P['etaL'] * L**P['q'] /
                         (P['KL']**P['q'] + L**P['q'])) * rR
          - P['gL'] * L)
    dR = (P['bR'] + F * (P['aR'] + P['etaR'] * R**P['q'] /
                         (P['KR']**P['q'] + R**P['q'])) * rL
          - P['gR'] * R)
    return [dL, dR]


def lr_jac(xy, S):
    L, R = xy
    F = S**P['n'] / (P['Ksl']**P['n'] + S**P['n'])
    rR = P['Krl']**P['p'] / (P['Krl']**P['p'] + R**P['p'])
    rL = P['Klr']**P['p'] / (P['Klr']**P['p'] + L**P['p'])
    selfL = (P['etaL'] * F * P['q'] * P['KL']**P['q'] * L**(P['q'] - 1) /
             (P['KL']**P['q'] + L**P['q'])**2)
    selfR = (P['etaR'] * F * P['q'] * P['KR']**P['q'] * R**(P['q'] - 1) /
             (P['KR']**P['q'] + R**P['q'])**2)
    prodL = P['aL'] * F + P['etaL'] * F * (L**P['q'] /
                                           (P['KL']**P['q'] + L**P['q']))
    prodR = P['aR'] * F + P['etaR'] * F * (R**P['q'] /
                                           (P['KR']**P['q'] + R**P['q']))
    return np.array([
        [selfL * rR - P['gL'],
         -prodL * P['p'] * P['Krl']**P['p'] * R**(P['p'] - 1) /
         (P['Krl']**P['p'] + R**P['p'])**2],
        [-prodR * P['p'] * P['Klr']**P['p'] * L**(P['p'] - 1) /
         (P['Klr']**P['p'] + L**P['p'])**2,
         selfR * rL - P['gR']],
    ])


def stable_states(S):
    sols = []
    for L0 in np.logspace(-2, 1.6, 14):
        for R0 in np.logspace(-2, 1.6, 14):
            r = root(lr_rhs, [L0, R0], args=(S,), method='hybr',
                     jac=lambda xy, S=S: lr_jac(xy, S),
                     options={'xtol': 1e-12, 'maxfev': 3000})
            if not r.success:
                continue
            x = r.x
            if x[0] < 0 or x[1] < 0:
                continue
            if any(np.max(np.abs(x - s)) < 1e-4 for s in sols):
                continue
            sols.append(x)
    st = []
    for x in sols:
        ev = np.linalg.eigvals(lr_jac(x, S))
        if np.all(np.real(ev) < 0):
            st.append(x)
    return st


def full_equilibrium_S(p):
    """Return S* of the full 4-variable system (dH=dS=dL=dR=0)."""
    n = p['n']

    def f(S):
        Hs = Hstar(S, p)
        return ((p['b0'] + p['bH'] * Hs**n / (p['Khs']**n + Hs**n))
                / p['gS'] - S)

    xs = np.linspace(0.001, 30.0, 3000)
    vals = f(xs)
    roots = []
    for i in range(len(xs) - 1):
        if vals[i] * vals[i + 1] < 0:
            roots.append(brentq(f, xs[i], xs[i + 1]))
    return roots


T_MAX = 36.0
T_ASYM = 3.0
DELTA = 0.02
t_pre = np.arange(0, T_ASYM + 1e-9, 0.25)
t_post = np.arange(T_ASYM, T_MAX + 1e-9, 0.25)


def run_trajectory(sign):
    y0 = [0.01, 0.05, 0.05, 0.05]
    pre = solve_ivp(lambda tt, y: full_rhs(tt, y, P), [0, T_ASYM], y0,
                    t_eval=t_pre, method='LSODA', max_step=0.1,
                    rtol=1e-9, atol=1e-11)
    y3 = pre.y[:, -1].copy()
    y3[2] += DELTA * sign
    y3[3] -= DELTA * sign
    post = solve_ivp(lambda tt, y: full_rhs(tt, y, P), [T_ASYM, T_MAX], y3,
                     t_eval=t_post, method='LSODA', max_step=0.1,
                     rtol=1e-9, atol=1e-11)
    t = np.concatenate([pre.t, post.t[1:]])
    y = np.hstack([pre.y, post.y[:, 1:]])
    return t, y


def draw_lr(ax, tt, L1, R1, ls, lw, alpha, title, show_x=True):
    ax.plot(tt, L1, color='steelblue', lw=lw, ls=ls, alpha=alpha,
            label='L (lag)')
    ax.plot(tt, R1, color='seagreen', lw=lw, ls=ls, alpha=alpha,
            label='R (regA)')
    if show_x:
        ax.set_xlabel('Time (h)', fontsize=11)
    else:
        ax.set_xlabel('')
        ax.tick_params(axis='x', labelbottom=False)
    ax.set_ylabel('Expression level (log)', fontsize=11)
    ax.set_xlim(0, T_MAX)
    ax.set_ylim(5e-3, 50)
    ax.set_yscale('log')
    ax.axvline(T_ASYM, color='gray', ls='--', lw=1.2)
    ax.legend(loc='center right', fontsize=9, frameon=False)
    ax.text(0.5, 0.95, title, transform=ax.transAxes, fontsize=9,
            ha='center', va='top', fontweight='bold')


print('Computing clamped-S branches...')
Sgrid = np.concatenate([np.linspace(0.001, 0.2, 40),
                        np.linspace(0.21, 1.5, 60),
                        np.linspace(1.6, 10, 30)])
rows = []
for S in Sgrid:
    st, _ = find_fixed_points(S, P, coarse_starts())
    rows.append({'S': S, 'L': [float(x[0]) for x in st],
                 'R': [float(x[1]) for x in st]})

HV = np.array([Hstar(S, P) for S in Sgrid])
n2 = [r['S'] for r in rows if len(r['L']) == 2]
S_th = n2[0] if n2 else 0.19
print('H* peak/steady ratio (S=4.8): {:.3f}'.format(
    float(HV.max() / Hstar(4.8, P))))

print('Running full dynamics...')
t, y_g = run_trajectory(+1)
_, y_s = run_trajectory(-1)
H_g, S_g, L_g, R_g = y_g
H_s, S_s, L_s, R_s = y_s

Seq_roots = full_equilibrium_S(P)
Seq = Seq_roots[0] if Seq_roots else S_g[-1]
Heq = Hstar(Seq, P)
eq_stable, _ = find_fixed_points(Seq, P, dense_starts())
eq_stable = sorted(eq_stable, key=lambda x: x[0])  # L-dominant last

fig, axs = plt.subplots(
    3, 3, figsize=(16.5, 14.0),
    gridspec_kw={'left': 0.07, 'right': 0.91, 'top': 0.97,
                 'bottom': 0.06, 'wspace': 0.18, 'hspace': 0.20})

# Row 1: H-L, H-R, L-R phase plane (S-coloured)
S_L = np.array([r['S'] for r in rows for _ in r['L']])
Lvals = np.array([Lv for r in rows for Lv in r['L']])
S_R = np.array([r['S'] for r in rows for _ in r['R']])
Rvals = np.array([Rv for r in rows for Rv in r['R']])

ax = axs[0, 0]
ax.plot(Sgrid, np.maximum(HV, 1e-6), color='#a65628', lw=3.5,
        label='H (HSP70A)')
ax.plot(S_L, Lvals, marker='o', mfc='steelblue', mec='steelblue', ms=5,
        ls='', alpha=0.8, label='L (lag) stable states')
ax.axvline(S_th, color='k', ls='--', alpha=0.5, lw=1,
           label='first bistable S={:.2f}'.format(S_th))
ax.set_xscale('log')
ax.set_xlabel('Cell size signal S', fontsize=14)
ax.set_ylabel('Expression level', fontsize=14)
ax.set_title('H and L', fontsize=14)
ax.set_ylim(-0.5, 13.5)
ax.tick_params(axis='both', labelsize=13)
ax.legend(loc='upper left', fontsize=11)

ax = axs[0, 1]
ax.plot(Sgrid, np.maximum(HV, 1e-6), color='#a65628', lw=3.5,
        label='H (HSP70A)')
ax.plot(S_R, Rvals, marker='o', mfc='seagreen', mec='seagreen', ms=5,
        ls='', alpha=0.8, label='R (regA) stable states')
ax.axvline(S_th, color='k', ls='--', alpha=0.5, lw=1,
           label='first bistable S={:.2f}'.format(S_th))
ax.set_xscale('log')
ax.set_xlabel('Cell size signal S', fontsize=14)
ax.set_ylabel('Expression level', fontsize=14)
ax.set_title('H and R', fontsize=14)
ax.set_ylim(-0.5, 13.5)
ax.tick_params(axis='both', labelsize=13)
ax.legend(loc='upper left', fontsize=11)

ax = axs[0, 2]
s_norm = matplotlib.colors.LogNorm(vmin=1e-3, vmax=10)
phase_x, phase_y, phase_s = [], [], []
for r in rows:
    for Lv, Rv in zip(r['L'], r['R']):
        phase_x.append(Lv)
        phase_y.append(Rv)
        phase_s.append(r['S'])
sc_phase = ax.scatter(phase_x, phase_y, c=phase_s, cmap='cividis_r',
                      norm=s_norm, s=20, alpha=0.8)
ax.set_xlabel('L (lag)', fontsize=14)
ax.set_ylabel('R (regA)', fontsize=14)
ax.set_title('L-R phase plane', fontsize=14)
phase_ticks = [0, 2, 4, 6, 8, 10, 12]
ax.set_xlim(-0.5, 13)
ax.set_ylim(-0.5, 13)
ax.set_xticks(phase_ticks)
ax.set_yticks(phase_ticks)
ax.tick_params(axis='both', labelsize=13)

# Row 2: log-log L-R phase portraits at S = 0.1, 0.3, 5
log_Svals = [0.1, 0.3, 5.0]
qvobj = None
for col, S in enumerate(log_Svals):
    ax = axs[1, col]
    stable, saddles = find_fixed_points(S, P, dense_starts())

    u = np.linspace(-3.0, 2.0, 180)
    Ug, Vg = np.meshgrid(u, u)
    Lg, Rg = 10**Ug, 10**Vg
    dL, dR = rhs_vec(Lg, Rg, S, P)
    ax.contour(Ug, Vg, dL, [0.0], colors='steelblue', linewidths=3.0)
    ax.contour(Ug, Vg, dR, [0.0], colors='seagreen', linewidths=3.0)

    qv = np.linspace(-2.7, 1.75, 14)
    QU, QV = np.meshgrid(qv, qv)
    QL, QR = 10**QU, 10**QV
    dQL, dQR = rhs_vec(QL, QR, S, P)
    dU = dQL / (QL * np.log(10.0))
    dV = dQR / (QR * np.log(10.0))
    speed = np.hypot(dU, dV)
    mask = speed > 1e-12
    uu = np.where(mask, dU/np.maximum(speed, 1e-12), 0.0)
    vv = np.where(mask, dV/np.maximum(speed, 1e-12), 0.0)
    qvobj = ax.quiver(QU, QV, uu, vv, speed, cmap=CMAP_BR,
                      norm=matplotlib.colors.LogNorm(vmin=1e-3, vmax=1e1),
                      width=0.003, headwidth=4.5, headlength=6,
                      headaxislength=5.5, alpha=1.0, zorder=2)

    is_bistable = len(stable) > 1
    diag_label = 'separatrix $L=R$' if is_bistable else 'symmetry axis $L=R$'
    ax.plot([u[0], u[-1]], [u[0], u[-1]], 'k--', lw=1.4, zorder=3,
            label=diag_label)
    for x in stable:
        ax.plot(np.log10(x[0]), np.log10(x[1]), 'o', ms=11, mfc='black',
                mec='black', zorder=6)
    for x in saddles:
        ax.plot(np.log10(x[0]), np.log10(x[1]), 'o', ms=12, mfc='none',
                mec='#d62728', mew=2.2, zorder=6)

    ax.set_xlim(u[0], u[-1])
    ax.set_ylim(u[0], u[-1])
    ax.set_xlabel(r'$\log_{10} L$ (lag)', fontsize=12)
    ax.set_ylabel(r'$\log_{10} R$ (regA)', fontsize=12)
    ax.set_title('Log-log L-R phase plane at S = {:g}'.format(S),
                 fontsize=12)
    if is_bistable:
        ax.text(0.98, 0.05, 'L-dominant basin', transform=ax.transAxes,
                fontsize=11, color='steelblue', fontweight='bold',
                ha='right', va='bottom')
        ax.text(0.02, 0.95, 'R-dominant basin', transform=ax.transAxes,
                fontsize=11, color='seagreen', fontweight='bold',
                ha='left', va='top')
    else:
        ax.text(0.50, 0.72, 'single stable state\n$L\\approx R$',
                transform=ax.transAxes, fontsize=11, color='black',
                fontweight='normal', ha='center', va='top')
    ax.legend(loc='upper right', fontsize=8, framealpha=0.9)

# Row 3, left: S and H time course
ax = axs[2, 0]
ax.plot(t, S_g, color='#111111', lw=3, label='S')
ax.plot(t, H_g, color='#a65628', lw=3.5, ls='-', label='H (HSP70A)')
ax.set_xlabel('Time (h)', fontsize=13)
ax.set_ylabel('Expression level (S, H)', fontsize=13)
ax.set_xlim(0, T_MAX)
ax.set_ylim(0, 6)
ax.axvline(T_ASYM, color='gray', ls='--', lw=1.2)
ax.text(T_ASYM + 0.5, 3.9, 'asymmetric division', color='gray',
        fontsize=10)
h_peak = H_g.max()
t_peak = t[np.argmax(H_g)]
ax.annotate('H peak ~{:.2f}'.format(h_peak), xy=(t_peak, h_peak),
            xytext=(t_peak + 3.0, h_peak + 0.8), fontsize=10,
            color='#a65628')
ax.annotate('H plateau ~{:.2f}'.format(H_g[-1]),
            xy=(T_MAX * 0.75, H_g[-1]),
            xytext=(T_MAX * 0.50, H_g[-1] + 0.45), fontsize=10,
            color='#a65628')
ax.axhline(Seq, color='gray', ls=':', lw=1.5)
ax.text(T_MAX * 0.55, Seq + 0.12, 'S equilibrium', color='gray', fontsize=9)
ax.legend(loc='upper right', fontsize=11, frameon=False)

# Row 3, right: dynamic endpoints on clamped L-R branches
ax = axs[2, 2]
bS = np.array([r['S'] for r in rows for _ in r['L']])
bL = np.array([np.log10(max(Lv, 1e-6)) for r in rows for Lv in r['L']])
bR = np.array([np.log10(max(Rv, 1e-6)) for r in rows for Rv in r['R']])
sc_end = ax.scatter(bL, bR, c=bS, cmap='cividis_r',
                    norm=matplotlib.colors.LogNorm(vmin=1e-3, vmax=10),
                    s=22, alpha=0.9, zorder=3)
ax.plot(np.log10(max(L_g[-1], 1e-6)), np.log10(max(R_g[-1], 1e-6)),
        marker='o', color='steelblue', mec='k', ms=16, ls='none',
        label='36 h endpoint (L-biased)')
ax.plot(np.log10(max(L_s[-1], 1e-6)), np.log10(max(R_s[-1], 1e-6)),
        marker='o', color='seagreen', mec='k', ms=16, ls='none',
        label='36 h endpoint (R-biased)')
for x in eq_stable:
    col = 'steelblue' if x[0] > x[1] else 'red'
    ax.plot(np.log10(max(x[0], 1e-6)), np.log10(max(x[1], 1e-6)),
            marker='*', color=col, mec='k', ms=24, ls='none', zorder=7)
ax.set_xlabel(r'$\log_{10} L$ (lag)', fontsize=12)
ax.set_ylabel(r'$\log_{10} R$ (regA)', fontsize=12)
ax.set_xlim(-3, 2)
ax.set_ylim(-3, 2)
markers = [Line2D([0], [0], marker='o', ls='none', color='steelblue',
                  markerfacecolor='steelblue', markeredgecolor='k',
                  label='36 h endpoint (L-biased)'),
           Line2D([0], [0], marker='o', ls='none', color='seagreen',
                  markerfacecolor='seagreen', markeredgecolor='k',
                  label='36 h endpoint (R-biased)'),
           Line2D([0], [0], marker='*', ls='none', color='steelblue',
                  markerfacecolor='steelblue', markeredgecolor='k', ms=18,
                  label='final equilibrium (L-dominant)'),
           Line2D([0], [0], marker='*', ls='none', color='red',
                  markerfacecolor='red', markeredgecolor='k', ms=18,
                  label='final equilibrium (R-dominant)')]
ax.legend(handles=markers, loc='upper right', fontsize=9, frameon=False)

# Split the middle-bottom cell into solid and dashed L/R panels
fig.canvas.draw()
pos = axs[2, 1].get_position()
axs[2, 1].remove()
gap = pos.height * 0.03
h = (pos.height - gap) / 2.0
ax_top = fig.add_axes([pos.x0, pos.y0 + h + gap, pos.width, h])
ax_bot = fig.add_axes([pos.x0, pos.y0, pos.width, h])
draw_lr(ax_top, t, L_g, R_g, '-', 2.5, 1.0,
        'solid: L-biased perturbation', show_x=False)
draw_lr(ax_bot, t, L_s, R_s, '--', 2.0, 0.85,
        'dashed: R-biased perturbation')

# Color bars for the three right-column panels
fig.canvas.draw()
p1 = axs[0, 2].get_position()
p2 = axs[1, 2].get_position()
p3 = axs[2, 2].get_position()
cax_s = fig.add_axes([p1.x1 + 0.012, p1.y0, 0.008, p1.height])
cax_sp = fig.add_axes([p2.x1 + 0.012, p2.y0, 0.008, p2.height])
cax_end = fig.add_axes([p3.x1 + 0.012, p3.y0, 0.008, p3.height])
cbar_s = plt.colorbar(sc_phase, cax=cax_s)
cbar_s.set_label('S', fontsize=12)
cbar_s.ax.tick_params(labelsize=11)
cbar_sp = plt.colorbar(qvobj, cax=cax_sp)
cbar_sp.set_label(r'$\left|\,(\mathrm{d}\log_{10}L/\mathrm{d}t,\ '
                  r'\mathrm{d}\log_{10}R/\mathrm{d}t)\,\right|$',
                  fontsize=10)
cbar_sp.ax.tick_params(labelsize=10)
cbar_end = plt.colorbar(sc_end, cax=cax_end)
cbar_end.set_label('S', fontsize=12)
cbar_end.ax.tick_params(labelsize=11)

fig.canvas.draw()
renderer = fig.canvas.get_renderer()
for ax, label in [(axs[0, 0], 'A'), (axs[1, 0], 'B'), (axs[2, 0], 'C')]:
    title_bb = ax.title.get_window_extent(renderer)
    ylabel_bb = ax.yaxis.label.get_window_extent(renderer)
    inv = ax.transAxes.inverted()
    x_left = inv.transform((ylabel_bb.x0, 0))[0]
    y_base = inv.transform((0, title_bb.y0))[1]
    ax.text(x_left, y_base, label, transform=ax.transAxes, ha='left',
            va='bottom', fontsize=18, fontweight='bold')

fig_name = os.path.join(OUT, 'fig31_3x3.png')
fig.savefig(fig_name, dpi=200)
fig.savefig(os.path.join(OUT, 'fig31_3x3.pdf'), dpi=200)
plt.close(fig)
print('Saved', fig_name)
