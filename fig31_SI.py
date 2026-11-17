import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import root

OUT = os.path.dirname(os.path.abspath(__file__))

P = {
    'n': 3, 'm': 3, 'q': 2, 'p': 4,
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
    stable = []
    for x in sols:
        ev = np.linalg.eigvals(jac(x[0], x[1], S, p))
        if np.all(np.real(ev) < 0):
            stable.append(x)
    return stable


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


T_MAX = 36.0
T_ASYM = 3.0
DELTA = 0.02
t_pre = np.arange(0, T_ASYM + 1e-9, 0.25)
t_post = np.arange(T_ASYM, T_MAX + 1e-9, 0.25)


def run_trajectory(p, sign):
    y0 = [0.01, 0.05, 0.05, 0.05]
    pre = solve_ivp(lambda tt, y: full_rhs(tt, y, p), [0, T_ASYM], y0,
                    t_eval=t_pre, method='LSODA', max_step=0.1,
                    rtol=1e-9, atol=1e-11)
    y3 = pre.y[:, -1].copy()
    y3[2] += DELTA * sign
    y3[3] -= DELTA * sign
    post = solve_ivp(lambda tt, y: full_rhs(tt, y, p), [T_ASYM, T_MAX], y3,
                     t_eval=t_post, method='LSODA', max_step=0.1,
                     rtol=1e-9, atol=1e-11)
    t = np.concatenate([pre.t, post.t[1:]])
    y = np.hstack([pre.y, post.y[:, 1:]])
    return t, y


P_NO_INH = dict(P, Krl=1e12, Klr=1e12)
P_NO_SA = dict(P, etaL=0.0, etaR=0.0)

Sgrid = np.concatenate([np.linspace(0.02, 1.0, 200),
                        np.linspace(1.05, 6.0, 120)])


def branch_points(p, starts):
    pts = []
    for S in Sgrid:
        for x in find_fixed_points(S, p, starts):
            pts.append((S, float(x[0]), float(x[1])))
    return pts


print('Computing SI perturbation data...')
pts_no_inh = branch_points(P_NO_INH, coarse_starts())
pts_no_sa = branch_points(P_NO_SA, coarse_starts())

t, y_g = run_trajectory(P, +1)
_, y_s = run_trajectory(P, -1)
t2, y_g_noinh = run_trajectory(P_NO_INH, +1)
_, y_s_noinh = run_trajectory(P_NO_INH, -1)
t3, y_g_nosa = run_trajectory(P_NO_SA, +1)
_, y_s_nosa = run_trajectory(P_NO_SA, -1)

H_g, S_g, L_g, R_g = y_g
H2, S2, L_noinh, R_noinh = y_g_noinh
H3, S3, L_nosa, R_nosa = y_g_nosa

fig, axs = plt.subplots(
    2, 2, figsize=(14.0, 11.5),
    gridspec_kw={'left': 0.085, 'right': 0.90, 'top': 0.95,
                 'bottom': 0.075, 'wspace': 0.30, 'hspace': 0.32})

# A: clamped-S phase plane without mutual inhibition
ax = axs[0, 0]
aS = np.array([p[0] for p in pts_no_inh])
aL = np.array([np.log10(max(p[1], 1e-6)) for p in pts_no_inh])
aR = np.array([np.log10(max(p[2], 1e-6)) for p in pts_no_inh])
scA = ax.scatter(aL, aR, c=aS, cmap='cividis_r',
                 norm=matplotlib.colors.LogNorm(vmin=0.02, vmax=6),
                 s=22, alpha=0.9, zorder=3)
ax.plot([-3, 2], [-3, 2], 'k--', lw=1.4)
ax.set_xlabel(r'$\log_{10} L$ (lag)', fontsize=13)
ax.set_ylabel(r'$\log_{10} R$ (regA)', fontsize=13)
ax.set_title('No mutual inhibition: single $L\\approx R$ branch',
             fontsize=12)
ax.set_xlim(-3, 2)
ax.set_ylim(-3, 2)
cbarA = fig.colorbar(scA, ax=ax, fraction=0.046, pad=0.04)
cbarA.set_label('S', fontsize=11)

# B: full dynamics without mutual inhibition
ax = axs[0, 1]
ax.plot(t2, L_noinh, color='#377eb8', lw=6, alpha=0.6, label='L (lag)',
        marker='o', markevery=(0, 24), ms=7, mfc='#377eb8', mec='k')
ax.plot(t2, R_noinh, color='#4daf4a', lw=3, alpha=1.0, label='R (regA)',
        marker='s', markevery=(12, 24), ms=6, mfc='#4daf4a', mec='k')
ax.set_xlabel('Time (h)', fontsize=13)
ax.set_ylabel('Expression level (log)', fontsize=13)
ax.set_xlim(0, T_MAX)
ax.set_ylim(5e-3, 50)
ax.set_yscale('log')
ax.axvline(T_ASYM, color='gray', ls='--', lw=1.2)
ax.set_title('No mutual inhibition: no divergence', fontsize=12)
ax.legend(loc='upper left', fontsize=11, frameon=False)

# C: clamped-S phase plane without self-activation
ax = axs[1, 0]
cS = np.array([p[0] for p in pts_no_sa])
cL = np.array([np.log10(max(p[1], 1e-6)) for p in pts_no_sa])
cR = np.array([np.log10(max(p[2], 1e-6)) for p in pts_no_sa])
scC = ax.scatter(cL, cR, c=cS, cmap='cividis_r',
                 norm=matplotlib.colors.LogNorm(vmin=0.02, vmax=6),
                 s=22, alpha=0.9, zorder=3)
ax.plot([-3, 2], [-3, 2], 'k--', lw=1.4)
ax.set_xlabel(r'$\log_{10} L$ (lag)', fontsize=13)
ax.set_ylabel(r'$\log_{10} R$ (regA)', fontsize=13)
ax.set_title('No self-activation: bistability persists', fontsize=12)
ax.set_xlim(-3, 2)
ax.set_ylim(-3, 2)
cbarC = fig.colorbar(scC, ax=ax, fraction=0.046, pad=0.04)
cbarC.set_label('S', fontsize=11)

# D: split into top time courses and bottom |L-R| difference
fig.canvas.draw()
pos_d = axs[1, 1].get_position()
axs[1, 1].remove()
gap_d = pos_d.height * 0.06
h_d = (pos_d.height - gap_d) / 2.0
ax_dtop = fig.add_axes([pos_d.x0, pos_d.y0 + h_d + gap_d,
                        pos_d.width, h_d])
ax_dbot = fig.add_axes([pos_d.x0, pos_d.y0, pos_d.width, h_d])

ax_dtop.plot(t, L_g, color='#377eb8', lw=3,
             label='L with self-activation')
ax_dtop.plot(t, R_g, color='#4daf4a', lw=3,
             label='R with self-activation')
ax_dtop.plot(t3, L_nosa, color='#377eb8', lw=2.2, ls='--',
             label='L without self-activation')
ax_dtop.plot(t3, R_nosa, color='#4daf4a', lw=2.2, ls='--',
             label='R without self-activation')
ax_dtop.set_xlabel('')
ax_dtop.set_ylabel('Expression level (log)', fontsize=13)
ax_dtop.set_xlim(0, T_MAX)
ax_dtop.set_ylim(5e-3, 50)
ax_dtop.set_yscale('log')
ax_dtop.axvline(T_ASYM, color='gray', ls='--', lw=1.2)
ax_dtop.tick_params(axis='x', labelbottom=False)
ax_dtop.set_title('No self-activation: slower divergence', fontsize=12)
ax_dtop.legend(loc='center right', fontsize=10, frameon=False)

ax_dbot.plot(t, np.log10(np.abs(L_g - R_g) + 1e-12), color='k', lw=2.5,
             label='with self-activation')
ax_dbot.plot(t3, np.log10(np.abs(L_nosa - R_nosa) + 1e-12), color='k',
             lw=2.2, ls='--', label='without self-activation')
ax_dbot.set_xlabel('Time (h)', fontsize=13)
ax_dbot.set_ylabel(r'$\log_{10}|L-R|$', fontsize=13)
ax_dbot.set_xlim(0, T_MAX)
ax_dbot.set_ylim(-3, 2)
ax_dbot.axvline(T_ASYM, color='gray', ls='--', lw=1.2)
ax_dbot.legend(loc='center right', fontsize=10, frameon=False)

for ax, label in [(axs[0, 0], 'A'), (axs[0, 1], 'B'),
                  (axs[1, 0], 'C'), (ax_dtop, 'D')]:
    ax.text(-0.22, 1.06, label, transform=ax.transAxes, ha='left',
            va='bottom', fontsize=18, fontweight='bold')

fig_name = os.path.join(OUT, 'fig31_SI.png')
fig.savefig(fig_name, dpi=200)
plt.close(fig)

print('final (full model): L={:.3f} R={:.3f}'.format(L_g[-1], R_g[-1]))
print('final (no inhibition): L={:.3f} R={:.3f}'.format(
    L_noinh[-1], R_noinh[-1]))
print('final (no self-activation): L={:.3f} R={:.3f}'.format(
    L_nosa[-1], R_nosa[-1]))
print('Saved', fig_name)
