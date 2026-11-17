import argparse
import json
import os

import numpy as np

import fig4_final as F

HERE = os.path.dirname(os.path.abspath(__file__))
JSON = os.path.join(HERE, 'fig4-final.json')


def iter_axes(fig):
    import matplotlib.axes as maxes
    out = []

    def rec(ax):
        out.append(ax)
        for c in ax.get_children():
            if isinstance(c, maxes.Axes) and c not in out:
                rec(c)
    for ax in fig.axes:
        rec(ax)
    return out


def report_text_overlaps(fig, tag, pad=-1.0):
    import matplotlib.text as mtext
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    items, boxes = [], []
    owner = {}
    for ai, ax in enumerate(iter_axes(fig)):
        tagax = 'ax%d' % ai
        for t in list(ax.texts) + [ax.title] + list(ax.get_xticklabels()) \
                + list(ax.get_yticklabels()):
            if isinstance(t, mtext.Text) and t.get_text().strip():
                items.append(t)
                owner[id(t)] = tagax
        lg = ax.get_legend()
        if lg is not None:
            boxes.append((lg.get_window_extent(r), 'legend'))
    for t in fig.texts:
        if t.get_text().strip():
            items.append(t)
            owner[id(t)] = 'fig'
    hits = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a = items[i].get_window_extent(r)
            b = items[j].get_window_extent(r)
            if (a.x1 < b.x0 + pad or b.x1 < a.x0 + pad
                    or a.y1 < b.y0 + pad or b.y1 < a.y0 + pad):
                continue
            hits.append((items[i].get_text()[:30] + ' @'
                         + owner.get(id(items[i]), '?'),
                         items[j].get_text()[:30] + ' @'
                         + owner.get(id(items[j]), '?')))
    for t in items:
        a = t.get_window_extent(r)
        for bb, nm in boxes:
            if (a.x1 < bb.x0 + pad or bb.x1 < a.x0 + pad
                    or a.y1 < bb.y0 + pad or bb.y1 < a.y0 + pad):
                continue
            hits.append((t.get_text()[:30], '<%s frame>' % nm))
    if hits:
        print('   [!!] %s: %d text overlaps' % (tag, len(hits)))
        for a, b in hits[:6]:
            print('        "%s" <-> "%s"' % (a, b))
    else:
        print('   [ok] %s: no text overlap' % tag)
    return len(hits)


def selfcheck(d):
    K7 = F.K7
    main, win = d['main_results'], d['within_results']

    print('\n--- check 1: main (C) vs (D) bit-identical (L-R symmetry under merged K) ---')
    a = np.array(main['log10(L_high/R_low)']['ST'])
    b = np.array(main['log10(R_high/R_low)']['ST'])
    a1 = np.array(main['log10(L_high/R_low)']['S1'])
    b1 = np.array(main['log10(R_high/R_low)']['S1'])
    print('   max|ST_C-ST_D| = %.3e   max|S1_C-S1_D| = %.3e'
          % (np.max(np.abs(a - b)), np.max(np.abs(a1 - b1))))

    print('\n--- check 2: inset (C) and (D) are L-R mirror images ---')
    wc = {k: np.array(win['log10(L_high/R_low)'][k]) for k in
          ('S1', 'ST', 'S1_conf', 'ST_conf')}
    wd = {k: np.array(win['log10(R_high/R_low)'][k]) for k in
          ('S1', 'ST', 'S1_conf', 'ST_conf')}
    mir = {'K_SL': 'K_SR', 'K_SR': 'K_SL', 'K_RL': 'K_LR', 'K_LR': 'K_RL',
           'K_L': 'K_R', 'K_R': 'K_L', 'K_SH': 'K_SH'}
    for n in K7:
        i, j = K7.index(n), K7.index(mir[n])
        print('   %-5s vs %-5s : ST %.4f / %.4f  (diff %+.2e)'
              % (n, mir[n], wc['ST'][i], wd['ST'][j],
                 wc['ST'][i] - wd['ST'][j]))

    print('\n--- check 3: K_SH in the (B) inset (structurally 1.00) ---')
    wh = {k: np.array(win['H peak/plateau'][k]) for k in
          ('S1', 'ST', 'S1_conf', 'ST_conf')}
    i = K7.index('K_SH')
    print('   K_SH : S1 = %.6f  ST = %.6f  (ST_95CI ±%.3f)'
          % (wh['S1'][i], wh['ST'][i], wh['ST_conf'][i]))
    rest = [K7[k] for k in range(7) if k != i]
    print('   ST of the other six thresholds: '
          + ', '.join('%s=%.2e' % (n, wh['ST'][K7.index(n)]) for n in rest))
    print('   -> the (H,S) subsystem decouples from (L,R); K_SH is the only threshold entering H,'
          'so ST = 1.00 is structural')

    print('\n--- check 4: bar-top range of the four insets ---')
    glo, gtop = 0.0, 0.0
    for key, _, _ in F.PANELS:
        r = win[key]
        s1 = np.array(r['S1'])
        st = np.array(r['ST'])
        c1 = np.array(r['S1_conf'])
        ct = np.array(r['ST_conf'])
        glo = min(glo, float(np.min(np.minimum(s1 - c1, st - ct))), 0.0)
        gtop = max(gtop, float(np.max(np.maximum(s1 + c1, st + ct))))
        print('   %-22s bar top (incl. CI) in [%.3f, %.3f]'
              % (key, float(np.min(np.minimum(s1 - c1, st - ct))),
                 float(np.max(np.maximum(s1 + c1, st + ct)))))
    print('   global [%.3f, %.3f]  common y-limit %s -> %s'
          % (glo, gtop, F.INSET_YLIM,
             'OK' if gtop <= F.INSET_YLIM[1] + 1e-9 else
             'the axis top clips the upper error bars (stated in the caption)'))

    print('\n--- check 5: main-panel sum(S1) / sum(ST) (values quoted in the text) ---')
    for key, _, _ in F.PANELS:
        r = main[key]
        print('   %-22s ΣS1 = %.3f   ΣST = %.3f'
              % (key, float(np.sum(r['S1'])), float(np.sum(r['ST']))))
    print()

TABLE1_VALUE = {'aH': 3.5, 'aL': 3.5, 'eta': 0.7, 'beta': 0.0035,
                'K': 0.5, 'Kc': 1.5, 'delta': 7.0, 'p': 3, 'n': 3, 'm': 3,
                'q': 2, 'b0': 0.055, 'bH': 0.9, 'Khs': 0.5, 'gS': 0.1}
TABLE1_RANGE = {'aH': (1.75, 5.25), 'aL': (1.75, 5.25), 'eta': (0.35, 1.05),
                'beta': (0.00175, 0.00525),
                'K': (0.5, 1.5), 'Kc': (1.0, 2.0), 'delta': (3.5, 10.5),
                'p': (2, 4), 'n': (2, 4), 'm': (2, 4), 'q': (1, 3),
                'b0': (0.03, 0.08), 'bH': (0.5, 1.3), 'Khs': (0.3, 0.8),
                'gS': (0.05, 0.20)}

def check_table1():
    print('=== Table 1 "Value used here" vs BASE ===')
    bad = 0
    for k, v in TABLE1_VALUE.items():
        got = F.BASE['K_SH'] if k == 'K' else F.BASE[k]
        if abs(got - v) > 1e-12:
            print('  %-6s table=%g  code=%g  <<< mismatch' % (k, v, got))
            bad += 1
    print('  -> %s' % ('all consistent' if not bad else '%d items differ' % bad))
    print('=== Table 1 ranges vs BOUNDS ===')
    bad = 0
    for k, rng in TABLE1_RANGE.items():
        if k == 'K':
            ok = all(tuple(F.BOUNDS[kk]) == rng for kk in F.K7)
            got = rng if ok else [tuple(F.BOUNDS[kk]) for kk in F.K7]
        else:
            got = tuple(F.BOUNDS[k])
            ok = all(abs(a - b) < 1e-12 for a, b in zip(got, rng))
        if not ok:
            print('  %-6s table=%s  code=%s  <<< mismatch' % (k, rng, got))
            bad += 1
    print('  -> %s' % ('all consistent' if not bad else '%d items differ' % bad))
    print('  note: for alpha/eta/beta Table 1 gives the literature range while sampling uses +-50%,'
          'the two are not equivalent (stated in the table caption).')

def main():
    ap = argparse.ArgumentParser(
        description='Fig.4 self-check: Table 1, C/D mirror, sum(S1)/sum(ST), text overlaps')
    ap.add_argument('--table1', action='store_true', help='check Table 1 only')
    ap.add_argument('--results', action='store_true',
                    help='result checks only (reads fig4-final.json)')
    ap.add_argument('--plot-qa', action='store_true',
                    help='plot and run the text-overlap check (rewrites fig4-final.*)')
    ap.add_argument('--dpi', type=int, default=200)
    args = ap.parse_args()

    if not (args.table1 or args.results or args.plot_qa):
        args.table1 = args.results = True

    if args.table1:
        check_table1()

    if args.results or args.plot_qa:
        if not os.path.exists(JSON):
            raise SystemExit('missing %s; run python fig4_final.py first' % JSON)
        with open(JSON, encoding='utf-8') as f:
            d = json.load(f)
        print('read %s (N=%s)' % (os.path.basename(JSON),
                                  format(d['meta']['N_base'], ',')))
        if args.results:
            selfcheck(d)
        if args.plot_qa:
            F.plot_all(d, dpi=args.dpi, qa=report_text_overlaps)


if __name__ == '__main__':
    main()
