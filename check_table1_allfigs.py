import ast, io, os, sys

FIG4 = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(FIG4)
F2 = os.path.join(CODE, 'fig2', 'fig31_3x3.py')
F2OLD = F2
F3 = os.path.join(CODE, 'fig3', 'Fig32.py')
F5 = os.path.join(CODE, 'fig5', 'fig5_v4.py')


def grab(path, names):
    src = io.open(path, encoding='utf-8').read()
    tree = ast.parse(src)
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            t = node.targets[0]
            if isinstance(t, ast.Name) and t.id in names:
                try:
                    out[t.id] = ast.literal_eval(node.value)
                except Exception:
                    pass
    return out


p2 = grab(F2, {'P'})['P']
p2o = grab(F2OLD, {'P'})['P']
d3 = grab(F3, {'BASE', 'BOUNDS'})
d5 = grab(F5, {'SYM'})['SYM']
sys.path.insert(0, FIG4)
import model_v7 as M  # noqa

print('=' * 78)
print('[0] do the two Fig.2 scripts use the same P? (guard against script drift)')
print('=' * 78)
print('  paper version  p=%s  pm=%s' % (p2.get('p'), p2.get('pm')))
print('  model_v0    p=%s  pm=%s' % (p2o.get('p'), p2o.get('pm')))
if p2 == p2o:
    print('  -> identical')
else:
    print('  -> *** differs, item by item ***')
    for k in sorted(set(p2) | set(p2o)):
        if p2.get(k) != p2o.get(k):
            print('     %-6s zxiu=%-8s model_v0=%s' % (k, p2.get(k), p2o.get(k)))
    print('     fix: make the model_v0 copy match the paper version (Table 1 follows the paper)')

print()
print('=' * 78)
print('[1] Table 1 "Value used here" vs BASE in the figure scripts')
print('=' * 78)
T1 = {'aH': 3.5, 'aL': 3.5, 'eta': 0.7, 'beta': 0.0035, 'K': 0.5, 'Kc': 1.5,
      'delta': 7.0, 'p': 3, 'n': 3, 'm': 3, 'q': 2,
      'b0': 0.055, 'bH': 0.9, 'Khs': 0.5, 'gS': 0.1}
M2 = {'aH': 'aH', 'aL': 'aL', 'eta': 'etaL', 'beta': 'bL', 'K': 'Ksl',
      'Kc': 'Kc', 'delta': 'd', 'p': 'p', 'n': 'n', 'm': 'm', 'q': 'q',
      'b0': 'b0', 'bH': 'bH', 'Khs': 'Khs', 'gS': 'gS'}
M4 = {k: k for k in T1}
M4['K'] = 'K_SH'

print('%-8s %-9s %-9s %-9s %-9s %s' % ('param', 'Table1', 'Fig.2', 'Fig.3', 'Fig.4', 'flag'))
bad = []
for k, tv in T1.items():
    a = p2.get(M2[k])
    b = d3['BASE'].get(k)
    key = M4[k]
    c = M.BASE.get(key)
    flags = []
    for nm, x in (('F2', a), ('F3', b), ('F4', c)):
        if x is None:
            flags.append(nm + ':NA')
        elif abs(x - tv) > 1e-9:
            flags.append('%s:%g' % (nm, x))
            bad.append((k, nm))
    print('%-8s %-9g %-9s %-9s %-9s %s' % (k, tv, a, b, c, ' '.join(flags) or 'ok'))

print()
print('=' * 78)
print('[2] Table 1 last-column ranges vs BOUNDS of Fig.3 / Fig.4')
print('=' * 78)
T1R = {'Kc': (1.0, 2.0), 'delta': (3.5, 10.5), 'b0': (0.03, 0.08),
       'bH': (0.5, 1.3), 'Khs': (0.3, 0.8), 'gS': (0.05, 0.20),
       'p': (2, 4), 'n': (2, 4), 'm': (2, 4), 'q': (1, 3),
       'K': (0.5, 1.5)}
names3 = ['aH', 'aL', 'eta', 'beta', 'K', 'Kc', 'delta', 'p', 'n', 'm', 'q',
          'b0', 'bH', 'Khs', 'gS']
b3 = dict(zip(names3, d3['BOUNDS']))
for k, rng in T1R.items():
    f3 = tuple(b3[k])
    f4 = tuple(M.BOUNDS[M4[k]]) if M4[k] in M.BOUNDS else None
    ok3 = (f3 == rng)
    ok4 = (f4 == rng) if f4 else None
    print('  %-6s table=%-12s Fig.3=%-12s %-4s Fig.4=%-12s %s' %
          (k, rng, f3, 'OK' if ok3 else '<<<', f4, 'OK' if ok4 else '<<<'))

print()
print('=' * 78)
print('[3] Table 1 caption retuned values vs SYM in the Fig.5 code')
print('=' * 78)
NOTE = {'Kc': 2.275, 'delta': 7.60, 'm': 3.87, 'b0': 0.0523, 'bH': 0.634,
        'Khs': 0.264, 'gS': 0.116}
for k, v in NOTE.items():
    got = d5.get(k)
    print('  %-6s caption=%-9g Fig.5=%-9s diff=%s' %
          (k, v, got, '%.2e' % abs(got - v) if got is not None else 'NA'))
print('  %-6s caption=%-9g Fig.5=%-9s diff=%.2e' % ('bL', 0.00394, d5.get('bL'),
                                                 abs(d5.get('bL', 0) - 0.00394)))
print('  Fig.5 p/q/n: %s / %s / %s' % (d5.get('p'), d5.get('q'), d5.get('n')))
print('  Fig.5 aH/eta/K: %s / %s / %s' % (d5.get('aH'), d5.get('etaL'), d5.get('KSL')))
print('  Fig.5 aR/etaR: %s / %s  (symmetry)' % (d5.get('aR'), d5.get('etaR')))

print()
print('=' * 78)
print('[4] conclusion')
print('=' * 78)
print('  mismatched items:', bad or 'none')
