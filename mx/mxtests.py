#!/usr/bin/env python3
"""MX test manifest tool: turns mx/tests.txt into test names and build flags.

  ./mxtests.py list [PATTERN]   test names (substring filter)
  ./mxtests.py show NAME        template, data header and compiler flags of one test
  ./mxtests.py mk               make fragment (mx/Makefile includes this)

Template tests generate their data header at build time (mx/golden); its golden uses the per mesh row
product / accumulator precision of a config header: the test's params=, else the manifest's `precision` default
for its DIM. Kernel tests (lutk, mxk_*) build an mx/src kernel against their own params header, with data blobs
generated at build time from the manifest's `data` lines.
"""
import argparse
import itertools
import os
import shlex
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
MANIFEST = os.path.join(ROOT, 'tests.txt')

FORMATS = {'e4m3', 'e5m2', 'e4m3s', 'e2m3', 'e3m2', 'fp4'}


class Test:
    """src: path under mx/ (templates/<t>.c or src/<k>.c); data: generated data header (template tests), made by
    `gen` (mx/golden command) with the precision of config `prec` (dim/bf16 resolved by load()); blob: generated
    data set (kernel tests); params: gemmini_params_<params>.h to compile against; group: make group."""
    def __init__(self, name, template, data, defs, src=None, blob=None, params=None, group='default', gen=None,
                 dim=16, bf16=False):
        self.name, self.template, self.data, self.defs = name, template, data, defs
        self.src = src or f'templates/{template}.c'
        self.blob, self.params, self.group = blob, params, group
        self.gen, self.dim, self.bf16, self.prec = gen, dim, bf16, None


SPEC = {'e4m3': 'fp8:e4m3', 'e5m2': 'fp8:e5m2', 'e4m3s': 'fp8:e4m3s', 'e2m3': 'fp6:e2m3', 'e3m2': 'fp6:e3m2',
        'fp4': 'fp4:e2m1'}


def mnk(size):
    d = [int(x) for x in size.split('x')]
    return d if len(d) == 3 else d + [d[1]]


def asym(p):
    act, wei = p['act'], p['wei']
    if act not in FORMATS or wei not in FORMATS:
        raise ValueError(f'unknown format in act={act} wei={wei}')
    dim, size = int(p.get('dim', 16)), p.get('size', '64x64')
    out, tag = p.get('out', 'bf16'), p.get('tag')
    name = f'matmul_tiled_asym_{act}_{wei}_{size}'
    name += '_rqfp4' if out == 'fp4' else ''
    name += f'_dim{dim}' if dim != 16 else ''
    name += f'_{tag}' if tag else ''
    data = name.replace('matmul_tiled_', 'matmul_data_').replace('_64x64', '')
    defs = {'ACT': act, 'WEI': wei, 'MESH_DIM': dim, 'OUT_FP4': int(out == 'fp4'), 'CYCLES': int(tag == 'bf16')}
    for k in ('alt', 'wdiff'):
        if k in p:
            defs[k.upper()] = int(p[k])
    M, N, K = mnk(size)
    gen = f'gen_asym.py --act {SPEC[act]} --wei {SPEC[wei]} --dim {dim} --M {M} --N {N} --K {K}'
    gen += ' --requant fp4' if out == 'fp4' else ''
    return Test(name, 'asym', data, defs, gen=gen, dim=dim, bf16=tag == 'bf16', params=p.get('params'))


LUT_PREFIX = {'e4m3': 'fp8_e4m3_lut', 'e5m2': 'fp8_e5m2', 'e2m3': 'fp6_e2m3_lut', 'e3m2': 'fp6_e3m2_lut'}


def lut_requant(p):
    fmt = p['fmt']
    if fmt not in LUT_PREFIX:
        raise ValueError(f'unknown fmt={fmt}')
    dim, size, tag = int(p.get('dim', 16)), p.get('size', '64x64'), p.get('tag')
    suffix = (f'_dim{dim}' if dim != 16 else '') + (f'_{tag}' if tag else '')
    name = f'matmul_tiled_{LUT_PREFIX[fmt]}_{size}_requant{suffix}'
    data = f'matmul_data_mx_lut_{fmt}_{size}{suffix}'
    return Test(name, 'lut_requant', data, {'FMT': fmt, 'MESH_DIM': dim}, gen=lut_gen(fmt, dim, size), dim=dim,
                bf16=tag == 'bf16', params=p.get('params'))


def lut_gen(fmt, dim, size):
    M, N, K = mnk(size)
    return f'gen_mx_lut.py --format {fmt} --dim {dim} --M {M} --N {N} --K {K}'


LUT_BF16_DIM = {'lut_bf16': 16, 'lut_bf16_d32': 32, 'lut_bf16_d8': 8}


def lut_bf16(kind):
    def make(p):
        fmt, dim, size, tag = p['fmt'], LUT_BF16_DIM[kind], p.get('size', '64x64'), p.get('tag')
        suffix = (f'_dim{dim}' if dim != 16 else '') + (f'_{tag}' if tag else '')
        mid = '_nonrequant' if kind == 'lut_bf16_d8' else ''
        name = f'matmul_tiled_{LUT_PREFIX[fmt]}_{size}{mid}{suffix}'
        return Test(name, kind, f'matmul_data_mx_lut_{fmt}_{size}{suffix}', {'FMT': fmt, 'MESH_DIM': dim},
                    gen=lut_gen(fmt, dim, size), dim=dim, bf16=tag == 'bf16', params=p.get('params'))
    return make


# Older plain-format tests: template -> (test name, data header); {d} = _dim<N> (empty for dim 16).
PLAIN = {
    'fp8_requant_sq':  ('matmul_tiled_fp8_{size}_requant{d}', 'matmul_fp8_{size}{d}'),
    'fp8_requant':     ('matmul_tiled_fp8_{size}_requant{d}', 'matmul_fp8_{size}{d}'),
    'fp8_requant_big': ('matmul_tiled_fp8_{size}_requant{d}', 'matmul_fp8_{size}{d}'),
    'fp8_single':      ('matmul_tiled_fp8_{size}_single{d}', 'matmul_fp8_{size}{d}'),
    'fp4_requant':     ('matmul_tiled_fp4_{size}_requant{d}', 'matmul_fp4_{size}{d}'),
    'fp4_nonrequant':  ('matmul_tiled_fp4_{size}_nonrequant{d}', 'matmul_fp4_{size}{d}'),
    'fp8_chain':       ('matmul_tiled_fp8_{size}_chain{d}', 'matmul_fp8_{size}_chain{d}'),
    'fp4_chain':       ('matmul_tiled_fp4_{size}_chain{d}', 'matmul_fp4_{size}_chain{d}'),
}


def plain(kind):
    def make(p):
        dim, size = int(p.get('dim', 16)), p.get('size', '64x64')
        d = f'_dim{dim}' if dim != 16 else ''
        name, data = (x.format(size=size, d=d) for x in PLAIN[kind])
        shape = data.removeprefix('matmul_')
        shape = 'chain_' + shape.replace('_chain', '') if '_chain' in shape else shape
        return Test(name, kind, data, {'MESH_DIM': dim}, gen=f'gen_plain.py {shape}', dim=dim, params=p.get('params'))
    return make


# --- kernel tests (opt-in group "mxk"): per-K-block LUT, mixed-K, MicroMix --------------------------------------

def _lutk_var(var):
    """fp8 = all MXFP8; lut<i> / fp4_<i> = mask i (LUT or plain MXFP4 on its low-precision K-blocks)."""
    if var == 'fp8':
        return {'LK_VAR': -1}
    if var.startswith('lut'):
        return {'LK_VAR': int(var[3:])}
    if var.startswith('fp4_'):
        return {'LK_VAR': int(var[4:]), 'LK_LO_FP4': 1}
    raise ValueError(f'unknown var={var}')


def _q(header):
    return f"'\"{header}\"'"


def lutk(p):
    tag, dim, var, pad = p['tag'], int(p['dim']), p['var'], int(p.get('pad', 0))
    name = f'lutk_{tag}_d{dim}_{var}' + (f'_pad{pad}' if pad else '')
    defs = {'LK_DATA_H': _q(f'lutk_{tag}.h'), **_lutk_var(var), **({'LK_PADI': pad} if pad else {})}
    return Test(name, None, None, defs, src='src/lutk_proj.c', blob=f'lutk_{tag}', params=f'pwr_base_dim{dim}', group='mxk')


def mxk_kblock(p):
    base, tag, dim, var = int(p['base']), p['tag'], int(p['dim']), p['var']
    blob = f'mxk_kblock6_{tag}' if base == 6 else f'lutk_{tag}'
    mix = var.endswith('_mix')
    defs = {'MK_CTR': 1, **({'MK_FP6': 1} if base == 6 else {}), **({'LK_MIX': 1} if mix else {}),
            'LK_DATA_H': _q(f'{blob}.h'), **_lutk_var(var.removesuffix('_mix'))}
    return Test(f'mxk_kblock{base}_{tag}_d{dim}_{var}', None, None, defs, src='src/mxk_kblock.c', blob=blob,
                params=f'pwr_base_dim{dim}', group='mxk')


def mxk_micromix(p):
    tag, dim, nf = p['tag'], int(p['dim']), int(p.get('nf', 0))
    defs = {'MK_CTR': 1, **({'MM_NOFENCE': 1} if nf else {}), 'MM_DATA_H': _q(f'mxk_micromix_{tag}.h')}
    return Test(f'mxk_micromix_{tag}_d{dim}' + ('_nf' if nf else ''), None, None, defs, src='src/mxk_micromix.c',
                blob=f'mxk_micromix_{tag}', params=f'pwr_base_dim{dim}', group='mxk')


def mxk_kmixa(p):
    tag, dim = p['tag'], int(p['dim'])
    return Test(f'mxk_kmixa_{tag}_d{dim}', None, None, {'KA_DATA_H': _q(f'mxk_kmixa_{tag}.h')},
                src='src/mxk_kmixa.c', blob=f'mxk_kmixa_{tag}', params=f'pwr_base_dim{dim}', group='mxk')


#: power kernels: name -> (kernel: k8 / k6 = mxk_kblock 8 / 6, mm = mxk_micromix; variant or micromix data tag)
PWR = {'fp8': ('k8', 'fp8'), 'fp6': ('k6', 'fp8'), 'w4a8': ('k8', 'fp4_0_mix'), 'lut': ('k8', 'lut0_mix'),
       'lutmix': ('k8', 'lut1_mix'), 'lutmix6': ('k6', 'lut1_mix'), 'a4w4': ('mm', 'pwr4'), 'micromix': ('mm', 'pwr468')}


def mxk_pwr(p):
    """Compute-only power kernels (no checks or counters, MK_PWR_ITERS repeats); chk=1: the same schedule, checked."""
    w, dim, chk = p['kernel'], int(p['dim']), int(p.get('chk', 0))
    kern, var = PWR[w]
    defs = {'MK_CB_DATA': 1} if chk else {'MK_POWER': 1, 'MK_PWR_ITERS': 8}
    if kern == 'mm':
        blob, src = f'mxk_micromix_{var}', 'src/mxk_micromix.c'
        defs.update({'MM_NOFENCE': 1, 'MM_DATA_H': _q(f'{blob}.h')})
    else:
        blob, src = ('mxk_kblock6_pwr' if kern == 'k6' else 'lutk_pwr'), 'src/mxk_kblock.c'
        defs.update({**({'MK_FP6': 1} if kern == 'k6' else {}), **({'LK_MIX': 1} if var.endswith('_mix') else {}),
                     'LK_DATA_H': _q(f'{blob}.h'), **_lutk_var(var.removesuffix('_mix'))})
    return Test(f'mxk_pwr_{w}_d{dim}' + ('_chk' if chk else ''), None, None, defs, src=src, blob=blob,
                params=f'pwr_base_dim{dim}', group='mxk')


def src(p):
    """A one-off test in mx/src/<name>.c, built against its own config's params header (params=)."""
    return Test(p['name'], None, None, {}, src=f"src/{p['name']}.c", params=p.get('params'))


TEMPLATES = {'asym': asym, 'src': src, 'lutk': lutk, 'mxk_pwr': mxk_pwr, 'mxk_kblock': mxk_kblock, 'mxk_micromix': mxk_micromix, 'mxk_kmixa': mxk_kmixa, 'lut_requant': lut_requant, **{k: lut_bf16(k) for k in LUT_BF16_DIM}, **{k: plain(k) for k in PLAIN}}


def load(path=MANIFEST, data=None):
    """Tests from the manifest; `data` lines (name=<blob> gen="<generator args>") go into the data dict."""
    tests, precision = [], {}
    data = {} if data is None else data
    for lineno, line in enumerate(open(path), 1):
        words = shlex.split(line, comments=True)
        if not words:
            continue
        kind, kvs = words[0], dict(w.split('=', 1) for w in words[1:])
        if kind == 'data':
            data[kvs['name']] = kvs['gen']
            continue
        if kind == 'precision':
            precision.update(kvs)
            continue
        if kind not in TEMPLATES:
            sys.exit(f'{path}:{lineno}: unknown template "{kind}"')
        keys = list(kvs)
        for combo in itertools.product(*(kvs[k].split(',') for k in keys)):
            try:
                tests.append(TEMPLATES[kind](dict(zip(keys, combo))))
            except (KeyError, ValueError) as e:
                sys.exit(f'{path}:{lineno}: {e}')
    for t in tests:
        if t.gen:
            key = ('bf16_' if t.bf16 else '') + f'dim{t.dim}'
            t.prec = t.params or precision.get(key)
            if not t.prec:
                sys.exit(f'{t.name}: no params= and no `precision {key}=` default')
    missing = {t.blob for t in tests if t.blob and t.blob not in data}
    if missing:
        sys.exit(f'no data line for: {sorted(missing)}')
    names = [t.name for t in tests]
    dups = {n for n in names if names.count(n) > 1}
    if dups:
        sys.exit(f'duplicate test names: {sorted(dups)}')
    return tests


def cflags(t):
    flags = [f'-DMX_DATA_H=\'"{t.data}.h"\''] if t.gen else [f'-DMX_DATA_H=\'"include/{t.data}.h"\''] if t.data else []
    flags += [f'-D{k}={v}' for k, v in t.defs.items()]
    return ' '.join(flags)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('list').add_argument('pattern', nargs='?', default='')
    sub.add_parser('show').add_argument('name')
    sub.add_parser('mk')
    a = ap.parse_args()
    data = {}
    tests = load(data=data)

    if a.cmd == 'list':
        for t in tests:
            if a.pattern in t.name:
                print(t.name)
    elif a.cmd == 'show':
        t = next((t for t in tests if t.name == a.name), None)
        if t is None:
            sys.exit(f'no test named {a.name}')
        print(f'source    mx/{t.src}')
        if t.gen:
            print(f'data      data/{t.prec}/{t.data}.h  <- mx/golden/{t.gen} --params include/gemmini_params_{t.prec}.h')
        elif t.data:
            print(f'data      include/{t.data}.h')
        if t.blob:
            print(f'data      {t.blob}.bin/.h  <- mx/golden/{data[t.blob]}')
        if t.params:
            print(f'params    include/gemmini_params_{t.params}.h')
        print(f'flags     {cflags(t)}')
    elif a.cmd == 'mk':
        print('# Generated by mxtests.py from mx/tests.txt -- do not edit.')
        for t in tests:
            print(f'mx_{t.group}_tests += {t.name}')
            print(f'MX_SRC_{t.name} := {t.src}')
            print(f'MX_DEFS_{t.name} := {cflags(t)}')
            if t.params:
                print(f'MX_PARAMS_{t.name} := {t.params}')
            if t.blob:
                print(f'MX_BLOB_{t.name} := {t.blob}')
            if t.gen:
                print(f'MX_HDR_{t.name} := {t.prec}/{t.data}')
        hdrs = {}
        for t in tests:
            if t.gen:
                hdrs.setdefault(f'{t.prec}/{t.data}', (t.gen, t.prec))
        for h, (gen, prec) in hdrs.items():
            print(f'mx_hdrs += {h}')
            print(f'MX_HGEN_{h} := {gen}')
            print(f'MX_HPREC_{h} := {prec}')
        for b, gen in data.items():
            print(f'mx_data_sets += {b}')
            print(f'MX_GEN_{b} := {gen}')
            print(f'MX_BPREC_{b} := {" ".join(sorted({t.params for t in tests if t.blob == b and t.params}))}')


if __name__ == '__main__':
    main()
