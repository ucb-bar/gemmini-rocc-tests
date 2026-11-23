"""Per mesh row (exp bits, frac bits) of the PE product and accumulator, for the MX goldens.

Source: a config's gemmini_params.h (MESH_PROD_PRECISION / MESH_ACC_PRECISION, written by
GemminiConfigs.generateHeader). Without --params, the ConfigsFP defaults (prodPrecision / accPrecision, or all
BF16 for the bf16Arith configs).
"""
import re

BF16 = (8, 7)
_loaded = {}   # dim -> (prod, acc) from --params headers


def mesh_precision(params_h):
    s = open(params_h).read()

    def get(name):
        m = re.search(r'#define %s (\{.*\})' % name, s)
        if not m:
            raise SystemExit(f'{params_h}: no {name} (elaborate the config with the current generator)')
        return [(int(e), int(f)) for e, f in re.findall(r'\{(\d+),(\d+)\}', m.group(1))]

    prod, acc = get('MESH_PROD_PRECISION'), get('MESH_ACC_PRECISION')
    dim = int(re.search(r'#define DIM (\d+)', s).group(1))
    assert len(prod) == len(acc) == dim, params_h
    return prod, acc


def default(dim, bf16_arith=False):
    if bf16_arith:
        return [BF16] * dim, [BF16] * dim
    if dim <= 8:
        return [(4, 3)] * dim, [BF16] * dim
    return [(4, 3)] * dim, [(4, 4)] * 8 + [(4, 5)] * 2 + [(4, 6)] * 5 + [BF16] * (dim - 15)


def add_arg(ap, many=False):
    ap.add_argument('--params', nargs='*' if many else None, default=None,
                    help="config's gemmini_params.h (per-row product/accumulator precision)")


def load(params):
    for p in ([params] if isinstance(params, str) else params or []):
        prod, acc = mesh_precision(p)
        _loaded[len(prod)] = (prod, acc)


def for_dim(dim, bf16_arith=False):
    if _loaded:
        if dim not in _loaded:
            raise SystemExit(f'--params has no DIM {dim} header (have {sorted(_loaded)})')
        return _loaded[dim]
    return default(dim, bf16_arith)
