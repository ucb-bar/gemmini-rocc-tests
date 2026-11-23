"""Round values to an fp6 element grid (subnormals kept, saturating). Same arithmetic as
microxcaling's _quantize_elemwise for these formats, so headers match the MXQuant-generated ones."""
import torch

# (exponent bits, mantissa bits incl. sign and implicit 1)
_FORMATS = {'fp6_e3m2': (3, 4), 'fp6_e2m3': (2, 5)}


def quantize_elemwise(A: torch.Tensor, fmt: str, round: str = 'nearest') -> torch.Tensor:
    """round: 'nearest' = half away from zero, 'even' = half to even."""
    ebits, mbits = _FORMATS[fmt]
    emax = 2 ** (ebits - 1)
    max_norm = 2 ** emax * float(2 ** (mbits - 1) - 1) / 2 ** (mbits - 2)
    exp = torch.floor(torch.log2(torch.abs(A) + (A == 0).type(A.dtype))).clip(min=-(2 ** (ebits - 1)) + 2)
    out = A / (2 ** exp) * (2 ** (mbits - 2))
    if round == 'nearest':
        out = torch.sign(out) * torch.floor(torch.abs(out) + 0.5)
    elif round == 'even':
        a = torch.abs(out)
        tie = ((a - 0.5) % 2 == torch.zeros_like(out)).type(out.dtype)
        out = torch.sign(out) * (torch.floor(a + 0.5) - tie)
    else:
        raise ValueError(f'unknown rounding {round}')
    out = out / (2 ** (mbits - 2)) * (2 ** exp)
    out = torch.clamp(out, min=-max_norm, max=max_norm)
    out[A == float('inf')] = float('inf')
    out[A == -float('inf')] = -float('inf')
    return out
