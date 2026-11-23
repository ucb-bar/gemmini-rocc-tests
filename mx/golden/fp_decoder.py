import math

def log2_up(n: int) -> int:
    """Ceil(log2(n)), for n >= 1."""
    if n <= 1:
        return 0
    return (n - 1).bit_length()

def bitslice(x: int, hi: int, lo: int) -> int:
    """Extract bits [hi:lo] (inclusive), LSB = bit 0."""
    width = hi - lo + 1
    return (x >> lo) & ((1 << width) - 1)

def recfn_to_ieee(rec: int, exp_bits: int, mant_bits_total: int) -> int:
    """
    Convert a recoded float (recFN) to an IEEE-like encoding.

    Parameters
    ----------
    rec : int
        recFN bits as integer.
    exp_bits : int
        Number of exponent bits (e.g. 8 for BF16).
    mant_bits_total : int
        TOTAL 'mantissa' bits in your convention = sign + fraction.
        (For BF16: 1 sign + 7 fraction = 8.)

    Internally this is mapped to HardFloat's `sigWidth = mant_bits_total`.
    """

    expWidth = exp_bits
    sigWidth = mant_bits_total  # HardFloat-style sigWidth

    # recFN width: 1(sign) + (expWidth+1) + (sigWidth-1)
    recWidth  = 1 + expWidth + sigWidth
    ieeeWidth = 1 + expWidth + (sigWidth - 1)  # sign + exp + frac

    # ---------- rawFloatFromRecFN ----------
    # sign bit is MSB of rec
    sign = bitslice(rec, recWidth - 1, recWidth - 1)

    # exponent field: in(expWidth + sigWidth - 1, sigWidth - 1)  => expWidth+1 bits
    exp_hi = sigWidth - 1 + expWidth
    exp_lo = sigWidth - 1
    exp = bitslice(rec, exp_hi, exp_lo)  # width expWidth+1

    # Top bits of exp (indexing like Scala: exp(expWidth, expWidth-2), etc.)
    top3       = bitslice(exp, expWidth,     expWidth - 2)  # 3 bits
    top2       = bitslice(exp, expWidth,     expWidth - 1)  # 2 bits
    bit_exp_m2 = bitslice(exp, expWidth - 2, expWidth - 2)  # single bit

    isZero    = (top3 == 0)
    isSpecial = (top2 == 0b11)
    isNaN     = isSpecial and (bit_exp_m2 == 1)
    isInf     = isSpecial and (bit_exp_m2 == 0)

    # sExp is just exp interpreted as integer
    sExp = exp

    # mantissa bits inside recFN input: in(sigWidth - 2, 0)  (sigWidth-1 bits)
    frac_rec = bitslice(rec, sigWidth - 2, 0)

    # sig = 0.U(1.W) ## !isZero ## in(sigWidth-2, 0)
    # => width = sigWidth + 1
    sig = ((0 << sigWidth) |
           ((0 if isZero else 1) << (sigWidth - 1)) |
           frac_rec)

    # ---------- fNFromRecFN ----------
    # minNormExp = (1 << (expWidth - 1)) + 2
    minNormExp = (1 << (expWidth - 1)) + 2

    isSubnormal = (sExp < minNormExp)

    # denormShiftDist = 1.U - rawIn.sExp(log2Up(sigWidth - 1) - 1, 0)
    if sigWidth - 1 > 0:
        k = log2_up(sigWidth - 1)
        if k > 0:
            sExp_low_for_denorm = sExp & ((1 << k) - 1)
            # UInt subtraction (wrap around)
            denormShiftDist = (1 - sExp_low_for_denorm) & ((1 << k) - 1)
        else:
            denormShiftDist = 0
    else:
        denormShiftDist = 0

    # denormFract = ((rawIn.sig >> 1) >> denormShiftDist)(sigWidth-2, 0)
    sig_shifted = sig >> 1
    sig_shifted >>= denormShiftDist
    denormFract = bitslice(sig_shifted, sigWidth - 2, 0)

    # expOut = Mux(isSubnormal, 0, rawIn.sExp(expWidth-1,0) - ((1<<(expWidth-1))+1))
    sExp_low = sExp & ((1 << expWidth) - 1)
    bias_term = (1 << (expWidth - 1)) + 1

    if isSubnormal:
        expOut = 0
    else:
        expOut = (sExp_low - bias_term) & ((1 << expWidth) - 1)

    # | Fill(expWidth, rawIn.isNaN || rawIn.isInf)
    if isNaN or isInf:
        expOut |= (1 << expWidth) - 1  # all ones if NaN/Inf

    # fractOut
    if isSubnormal:
        fractOut = denormFract
    else:
        if isInf:
            fractOut = 0
        else:
            fractOut = bitslice(sig, sigWidth - 2, 0)

    # Final IEEE-style encoding: sign | expOut | fractOut
    ieee = (sign << (expWidth + (sigWidth - 1))) \
           | (expOut << (sigWidth - 1)) \
           | fractOut

    ieee &= (1 << ieeeWidth) - 1
    return ieee

def ieee_to_float(ieee: int, exp_bits: int, mant_bits_total: int) -> float:
    """
    Interpret a generic IEEE-like value (sign + exponent + fraction)
    as a Python float.

    Parameters
    ----------
    ieee : int
        Encoded bits.
    exp_bits : int
        Exponent bits.
    mant_bits_total : int
        TOTAL mantissa bits in your convention = sign + fraction.
        Fraction bits = mant_bits_total - 1.
    """
    fracWidth = mant_bits_total - 1
    expWidth  = exp_bits

    sign = (ieee >> (expWidth + fracWidth)) & 0x1
    exp  = (ieee >> fracWidth) & ((1 << expWidth) - 1)
    frac = ieee & ((1 << fracWidth) - 1)

    bias = (1 << (expWidth - 1)) - 1

    if exp == 0:
        if frac == 0:
            # Zero
            return -0.0 if sign else 0.0
        # Subnormal
        e = 1 - bias
        m = frac / (1 << fracWidth)
        return ((-1.0)**sign) * (2.0**e) * m

    if exp == (1 << expWidth) - 1:
        if frac == 0:
            # Infinity
            return float('-inf') if sign else float('inf')
        # NaN
        return float('nan')

    # Normalized
    e = exp - bias
    m = 1.0 + frac / (1 << fracWidth)
    return ((-1.0)**sign) * (2.0**e) * m

def unpack_recfn_hex(hex_str: str, num_vals: int, exp_bits: int, mant_bits_total: int):
    """
    Given a packed word of recFN values in hex, MSB-first, print:
      - recFN bits (hex)
      - IEEE-style BF16 bits (hex)
      - decimal BF16 value (as float)

    Parameters
    ----------
    hex_str : str
        Hex string with all recFN values packed MSB-first.
    num_vals : int
        Number of recoded values packed.
    exp_bits : int
        Exponent bits (e.g. 8 for BF16).
    mant_bits_total : int
        TOTAL mantissa bits in your convention = sign + fraction.
        (For BF16: 8 = 1 sign + 7 fraction.)
    """
    expWidth = exp_bits
    sigWidth = mant_bits_total

    recWidth  = 1 + expWidth + sigWidth
    ieeeWidth = 1 + expWidth + (sigWidth - 1)

    word = int(hex_str, 16)
    total_bits = recWidth * num_vals
    word &= (1 << total_bits) - 1  # trim if hex had padding

    ieee_hex_digits = (ieeeWidth + 3) // 4

    results = []
    for i in range(num_vals):
        shift = (num_vals - 1 - i) * recWidth
        rec_val = (word >> shift) & ((1 << recWidth) - 1)

        ieee_val = recfn_to_ieee(rec_val, exp_bits=exp_bits, mant_bits_total=mant_bits_total)
        dec_val  = ieee_to_float(ieee_val, exp_bits=exp_bits, mant_bits_total=mant_bits_total)

        ieee_hex = f"0x{ieee_val:0{ieee_hex_digits}X}"
        results.append((i, rec_val, ieee_val, ieee_hex, dec_val))

    return results

if __name__ == "__main__":
    # Example: 4 BF16 recoded values
    # (recWidth = 1 + 8 + 8 = 17 bits -> 4×17 = 68 bits -> 17 hex digits)
    #
    # Put your 68-bit word here:
    hex_str = "3e6b"  # <-- replace with your packed recFN word

    num_vals        = 1
    exp_bits        = 8   # BF16 exponent
    mant_bits_total = 8   # YOUR convention: 1 sign + 7 fraction

    vals = unpack_recfn_hex(hex_str, num_vals, exp_bits, mant_bits_total)

    for idx, rec_bits, ieee_bits, ieee_hex, dec_bf16 in vals:
        print(f"Value {idx}:")
        print(f"  recFN bits (hex):   0x{rec_bits:X}")
        print(f"  IEEE BF16 (hex):    {ieee_hex}")
        print(f"  BF16 decimal value: {dec_bf16!r}")
        print()

    print("IEEE to decimal:")
    print(ieee_to_float(0xb9, exp_bits=4, mant_bits_total=4))
    print(ieee_to_float(0xa6, exp_bits=4, mant_bits_total=4))
    print(ieee_to_float(0x3f80, exp_bits=8, mant_bits_total=8))
    print()
    print("IEEE floats:")
    print(ieee_to_float(0x3e7c, exp_bits=8, mant_bits_total=8))
    print(ieee_to_float(0x3d34, exp_bits=8, mant_bits_total=8))
    print(ieee_to_float(0x3e6b, exp_bits=8, mant_bits_total=8))
    print("Expected:")
    print(ieee_to_float(0x3f6b, exp_bits=8, mant_bits_total=8))
