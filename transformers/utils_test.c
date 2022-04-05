#include <stdio.h>
#include "include/gemmini.h"
#include "include/gemmini_nn.h"

// assuming int8 maps [-128, 128] to [-2, 2]

// scaling factor S = 2 / 128

// ERF_A = -0.2888 / 1 * 128 <- scale is [-1, 1]
#define ERF_A -37
// ERF_B = -1.769 / 2 * 128
#define ERF_B -113
// SQRT2_INV = (1 / sqrt(2)) * 128 (shift 7 after multiply)
#define SQRT2_INV 91

void i_erf(int8_t *x, const int len) {
    for (int i = 0; i < len; i++) {
        int8_t q = x[i];
        int8_t q_sign = q > 0 ? 1 : -1; // 1 - (((uint8_t) q >> 7) << 1);
        q = q == -128 ? 127 : q * q_sign; // abs

        // this line is needed ONLY if max(abs(x)) >= 1.769 * 1.414
        // in this case, 2 < 1.769 * 1.414
        // q = q > -ERF_B ? -ERF_B : q; // q = min(-ERF_B, q)

        q = q < (-128 - ERF_B) ? -128 : q + ERF_B; 
        int16_t prod16 = q * q;        

        // note: shifting 7 instead of 6 because ERF_A compensates for 1
        // note: no saturation needed since max(prod16) = 2^14
        int8_t prod8 = prod16 >> 7;
        prod16 = prod8 * (int8_t) ERF_A;

        // shift and saturate product
        prod8 = prod16 <= -(1 << 13) ? -128 : (prod16 >> 6);

        x[i] = q_sign * (prod8 > 63 ? 127 : prod8 + 64);
    }
}

void i_erf2(int8_t *x, const int len, const int frac_bits) {
    int16_t erf_b_16 = -28983;
    int8_t erf_b = frac_bits >= 7 ? -128 : erf_b_16 >> (14 - frac_bits);
    int one = 1 << frac_bits;

    for (int i = 0; i < len; i++) {
        int8_t q = x[i];
        int8_t q_sign = q > 0 ? 1 : -1; // 1 - (((uint8_t) q >> 7) << 1);
        q = q == -128 ? 127 : q * q_sign; // abs

        q = q > -erf_b ? -erf_b : q; // q = min(-ERF_B, q)

        // saturate to -127 because (-128 * -128) >= 2^14
        q = q < (-127 - erf_b) ? -127 : q + erf_b; 
        int16_t prod16 = q * q;

        // note: no saturation needed since max(prod16) = 2^14
        int8_t prod8 = prod16 >> 7;
        prod16 = prod8 * (int8_t) ERF_A;

        // shift and saturate product
        prod8 = prod16 <= -(1 << (7 + frac_bits)) ? -128 : (prod16 >> frac_bits);

        x[i] = q_sign * (prod8 > (127 - one) ? 127 : prod8 + one);
    }
}

void i_gelu(int8_t *x, const int len) {
    for (int i = 0; i < len; i++) {
        int8_t q = x[i];
        int16_t prod16 = (q * (int8_t) SQRT2_INV);
        q = prod16 >> 7;

        int8_t q_sign = q > 0 ? 1 : -1; // 1 - (((uint8_t) q >> 7) << 1);
        q = q == -128 ? 127 : q * q_sign; // abs

        // this line is needed ONLY if max(abs(x)) >= 1.769 * 1.414
        // in this case, 2 < 1.769 * 1.414
        // q = q > -ERF_B ? -ERF_B : q; // q = min(-ERF_B, q)

        q = q < (-128 - ERF_B) ? -128 : q + ERF_B; 
        prod16 = q * q;        

        // note: shifting 7 instead of 6 because ERF_A compensates for 1
        // note: no saturation needed since max(prod16) = 2^14
        int8_t prod8 = prod16 >> 7;
        prod16 = prod8 * (int8_t) ERF_A;

        // shift and saturate product
        prod8 = prod16 <= -(1 << 13) ? -128 : (prod16 >> 6);

        q = q_sign * (prod8 > 63 ? 127 : prod8 + 64);

        q = q > 63 ? 127 : q + 64;
        prod16 = q * x[i];
        x[i] = prod16 >> 7; // divided by 2 included
    }
}

void i_gelu2(int8_t *x, const int len, const int frac_bits) {
    int fb = frac_bits >= 7 ? 6 : frac_bits;
    // if fractional bits >= 7, we need to 

    int16_t erf_b_16 = -28983;
    int8_t erf_b = fb >= 7 ? -128 : erf_b_16 >> (14 - fb);
    int one = 1 << fb;

    for (int i = 0; i < len; i++) {
        int8_t q = x[i];

        q = q >> (frac_bits - fb); // max shift 6

        int16_t prod16 = (q * (int8_t) SQRT2_INV);
        q = prod16 >> 7;

        int8_t q_sign = q > 0 ? 1 : -1; // 1 - (((uint8_t) q >> 7) << 1);
        q = q == -128 ? 127 : q * q_sign; // abs

        q = q > -erf_b ? -erf_b : q; // q = min(-ERF_B, q)

        // saturate to -127 because (-128 * -128) >= 2^14
        q = q < (-127 - erf_b) ? -127 : q + erf_b; 
        prod16 = q * q;

        // note: no saturation needed since max(prod16) = 2^14
        int8_t prod8 = prod16 >> 7;
        prod16 = prod8 * (int8_t) ERF_A;

        // shift and saturate product
        prod8 = prod16 <= -(1 << (7 + fb)) ? -128 : (prod16 >> fb);

        q = q_sign * (prod8 > (127 - one) ? 127 : prod8 + one);

        q = q > (127 - one) ? 127 : q + one;
        prod16 = q * x[i];

        x[i] = prod16 >> (fb >= 7 ? 7 : 1 + fb); // divided by 2 included
    }
    printf("\n");
}

int main() {
    int8_t x[256];
    for (int i = -128; i < 128; i++) x[i + 128] = i;
    i_gelu2(x, 256, 9);

    for (int i = -128; i < 128; i++) {
        printf("%d ", x[i + 128]);
    }
    printf("\n");

    return 0;
}