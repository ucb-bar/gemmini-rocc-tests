#include <iostream>
#include <stdlib.h>
#include <math.h>

typedef struct {
  int q;
  float S;
} quantized_t;

quantized_t ipoly(int q, float S, float a, float b, float c) {
  const int qb = b / S;
  const int qc = c / (a*S*S);

  const int q_out = (q + qb)*(q + qb) + qc;
  const float S_out = a*S*S;

  quantized_t result;
  result.q=q_out;
  result.S=S_out;
  return result;
}

quantized_t ierf(const int q, const float S) {
  constexpr float a = -0.2888;
  constexpr float b = -1.769;
  constexpr float c = 1.0;

  const int q_sgn = q < 0 ? -1 : 1;
  const int q_clipped = abs(q) > -b/S ? -b/S : abs(q);

  const quantized_t qS_L = ipoly(q_clipped, S, a, b, c);

  quantized_t result;
  result.q = q_sgn * qS_L.q;
  result.S = qS_L.S;
  return result;
}

quantized_t igelu(int q, float S) {
  // Algorithm 2 in the I-BERT paper
  quantized_t qS_erf  = ierf(q, S/sqrt(2.0));
  int q1 = 1.0 / qS_erf.S;

  quantized_t result;
  result.q = q * (qS_erf.q + q1);
  result.S = S * qS_erf.S / 2;
  return result;
}

int igelu_q_impl(int q, int qb, int qc) {
  const int q_sign = q < 0 ? -1 : 1;
  const int q_clipped = abs(q) > (-qb) ? (-qb) : abs(q);
  const int q_poly = (q_clipped + qb)*(q_clipped + qb) + qc;
  const int q_erf = q_sign * q_poly;
  return q * (q_erf + qc);
}

constexpr float igelu_S_impl(float S) {
  constexpr float sqrt_2 = 1.41421356237;
  return (S * (-0.2888 * (S/sqrt_2)*(S/sqrt_2))) / 2;
}

int main() {
  constexpr float S = 0.008754;

  constexpr float sqrt_2 = 1.41421356237;
  constexpr float S_erf = (-0.2888 * ((S*S)/2));
  constexpr int qb = -1.769 / (S / sqrt_2);
  constexpr int qc = 1.0 / S_erf;

  // std::cout << std::hex << q1 << " " << qb << " " << qc << " " << std::endl;
  // return 0;

  for (int q = -1000000; q < 1000000; q += 13) {
    const quantized_t result = igelu(q, S);
    // std::cout << result.q << " " << result.S << std::endl;

    if (result.S != igelu_S_impl(S)) {
      std::cerr << "wrongS " << result.S << " " << igelu_S_impl(S) << std::endl;
      // return 1;
    }

    if (result.q != igelu_q_impl(q, qb, qc)) {
      std::cerr << "wrongQ " << result.S << " " << igelu_S_impl(S) << std::endl;
      return 1;
    }
  }

  return 0;
}

