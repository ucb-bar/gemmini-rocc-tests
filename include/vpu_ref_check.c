// Host tool: compare a TestVpu VPU_DUMP file (RTL) against vpu_ref.h bit-for-bit.
//   gcc -O2 -I. vpu_ref_check.c -o vpu_ref_check && ./vpu_ref_check vpu_dump.txt
#include <stdio.h>
#include "vpu_ref.h"

int main(int argc, char **argv) {
  FILE *f = fopen(argc > 1 ? argv[1] : "vpu_dump.txt", "r");
  if (!f) { perror("open"); return 2; }
  int op; unsigned x, got; long n = 0, bad = 0;
  while (fscanf(f, "%d %x %x", &op, &x, &got) == 3) {
    uint16_t exp = op == VPU_EXP ? vpu_exp(x) : op == VPU_RCP ? vpu_rcp(x) : vpu_rsqrt(x);
    n++;
    if (exp != got && bad++ < 20) printf("op %d x=%04x rtl=%04x ref=%04x\n", op, x, got, exp);
  }
  printf("%ld values, %ld mismatches -> %s\n", n, bad, bad ? "FAIL" : "PASS");
  return bad != 0;
}
