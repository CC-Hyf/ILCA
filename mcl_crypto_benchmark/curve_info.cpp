// Print BN254 field/group parameters from MCL itself (authoritative).
#include <mcl/bn.hpp>
#include <cstdio>
#include <cstring>

using namespace mcl::bn;

static void showMod(const char *name, const char *dec) {
    // count decimal digits and approximate bit length
    size_t digits = strlen(dec);
    printf("  %-6s modulus (dec, %zu digits):\n    %s\n", name, digits, dec);
}

int main() {
    initPairing(mcl::BN254);

    char buf[256];
    printf("=== MCL BN254 parameters ===\n");

    size_t n1 = Fp::getModulo(buf, sizeof(buf));
    (void)n1;
    showMod("Fp(p)", buf);

    size_t n2 = Fr::getModulo(buf, sizeof(buf));
    (void)n2;
    showMod("Fr(r)", buf);

    // bit sizes via serialization byte length
    Fp a; a.setByCSPRNG();
    Fr s; s.setByCSPRNG();
    G1 P; hashAndMapToG1(P, "x", 1);
    G2 Q; hashAndMapToG2(Q, "y", 1);
    Fp12 e; pairing(e, P, Q);

    uint8_t b[1024];
    printf("--- serialized element sizes (bytes) ---\n");
    printf("  Fp   element : %zu\n", a.serialize(b, sizeof(b)));
    printf("  Fr   scalar  : %zu\n", s.serialize(b, sizeof(b)));
    printf("  G1   point   : %zu\n", P.serialize(b, sizeof(b)));
    printf("  G2   point   : %zu\n", Q.serialize(b, sizeof(b)));
    printf("  GT/Fp12      : %zu\n", e.serialize(b, sizeof(b)));
    return 0;
}
