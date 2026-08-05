// Timing of cryptographic primitives (BN254) using the MCL library.
// Reports average time per operation in milliseconds, matching the
// notation of TABLE A (T_h, T_a, T_m, T_e, T_in, T_mn, T_p, T_sa).
//
// Build (MinGW g++):
//   g++ -O3 -std=c++14 -fPIC -DNDEBUG -DMCL_USE_LLVM=0 -DMCL_MSM=0 \
//       -I include -I src bench_apeg.cpp src/fp.cpp src/asm/bint-x64-mingw.S \
//       -o bench_apeg.exe
#include <mcl/bn.hpp>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <string>
#include <functional>

using namespace mcl::bn;
using Clock = std::chrono::high_resolution_clock;

static volatile uint64_t g_sink = 0;

template<class T>
static uint64_t sinkOf(const T &v) {
    uint8_t buf[768];
    size_t n = v.serialize(buf, sizeof(buf));
    uint64_t s = 0;
    for (size_t i = 0; i < n; i++) s += buf[i];
    return s;
}

static double ms_per_op(const char *name, int iters, const std::function<void(int)> &body) {
    auto t0 = Clock::now();
    body(iters);
    auto t1 = Clock::now();
    double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    double ms = (us / iters) / 1000.0;
    printf("  %-26s %12.6f ms   (%d iters)\n", name, ms, iters);
    return ms;
}

int main(int argc, char **argv) {
    std::string curve = (argc > 1) ? argv[1] : "bn254";
    const mcl::CurveParam *cvp = &mcl::BN254;
    const char *cvName = "BN254";
    if (curve == "bls12_381") { cvp = &mcl::BLS12_381; cvName = "BLS12-381"; }
    else if (curve == "bn462") { cvp = &mcl::BN462; cvName = "BN462"; }
    else if (curve == "bn381") { cvp = &mcl::BN381_1; cvName = "BN381"; }
    initPairing(*cvp);
    printf("  curve = %s\n", cvName);

    Fp a, b, c;
    a.setByCSPRNG();
    b.setByCSPRNG();
    c.setByCSPRNG();

    G1 P, Q, R;
    G2 U, V;
    hashAndMapToG1(P, "apeg-point-1", 12);
    hashAndMapToG1(Q, "apeg-point-2", 12);
    hashAndMapToG2(U, "apeg-point-3", 12);
    hashAndMapToG2(V, "apeg-point-4", 12);
    Fr s;
    s.setByCSPRNG();
    Fp12 e;
    std::string msg(32, 'x');

    printf("=====================================================\n");
    printf("  MCL BN254 cryptographic operation timings\n");
    printf("=====================================================\n");

    // warm-up
    for (int i = 0; i < 1000; i++) { Fp::mul(c, c, b); }
    g_sink += sinkOf(c);

    double t_sa = ms_per_op("T_sa  modular addition", 5000000, [&](int n){
        for (int i = 0; i < n; i++) { Fp::add(c, c, b); }
        g_sink += sinkOf(c);
    });
    double t_mn = ms_per_op("T_mn  modular multiply", 3000000, [&](int n){
        for (int i = 0; i < n; i++) { Fp::mul(c, c, b); }
        g_sink += sinkOf(c);
    });
    double t_in = ms_per_op("T_in  modular inversion", 200000, [&](int n){
        for (int i = 0; i < n; i++) { Fp::inv(c, a); a += b; }
        g_sink += sinkOf(c);
    });
    double t_e = ms_per_op("T_e   modular exponent", 30000, [&](int n){
        for (int i = 0; i < n; i++) { Fp::pow(c, a, b); }
        g_sink += sinkOf(c);
    });
    double t_h = ms_per_op("T_h   one-way hash", 500000, [&](int n){
        for (int i = 0; i < n; i++) { a.setHashOf(msg.data(), msg.size()); }
        g_sink += sinkOf(a);
    });
    double t_a = ms_per_op("T_a   EC point addition", 1000000, [&](int n){
        for (int i = 0; i < n; i++) { G1::add(R, R, P); }
        g_sink += sinkOf(R);
    });
    double t_m = ms_per_op("T_m   EC point multiply", 5000, [&](int n){
        for (int i = 0; i < n; i++) { G1::mul(R, P, s); s += 1; }
        g_sink += sinkOf(R);
    });
    double t_p = ms_per_op("T_p   bilinear pairing", 1000, [&](int n){
        for (int i = 0; i < n; i++) { pairing(e, P, U); }
        g_sink += sinkOf(e);
    });

    printf("-----------------------------------------------------\n");
    printf("  summary (ms):\n");
    printf("    T_h  (hash)      = %.4f\n", t_h);
    printf("    T_a  (EC add)    = %.4f\n", t_a);
    printf("    T_m  (EC mul)    = %.4f\n", t_m);
    printf("    T_e  (mod exp)   = %.4f\n", t_e);
    printf("    T_in (mod inv)   = %.4f\n", t_in);
    printf("    T_mn (mod mul)   = %.4f\n", t_mn);
    printf("    T_p  (pairing)   = %.4f\n", t_p);
    printf("    T_sa (mod add)   = %.6f\n", t_sa);
    printf("=====================================================\n");
    (void)g_sink;
    return 0;
}
