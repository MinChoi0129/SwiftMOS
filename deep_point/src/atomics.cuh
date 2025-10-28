#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

#define THREADS 1024
#define BLOCKS(N) ((int)(((N) + THREADS - 1) / THREADS))

// ------------------------------
// 64-bit CAS helpers (ULL 전용)
// ------------------------------
static inline __device__ unsigned long long cas_ull(unsigned long long* addr,
                                                    unsigned long long cmp,
                                                    unsigned long long val) {
    return atomicCAS(addr, cmp, val);
}

// ------------------------------
// int64 원자 연산 (CAS 기반)
// ------------------------------
static inline __device__ int64_t atomicAdd_ll(int64_t* p, int64_t val) {
    auto* a = reinterpret_cast<unsigned long long*>(p);
    unsigned long long old = *a, assumed;
    do {
        assumed = old;
        int64_t cur = static_cast<int64_t>(assumed);
        int64_t nxt = cur + val;
        old = cas_ull(a, assumed, static_cast<unsigned long long>(nxt));
    } while (assumed != old);
    return static_cast<int64_t>(old);
}

static inline __device__ int64_t atomicMax_ll(int64_t* p, int64_t val) {
    auto* a = reinterpret_cast<unsigned long long*>(p);
    unsigned long long old = *a, assumed;
    do {
        assumed = old;
        int64_t cur = static_cast<int64_t>(assumed);
        if (cur >= val) break;
        old = cas_ull(a, assumed, static_cast<unsigned long long>(val));
    } while (assumed != old);
    return static_cast<int64_t>(old);
}

static inline __device__ int64_t atomicMin_ll(int64_t* p, int64_t val) {
    auto* a = reinterpret_cast<unsigned long long*>(p);
    unsigned long long old = *a, assumed;
    do {
        assumed = old;
        int64_t cur = static_cast<int64_t>(assumed);
        if (cur <= val) break;
        old = cas_ull(a, assumed, static_cast<unsigned long long>(val));
    } while (assumed != old);
    return static_cast<int64_t>(old);
}

// ------------------------------
// double 원자 연산 (CAS 기반)
// ------------------------------
static inline __device__ double atomicAdd_double(double* p, double val) {
    auto* a = reinterpret_cast<unsigned long long*>(p);
    unsigned long long old = *a, assumed;
    do {
        assumed = old;
        double cur = __longlong_as_double(assumed);
        double nxt = cur + val;
        old = cas_ull(a, assumed, __double_as_longlong(nxt));
    } while (assumed != old);
    return __longlong_as_double(old);
}

static inline __device__ double atomicMax_double(double* p, double val) {
    auto* a = reinterpret_cast<unsigned long long*>(p);
    unsigned long long old = *a, assumed;
    do {
        assumed = old;
        double cur = __longlong_as_double(assumed);
        if (cur >= val) break;
        old = cas_ull(a, assumed, __double_as_longlong(val));
    } while (assumed != old);
    return __longlong_as_double(old);
}

static inline __device__ double atomicMin_double(double* p, double val) {
    auto* a = reinterpret_cast<unsigned long long*>(p);
    unsigned long long old = *a, assumed;
    do {
        assumed = old;
        double cur = __longlong_as_double(assumed);
        if (cur <= val) break;
        old = cas_ull(a, assumed, __double_as_longlong(val));
    } while (assumed != old);
    return __longlong_as_double(old);
}

// =====================================================
// Unified atom* API (필요한 타입만 구현: int32/int64/float/double)
// =====================================================

// ---- Add ----
static inline __device__ void atomAdd(int32_t* address, int32_t val) { atomicAdd(address, val); }
static inline __device__ void atomAdd(int64_t* address, int64_t val) { (void)atomicAdd_ll(address, val); }
static inline __device__ void atomAdd(float*   address, float   val) { atomicAdd(address, val); }
static inline __device__ void atomAdd(double*  address, double  val) { (void)atomicAdd_double(address, val); }

// ---- Max ----
static inline __device__ void atomMax(int32_t* address, int32_t val) { atomicMax(address, val); }
static inline __device__ void atomMax(int64_t* address, int64_t val) { (void)atomicMax_ll(address, val); }
static inline __device__ void atomMax(float*   address, float   val) {
    // float용 CAS max
    int* addr = reinterpret_cast<int*>(address);
    int old = *addr, assumed;
    do {
        assumed = old;
        float cur = __int_as_float(assumed);
        if (cur >= val) break;
        old = atomicCAS(addr, assumed, __float_as_int(val));
    } while (assumed != old);
}
static inline __device__ void atomMax(double*  address, double  val) { (void)atomicMax_double(address, val); }

// ---- Min ----
static inline __device__ void atomMin(int32_t* address, int32_t val) { atomicMin(address, val); }
static inline __device__ void atomMin(int64_t* address, int64_t val) { (void)atomicMin_ll(address, val); }
static inline __device__ void atomMin(float*   address, float   val) {
    int* addr = reinterpret_cast<int*>(address);
    int old = *addr, assumed;
    do {
        assumed = old;
        float cur = __int_as_float(assumed);
        if (cur <= val) break;
        old = atomicCAS(addr, assumed, __float_as_int(val));
    } while (assumed != old);
}
static inline __device__ void atomMin(double*  address, double  val) { (void)atomicMin_double(address, val); }
