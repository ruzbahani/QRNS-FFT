"""
QRNS-FFT Implementation
Quadratic Residue Number System for Fast Fourier Transform (QRNS-FFT)

Author: Ali Mohammadi Ruzbahani
Course: ENEL 637 - Arithmetic Techniques with DSP Applications
Instructor: Prof. Vassil Simeonov Dimitrov
Department of Electrical and Software Engineering
University of Calgary, Fall 2025

This module implements a complete Fast Fourier Transform fully in the
Quadratic Residue Number System (QRNS), using exact integer arithmetic
over the quadratic field extension F_p[i] with i^2 = -1 (mod p).

The design follows the theoretical framework presented in the project
report:
- The modulus p is chosen such that p ≡ 1 (mod 4) and N | (p − 1),
  ensuring the existence of:
  (1) an element i with i^2 ≡ −1 (mod p), and
  (2) a primitive N-th root of unity ω in F_p.
- Complex values a + jb are represented as pairs (a, b) in F_p[i].
- All FFT butterflies are computed using modular arithmetic in F_p[i],
  so every intermediate value is exact and there is no accumulation of
  floating-point round-off error.

Main components of this module:

1. Modular arithmetic and prime selection
   - extended_gcd, mod_inv, mod_pow implement core number-theoretic
     operations.
   - is_prime uses the Miller–Rabin test for primality.
   - find_prime searches for primes of the form p = kN + 1 with
     k ≡ 0 (mod 4), so that p ≡ 1 (mod 4) and N | (p − 1).
   - prime_factors and find_primitive_root construct a generator g of
     the multiplicative group F_p^*, from which ω = g^{(p−1)/N} is
     obtained as a primitive N-th root of unity.

2. Quadratic residue / field extension layer
   - legendre_symbol and tonelli_shanks are used to find an element
     i with i^2 ≡ −1 (mod p), which exists because p ≡ 1 (mod 4).
   - FieldExtension implements arithmetic in F_p[i]:
       (a1 + b1 i) · (a2 + b2 i) = (a1 a2 − b1 b2) + (a1 b2 + a2 b1) i
     together with inversion based on the norm a^2 + b^2 (mod p).

3. QRNS-FFT core (class QRNS_FFT)
   - Automatically selects a suitable prime p (or uses a user-specified
     prime).
   - Constructs:
       • a primitive root g,
       • a primitive N-th root of unity ω,
       • the imaginary unit i satisfying i^2 ≡ −1 (mod p).
   - Precomputes twiddle factors ω^k and ω^{−k}.
   - Implements an in-place, iterative Cooley–Tukey radix-2 FFT and
     IFFT in F_p[i] using bit-reversal and butterfly operations.
   - Provides conversion between complex numpy arrays and the QRNS
     representation (pairs of modular integers) so that experiments
     can be run directly on DSP-style signals.

4. Utility and experiment functions
   - compare_fft_accuracy(x, qrns_fft) compares QRNS-FFT against NumPy’s
     FFT and returns error metrics (max, mean, RMS).
   - benchmark_fft_performance(N, num_trials) runs timing experiments
     and reports NumPy FFT time, QRNS-FFT time, and the slowdown factor.

5. Example applications (used in the report’s experimental section)
   - example_exact_convolution():
       • Demonstrates exact integer convolution using FFT-based
         multiplication with QRNS-FFT.
       • For h = [1, 2, 3, 4] and x = [5, 6, 7, 8], using N_fft = 8
         and prime p = 577, the QRNS-FFT-based convolution exactly
         matches NumPy’s convolution:
             [5, 16, 34, 60, 61, 52, 32]
         with maximum error 0.0.
   - example_signal_processing():
       • Uses a 16-point symmetric integer signal in the range [−4, 4].
       • For prime p = 1153, the round-trip test
             x → FFT → IFFT → x
         yields perfect reconstruction with maximum error 0.0,
         while NumPy’s double-precision FFT exhibits a maximum
         reconstruction error of approximately 5.44 × 10⁻¹⁶, due to
         floating-point rounding.
       • This illustrates the main advantage of QRNS-FFT: all
         intermediate values are exact integers modulo p, and no
         floating-point error accumulates during the transform.

6. Performance benchmarks (software, Python-level)
   The main script runs benchmark_fft_performance for N ∈ {64, 128,
   256, 512}. On the test platform used for the report, typical
   results are:

       N = 64,   p = 769:
           NumPy FFT ≈ 0.018 ms
           QRNS-FFT ≈ 0.375 ms
           Slowdown ≈ 20.5×
       N = 128,  p = 7681:
           NumPy FFT ≈ 0.028 ms
           QRNS-FFT ≈ 0.550 ms
           Slowdown ≈ 19.5×
       N = 256,  p = 12289:
           NumPy FFT ≈ 0.027 ms
           QRNS-FFT ≈ 1.389 ms
           Slowdown ≈ 51.7×
       N = 512,  p = 12289:
           NumPy FFT ≈ 0.037 ms
           QRNS-FFT ≈ 3.019 ms
           Slowdown ≈ 82.0×

   Both algorithms exhibit O(N log N) scaling, but QRNS-FFT has a
   larger constant factor due to the cost of modular arithmetic. These
   software timings are consistent with the analysis in the report:
   QRNS-FFT trades runtime for exactness and deterministic, platform-
   independent behaviour, and is particularly suitable for scenarios
   where numerical precision and reproducibility are more important
   than raw speed (e.g., high-precision DSP, cryptographic polynomial
   arithmetic, or hardware realizations on FPGAs/ASICs).

This file is a self-contained reference implementation for the ENEL 637
project and directly underpins the experimental results and figures
reported in the QRNS-FFT term paper.
"""

import numpy as np
from typing import Tuple, List, Optional
import math
from functools import lru_cache
import matplotlib.pyplot as plt

def mod_inv(a: int, m: int) -> int:
    if a < 0:
        a = (a % m + m) % m
    g, x, _ = extended_gcd(a, m)
    if g != 1:
        raise ValueError(f"Modular inverse does not exist for {a} mod {m}")
    return (x % m + m) % m


def extended_gcd(a: int, b: int) -> Tuple[int, int, int]:
    if a == 0:
        return b, 0, 1
    g, x1, y1 = extended_gcd(b % a, a)
    x = y1 - (b // a) * x1
    y = x1
    return g, x, y


def mod_pow(base: int, exp: int, mod: int) -> int:
    result = 1
    base = base % mod
    while exp > 0:
        if exp % 2 == 1:
            result = (result * base) % mod
        exp = exp >> 1
        base = (base * base) % mod
    return result


def is_prime(n: int, k: int = 10) -> bool:
    if n < 2:
        return False
    if n == 2 or n == 3:
        return True
    if n % 2 == 0:
        return False
    r, d = 0, n - 1
    while d % 2 == 0:
        r += 1
        d //= 2
    for _ in range(k):
        a = np.random.randint(2, n - 1)
        x = mod_pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = (x * x) % n
            if x == n - 1:
                break
        else:
            return False
    return True


def find_prime(min_val: int, n: int, form: str = 'kn+1') -> int:
    k = max(1, (min_val - 1 + n - 1) // n)
    if k % 4 != 0:
        k += (4 - k % 4)
    while True:
        candidate = k * n + 1
        if candidate >= min_val and is_prime(candidate):
            return candidate
        k += 4


def prime_factors(n: int) -> List[int]:
    factors = []
    d = 2
    while d * d <= n:
        if n % d == 0:
            factors.append(d)
            while n % d == 0:
                n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors


def find_primitive_root(p: int) -> int:
    if p == 2:
        return 1
    factors = prime_factors(p - 1)
    for g in range(2, p):
        is_primitive = True
        for factor in factors:
            if mod_pow(g, (p - 1) // factor, p) == 1:
                is_primitive = False
                break
        if is_primitive:
            return g
    raise ValueError(f"No primitive root found for p={p}")


def legendre_symbol(a: int, p: int) -> int:
    result = mod_pow(a, (p - 1) // 2, p)
    if result == 0:
        return 0
    elif result == 1:
        return 1
    else:
        return -1


def tonelli_shanks(n: int, p: int) -> Optional[int]:
    if legendre_symbol(n, p) != 1:
        return None
    if p % 4 == 3:
        return mod_pow(n, (p + 1) // 4, p)
    Q, S = p - 1, 0
    while Q % 2 == 0:
        Q //= 2
        S += 1
    z = 2
    while legendre_symbol(z, p) != -1:
        z += 1
    M = S
    c = mod_pow(z, Q, p)
    t = mod_pow(n, Q, p)
    R = mod_pow(n, (Q + 1) // 2, p)
    while True:
        if t == 0:
            return 0
        if t == 1:
            return R
        i = 1
        temp = (t * t) % p
        while temp != 1 and i < M:
            temp = (temp * temp) % p
            i += 1
        if i == M:
            return None
        b = mod_pow(c, 1 << (M - i - 1), p)
        M = i
        c = (b * b) % p
        t = (t * c) % p
        R = (R * b) % p


class FieldExtension:
    def __init__(self, p: int, i_val: int):
        self.p = p
        self.i_val = i_val
        if (i_val * i_val) % p != (p - 1):
            raise ValueError(f"i_val^2 must equal -1 mod p")

    def add(self, z1: Tuple[int, int], z2: Tuple[int, int]) -> Tuple[int, int]:
        a1, b1 = z1
        a2, b2 = z2
        return ((a1 + a2) % self.p, (b1 + b2) % self.p)

    def sub(self, z1: Tuple[int, int], z2: Tuple[int, int]) -> Tuple[int, int]:
        a1, b1 = z1
        a2, b2 = z2
        return ((a1 - a2) % self.p, (b1 - b2) % self.p)

    def mult(self, z1: Tuple[int, int], z2: Tuple[int, int]) -> Tuple[int, int]:
        a1, b1 = z1
        a2, b2 = z2
        real = (a1 * a2 - b1 * b2) % self.p
        imag = (a1 * b2 + a2 * b1) % self.p
        return (real, imag)

    def mult_scalar(self, scalar: int, z: Tuple[int, int]) -> Tuple[int, int]:
        a, b = z
        return ((scalar * a) % self.p, (scalar * b) % self.p)

    def inv(self, z: Tuple[int, int]) -> Tuple[int, int]:
        a, b = z
        if a == 0 and b == 0:
            raise ValueError("Cannot invert zero")
        norm = (a * a + b * b) % self.p
        norm_inv = mod_inv(norm, self.p)
        return ((a * norm_inv) % self.p, (-b * norm_inv) % self.p)

    def div(self, z1: Tuple[int, int], z2: Tuple[int, int]) -> Tuple[int, int]:
        return self.mult(z1, self.inv(z2))

    def pow(self, z: Tuple[int, int], exp: int) -> Tuple[int, int]:
        if exp == 0:
            return (1, 0)
        result = (1, 0)
        base = z
        while exp > 0:
            if exp % 2 == 1:
                result = self.mult(result, base)
            base = self.mult(base, base)
            exp //= 2
        return result


class QRNS_FFT:
    def __init__(self, N: int, max_value: int = None, prime: int = None):
        if N & (N - 1) != 0:
            raise ValueError("N must be a power of 2")
        self.N = N
        if prime is not None:
            self.p = prime
        else:
            if max_value is None:
                max_value = 1000
            min_prime = N * max_value + 1
            self.p = find_prime(min_prime, N)
        print(f"Using prime p = {self.p}")
        g = find_primitive_root(self.p)
        self.omega = mod_pow(g, (self.p - 1) // N, self.p)
        self.omega_inv = mod_inv(self.omega, self.p)
        self.i_val = tonelli_shanks(self.p - 1, self.p)
        if self.i_val is None:
            raise ValueError(f"Could not find i with i^2 = -1 mod {self.p}")
        self.field = FieldExtension(self.p, self.i_val)
        self._precompute_twiddles()
        print(f"Primitive root: g = {g}")
        print(f"Root of unity: ω = {self.omega}")
        print(f"Imaginary unit: i = {self.i_val}")

    def _precompute_twiddles(self):
        self.twiddles = [1]
        for k in range(1, self.N):
            self.twiddles.append((self.twiddles[-1] * self.omega) % self.p)
        self.twiddles_inv = [1]
        for k in range(1, self.N):
            self.twiddles_inv.append(
                (self.twiddles_inv[-1] * self.omega_inv) % self.p
            )

    def _bit_reverse_copy(self, x: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        n = len(x)
        log_n = int(math.log2(n))
        result = [None] * n
        for i in range(n):
            rev = 0
            for j in range(log_n):
                if (i >> j) & 1:
                    rev |= 1 << (log_n - 1 - j)
            result[rev] = x[i]
        return result

    def fft(self, x: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        n = len(x)
        if n == 1:
            return x
        x_reordered = self._bit_reverse_copy(x)
        log_n = int(math.log2(n))
        for stage in range(log_n):
            m = 2 ** (stage + 1)
            m_half = m // 2
            step = n // m
            for k in range(0, n, m):
                for j in range(m_half):
                    twiddle_idx = (j * step) % n
                    omega_k = self.twiddles[twiddle_idx]
                    t_idx = k + j + m_half
                    u_idx = k + j
                    t = self.field.mult_scalar(omega_k, x_reordered[t_idx])
                    u = x_reordered[u_idx]
                    x_reordered[u_idx] = self.field.add(u, t)
                    x_reordered[t_idx] = self.field.sub(u, t)
        return x_reordered

    def ifft(self, X: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        n = len(X)
        if n == 1:
            return X
        X_reordered = self._bit_reverse_copy(X)
        log_n = int(math.log2(n))
        for stage in range(log_n):
            m = 2 ** (stage + 1)
            m_half = m // 2
            step = n // m
            for k in range(0, n, m):
                for j in range(m_half):
                    twiddle_idx = (j * step) % n
                    omega_k = self.twiddles_inv[twiddle_idx]
                    t_idx = k + j + m_half
                    u_idx = k + j
                    t = self.field.mult_scalar(omega_k, X_reordered[t_idx])
                    u = X_reordered[u_idx]
                    X_reordered[u_idx] = self.field.add(u, t)
                    X_reordered[t_idx] = self.field.sub(u, t)
        N_inv = mod_inv(n, self.p)
        x = [self.field.mult_scalar(N_inv, xi) for xi in X_reordered]
        return x

    def to_qrns(self, x_complex: np.ndarray) -> List[Tuple[int, int]]:
        result = []
        for val in x_complex:
            real = int(np.real(val)) % self.p
            imag = int(np.imag(val)) % self.p
            result.append((real, imag))
        return result

    def from_qrns(self, x_qrns: List[Tuple[int, int]]) -> np.ndarray:
        result = np.zeros(len(x_qrns), dtype=complex)
        for i, (real, imag) in enumerate(x_qrns):
            r = real if real <= self.p // 2 else real - self.p
            im = imag if imag <= self.p // 2 else imag - self.p
            result[i] = complex(r, im)
        return result


def compare_fft_accuracy(x: np.ndarray, qrns_fft: QRNS_FFT) -> dict:
    X_numpy = np.fft.fft(x)
    x_qrns = qrns_fft.to_qrns(x)
    X_qrns_internal = qrns_fft.fft(x_qrns)
    X_qrns = qrns_fft.from_qrns(X_qrns_internal)
    abs_error = np.abs(X_qrns - X_numpy)
    rel_error = abs_error / (np.abs(X_numpy) + 1e-10)
    return {
        'max_abs_error': np.max(abs_error),
        'mean_abs_error': np.mean(abs_error),
        'max_rel_error': np.max(rel_error),
        'mean_rel_error': np.mean(rel_error),
        'rms_error': np.sqrt(np.mean(abs_error**2))
    }


def benchmark_fft_performance(N: int, num_trials: int = 10, return_values: bool = False):
    import time
    x = np.random.randn(N) + 1j * np.random.randn(N)
    qrns_fft = QRNS_FFT(N, max_value=10)
    x_qrns = qrns_fft.to_qrns(x)
    start = time.time()
    for _ in range(num_trials):
        _ = np.fft.fft(x)
    numpy_time = (time.time() - start) / num_trials
    start = time.time()
    for _ in range(num_trials):
        _ = qrns_fft.fft(x_qrns)
    qrns_time = (time.time() - start) / num_trials
    print(f"FFT Length: {N}")
    print(f"NumPy FFT: {numpy_time*1000:.3f} ms")
    print(f"QRNS-FFT:  {qrns_time*1000:.3f} ms")
    print(f"Slowdown:  {qrns_time/numpy_time:.1f}x")
    if return_values:
        return numpy_time, qrns_time


def example_exact_convolution():
    print("\n" + "="*70)
    print("Example: Exact Integer Convolution")
    print("="*70)
    h = np.array([1, 2, 3, 4], dtype=complex)
    x = np.array([5, 6, 7, 8], dtype=complex)
    N = len(h) + len(x) - 1
    N_fft = 2 ** int(np.ceil(np.log2(N)))
    h_padded = np.pad(h, (0, N_fft - len(h)))
    x_padded = np.pad(x, (0, N_fft - len(x)))
    max_val = max(np.max(np.abs(h)), np.max(np.abs(x)))
    qrns_fft = QRNS_FFT(N_fft, max_value=int(max_val) * N_fft)
    h_qrns = qrns_fft.to_qrns(h_padded)
    x_qrns = qrns_fft.to_qrns(x_padded)
    H_qrns = qrns_fft.fft(h_qrns)
    X_qrns = qrns_fft.fft(x_qrns)
    Y_qrns = [qrns_fft.field.mult(H_qrns[k], X_qrns[k]) for k in range(N_fft)]
    y_qrns = qrns_fft.ifft(Y_qrns)
    y_result = qrns_fft.from_qrns(y_qrns)[:N]
    y_numpy = np.convolve(h, x)
    print(f"\nInput h: {h}")
    print(f"Input x: {x}")
    print(f"\nQRNS-FFT convolution: {np.real(y_result)}")
    print(f"NumPy convolution:    {np.real(y_numpy)}")
    print(f"\nMaximum error: {np.max(np.abs(y_result - y_numpy))}")


def example_signal_processing():
    print("\n" + "="*70)
    print("Example: Signal Processing with QRNS-FFT")
    print("="*70)
    N = 16
    signal = np.array(
        [1, 2, 3, 4, 3, 2, 1, 0, -1, -2, -3, -4, -3, -2, -1, 0],
        dtype=complex
    )
    max_val = int(np.max(np.abs(signal)))
    qrns_fft = QRNS_FFT(N, max_value=max_val * N)
    print(f"\nSignal length: {N}")
    print(f"Input signal: {signal[:8].real} ...")
    print(f"Signal range: [{int(np.real(np.min(signal)))}, {int(np.real(np.max(signal)))}]")
    print(f"\n--- Round-Trip Transform Test ---")
    print("This tests: x → FFT → IFFT → x")
    x_qrns = qrns_fft.to_qrns(signal)
    X_qrns = qrns_fft.fft(x_qrns)
    x_reconstructed_qrns = qrns_fft.ifft(X_qrns)
    x_reconstructed = qrns_fft.from_qrns(x_reconstructed_qrns)
    reconstruction_error = np.max(np.abs(signal - x_reconstructed))
    print(f"  Original:      {signal[:8].real}")
    print(f"  Reconstructed: {x_reconstructed[:8].real}")
    print(f"  Max error: {reconstruction_error:.2e}")
    if reconstruction_error < 1e-10:
        print(f"  ✓ Perfect reconstruction! QRNS-FFT maintains exact precision.")
    print(f"\n--- QRNS Arithmetic Details ---")
    print(f"Prime modulus p = {qrns_fft.p}")
    print(f"All computations are exact integer arithmetic modulo {qrns_fft.p}")
    print(f"No floating-point errors introduced during FFT computation")
    print(f"\n--- Comparison with Standard FFT ---")
    X_numpy = np.fft.fft(signal)
    x_numpy_reconstructed = np.fft.ifft(X_numpy)
    numpy_error = np.max(np.abs(signal - x_numpy_reconstructed))
    print(f"Standard FFT reconstruction error: {numpy_error:.2e}")
    print(f"QRNS-FFT reconstruction error:     {reconstruction_error:.2e}")
    print(f"\nBoth methods achieve excellent accuracy for this integer signal.")
    print(f"QRNS advantage: All intermediate values are EXACT integers,")
    print(f"no accumulation of floating-point rounding errors.")
    X_qrns_complex = qrns_fft.from_qrns(X_qrns)
    t = np.arange(N)
    plt.figure()
    plt.plot(t, signal.real, "o-", label="Original signal")
    plt.plot(t, x_reconstructed.real, "s--", label="Reconstructed (QRNS-FFT)")
    plt.xlabel("n")
    plt.ylabel("Amplitude")
    plt.title("Time-Domain Signal: Original vs Reconstructed")
    plt.legend()
    plt.grid(True)
    freqs = np.arange(N)
    plt.figure()
    plt.plot(freqs, np.abs(X_numpy), "o-", label="NumPy |X[k]|")
    plt.plot(freqs, np.abs(X_qrns_complex), "s--", label="QRNS-FFT |X[k]|")
    plt.xlabel("k")
    plt.ylabel("|X[k]|")
    plt.title("Magnitude Spectrum: NumPy vs QRNS-FFT")
    plt.legend()
    plt.grid(True)
if __name__ == "__main__":
    print("QRNS-FFT Implementation")
    print("Ali Mohammadi Ruzbahani - ENEL 637")
    print("University of Calgary, Fall 2025\n")
    example_exact_convolution()
    example_signal_processing()
    print("\n" + "="*70)
    print("Performance Benchmarks")
    print("="*70 + "\n")
    Ns = [64, 128, 256, 512]
    numpy_ms = []
    qrns_ms = []
    slowdowns = []
    for N in Ns:
        numpy_time, qrns_time = benchmark_fft_performance(N, num_trials=5, return_values=True)
        numpy_ms.append(numpy_time * 1000.0)
        qrns_ms.append(qrns_time * 1000.0)
        slowdowns.append(qrns_time / numpy_time)
        print()
    plt.figure()
    plt.plot(Ns, numpy_ms, "o-", label="NumPy FFT")
    plt.plot(Ns, qrns_ms, "s--", label="QRNS-FFT")
    plt.xlabel("FFT length N")
    plt.ylabel("Time (ms)")
    plt.title("Execution Time vs FFT Length")
    plt.xscale("log")
    plt.legend()
    plt.grid(True)
    plt.figure()
    plt.plot(Ns, slowdowns, "o-")
    plt.xlabel("FFT length N")
    plt.ylabel("Slowdown factor (QRNS / NumPy)")
    plt.title("Slowdown vs FFT Length")
    plt.xscale("log")
    plt.grid(True)
    plt.show()