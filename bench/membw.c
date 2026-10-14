/* STREAM-style memory bandwidth test (our own implementation, not official STREAM).
 *
 *   Linux host: gcc -O3 -march=native -fopenmp membw.c -o membw && ./membw 2048
 *   macOS:      clang -O3 membw.c -o membw && ./membw 2048      (single-thread, no OpenMP)
 *
 * Argument: array size in MiB per array (3 arrays). Reports the best of 10 trials
 * for copy (2 streams) and triad (3 streams), in GB/s (1e9 bytes). Use the triad
 * number as the measured host memory bandwidth.
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    size_t mib = argc > 1 ? strtoul(argv[1], NULL, 10) : 1024;
    size_t n = mib * 1024 * 1024 / sizeof(double);
    double *a = aligned_alloc(64, n * sizeof(double));
    double *b = aligned_alloc(64, n * sizeof(double));
    double *c = aligned_alloc(64, n * sizeof(double));
    if (!a || !b || !c) { fprintf(stderr, "alloc failed\n"); return 1; }
    #pragma omp parallel for
    for (size_t i = 0; i < n; i++) { a[i] = 1.0; b[i] = 2.0; c[i] = 0.0; }  /* first touch */

    double best_copy = 1e30, best_triad = 1e30, s = 3.0;
    for (int t = 0; t < 10; t++) {
        double t0 = now();
        #pragma omp parallel for
        for (size_t i = 0; i < n; i++) c[i] = a[i];
        double t1 = now();
        #pragma omp parallel for
        for (size_t i = 0; i < n; i++) a[i] = b[i] + s * c[i];
        double t2 = now();
        if (t1 - t0 < best_copy) best_copy = t1 - t0;
        if (t2 - t1 < best_triad) best_triad = t2 - t1;
    }
    double bytes = (double)n * sizeof(double);
    int threads = 1;
#ifdef _OPENMP
    threads = omp_get_max_threads();
#endif
    printf("{\"array_mib\": %zu, \"threads\": %d, \"copy_gbs\": %.1f, \"triad_gbs\": %.1f, \"check\": %.1f}\n",
           mib, threads, 2 * bytes / best_copy / 1e9, 3 * bytes / best_triad / 1e9, a[n / 2]);
    free(a); free(b); free(c);
    return 0;
}
