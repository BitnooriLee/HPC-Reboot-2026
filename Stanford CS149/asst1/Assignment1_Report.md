# CS149 Assignment 1: Performance Analysis on a Multi-Core CPU

**Platform:** Apple MacBook Air, Apple Silicon M-series (4 P-cores + 4 E-cores)  
**Note:** The assignment targets Stanford myth machines (Intel Core i7, 4 cores / 8 hyper-threads, AVX2). Results on Apple Silicon will differ due to asymmetric core architecture and ARM NEON SIMD (not AVX2).

---

## Program 1: Mandelbrot Threads

### Step 1: 2-Thread Spatial Decomposition

Parallelized Mandelbrot generation using two threads:
- Thread 0 → top half (rows 0 to height/2)
- Thread 1 → bottom half (rows height/2 to height)

| | Time | Speedup |
|---|---|---|
| Serial | 319 ms | 1.00x |
| 2 threads | 163 ms | **1.95x** |

### Step 2: Speedup vs. Thread Count (Block Decomposition, View 1)

Each thread receives a contiguous block of height/N rows.

| Threads | Time (ms) | Speedup |
|---|---|---|
| 1 (serial) | 319 | 1.00x |
| 2 | 165 | 1.94x |
| 3 | 203 | 1.57x ← regression |
| 4 | 138 | 2.32x |
| 5 | 139 | 2.31x |
| 6 | 107 | 2.98x |
| 7 | 104 | 3.06x |
| 8 | 93 | 3.41x |

Speedup is not linear. The 3-thread case is notably worse than 2 threads.

**Hypothesis:** Mandelbrot computation is non-uniform across rows. Center rows (near height/2) require many more iterations to determine convergence, while top/bottom rows diverge quickly. With block decomposition, the thread assigned to the center block becomes a bottleneck — other threads finish early and wait idle. Total runtime is determined by the slowest thread.

### Step 3: Per-Thread Timing (3-Thread Case)

Measured with `CycleTimer::currentSeconds()` at start/end of `workerThreadStart()`:

| Thread | Rows | Time |
|---|---|---|
| Thread 0 | top 1/3 | ~0.070 s |
| Thread 1 | middle 1/3 | **~0.204 s** |
| Thread 2 | bottom 1/3 | ~0.070 s |

Thread 1 takes ~3× longer, confirming the load imbalance hypothesis. The 3-thread result is worse than 2-thread because the 2-thread top/bottom split happens to be more balanced than the 3-thread split that isolates the heavy center rows into a single thread.

### Step 4: Improved Decomposition — Round-Robin Interleaving

**Approach:** Each thread processes rows in a round-robin pattern:
- Thread `i` handles rows `i, i+N, i+2N, ...` (stride = numThreads)

This uniformly distributes cheap and expensive rows across all threads with no synchronization needed.

```cpp
int row = args->threadId;
while (row < (int)args->height) {
    mandelbrotSerial(..., row, 1, ...);
    row += args->numThreads;
}
```

**Results (View 1):**

| Threads | Time (ms) | Speedup |
|---|---|---|
| 1 (serial) | 319 | 1.00x |
| 2 | 167 | 1.93x |
| 3 | 119 | 2.70x |
| 4 | 90 | 3.58x |
| 5 | 80 | 3.99x |
| 6 | 70 | 4.62x |
| 7 | 62 | 5.19x |
| 8 | 58 | **5.53x** |

Speedup is now monotonically increasing and the 3-thread anomaly is eliminated.

**Final 8-thread speedup: 5.53x**

> The assignment targets 7–8x on an Intel myth machine (symmetric cores). On Apple Silicon, speedup caps at ~5.5x due to the asymmetric P-core/E-core architecture — the 4 E-cores assigned to threads 4–7 run significantly slower than P-cores, reducing overall parallel efficiency.

### Step 5: 16-Thread Performance

| Threads | Speedup |
|---|---|
| 8 | 5.53x |
| 16 | 5.33x |

16-thread performance is not greater than 8 threads — it is slightly worse. The machine has only 8 physical cores, so 16 threads require 2 threads per core. This introduces context-switching overhead (register save/restore, cache pressure) with no additional hardware parallelism. Beyond the physical core count, adding threads only adds overhead.

---

## Program 2: Vectorizing Code Using SIMD Intrinsics

*[ To be completed ]*

### Task 1: Implement `clampedExpVector`

### Task 2: Vector Utilization vs. VECTOR_WIDTH

---

## Program 3: Parallel Fractal Generation Using ISPC

*[ To be completed ]*

### Part 1: ISPC SIMD Speedup

### Part 2: ISPC Tasks (Multi-core)

---

## Program 4: Iterative `sqrt`

*[ To be completed ]*

### Task 1: SIMD and Multi-core Speedup

### Task 2: Input That Maximizes Speedup

### Task 3: Input That Minimizes Speedup

---

## Program 5: BLAS `saxpy`

*[ To be completed ]*

### Task 1: ISPC Speedup and Analysis

---

## Program 6: Making K-Means Faster

*[ To be completed ]*

### Profiling: Identifying the Bottleneck

### Optimization: Parallelization Approach and Speedup
