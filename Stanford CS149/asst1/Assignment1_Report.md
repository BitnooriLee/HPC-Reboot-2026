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

### Task 1: Implement `clampedExpVector`

`clampedExpVector` is a vectorized implementation of `clampedExpSerial` using CS149 fake SIMD intrinsics.

**Implementation strategy:**

1. **Process the array in chunks of VECTOR_WIDTH** — For the last chunk where the remaining elements are fewer than VECTOR_WIDTH, use a partial mask via `_cs149_init_ones(remaining)` to activate only the valid lanes.
2. **Handle `exponent == 0`** — Initialize `result = 1.0` for all valid lanes, then use `_cs149_veq_int` to identify zero-exponent lanes and exclude them from the multiplication loop.
3. **Accumulation loop** — Use `_cs149_cntbits(maskCountGtZero) > 0` as the loop condition, continuing as long as at least one lane still has `count > 0`. This is necessary because each lane has a different exponent and thus exits the loop at a different iteration.
4. **Clamp** — Use `_cs149_vgt_float` to identify lanes exceeding 9.999999 and overwrite them.

**Verification results:**

| Input size | Result | Vector Utilization |
|---|---|---|
| N=16 (default) | Passed | 85.9% |
| N=3 (non-multiple) | Passed | 72.9% |
| N=17 (non-multiple) | Passed | 78.5% |

### Task 2: Vector Utilization vs. VECTOR_WIDTH

Measured with `./myexp -s 10000` (N=10,000 elements), sweeping VECTOR_WIDTH from 2 to 16:

| VECTOR_WIDTH | Total Vector Instructions | Vector Utilization |
|:---:|---:|---:|
| 2 | 168,027 | 86.8% |
| 4 | 97,059 | 81.2% |
| 8 | 52,785 | 78.4% |
| 16 | 27,576 | 77.1% |

**Analysis:**

Vector utilization **decreases** as VECTOR_WIDTH increases.

The root cause lies in the inner while loop of `clampedExpVector`. Each lane holds a different exponent value, so lanes exit the loop at different iterations. As VECTOR_WIDTH grows, a single chunk is more likely to contain one lane with a large exponent that forces all other already-finished lanes to remain idle until it completes. On the other hand, Total Vector Instructions drops by roughly half each time VECTOR_WIDTH doubles — this trade-off between utilization and instruction count is a key consideration when selecting vector width on real hardware.

### Extra Credit: `arraySumVector`

Implemented with `O(N/VECTOR_WIDTH + log2(VECTOR_WIDTH))` complexity instead of the serial `O(N)`.

**Phase 1 — chunk accumulation** `O(N/VECTOR_WIDTH)`: Iterate over the array in steps of VECTOR_WIDTH, adding each chunk into a vector accumulator.

**Phase 2 — in-vector reduction** `O(log2(VECTOR_WIDTH))`: Apply `hadd` + `interleave` for log2(VECTOR_WIDTH) iterations.

```
sum = [a, b, c, d]
hadd       → [a+b, a+b, c+d, c+d]
interleave → [a+b, c+d, a+b, c+d]
hadd       → [a+b+c+d, ...]
result = sum.value[0]
```

Passed for both N=16 and N=10,000.

---

## Program 3: Parallel Fractal Generation Using ISPC

**Platform note:** The assignment targets AVX2 8-wide on myth machines. Results here use NEON 4-wide on Apple Silicon (theoretical max 4x instead of 8x).

### Part 1: ISPC SIMD Speedup

| View | Serial (ms) | ISPC (ms) | Speedup |
|---|---|---|---|
| View 1 | 155.6 | 84.4 | **1.84x** |
| View 2 | — | 53.4 | **1.55x** |

**Maximum expected speedup:** NEON 4-wide → theoretical max **4x** (myth machine AVX2 8-wide → 8x).

**Why actual speedup is well below the theoretical maximum:**

The Mandelbrot computation is non-uniform — pixels in the interior of the set require the maximum number of iterations, while pixels far from the boundary diverge almost immediately. When ISPC processes a gang of 4 program instances (one SIMD vector), all 4 must wait for the slowest element to finish before the gang can retire. This is the same lane utilization problem observed in Program 2: the more divergent the workload, the more idle lanes accumulate.

**Why View 2 shows lower speedup than View 1:**

View 2 zooms into the boundary of the Mandelbrot set, where nearly every pixel requires close to the maximum iteration count. The computation is more uniform in total work but the boundary itself produces a high density of pixels that diverge at different rates, worsening SIMD lane utilization compared to View 1.

### Part 2: ISPC Tasks (Multi-core)

**Baseline (2 tasks):**

| Version | Time (ms) | Speedup over serial |
|---|---|---|
| Serial | 155.6 | 1.00x |
| ISPC (no tasks) | 84.4 | 1.84x |
| ISPC with 2 tasks | — | 3.57x |

**Task count sweep (View 1):**

| Tasks | Speedup | Notes |
|:---:|:---:|---|
| 2 | 3.57x | default |
| 32 | **10.79x** | optimal |
| 80 | 10.74x | no further gain |

**Optimal task count: 32**

The machine has 8 cores (4 P-cores + 4 E-cores). Setting tasks = 32 (4× the core count) gives the best speedup by ensuring each core receives multiple tasks. This distributes the load imbalance caused by non-uniform per-row computation costs across all cores — similar to how round-robin interleaving helped in Program 1. Beyond 32 tasks, task scheduling overhead offsets any further load-balancing benefit, so speedup plateaus.

On myth machines (AVX2 8-wide + 8 symmetric cores), the same approach with 32+ tasks is expected to exceed 32x by combining 8x SIMD width with 8x multi-core parallelism minus overhead.

### Extra Credit: Thread Abstraction vs. ISPC Task Abstraction

**Launching 10,000 threads (pthread):**
Each `pthread_create()` asks the OS to create a real execution unit — allocating a stack (typically several MB), registering with the OS scheduler, and requiring context switches to run on the available cores. Launching 10,000 threads would consume tens of gigabytes of memory, generate massive scheduling overhead, and likely crash or hang the system.

**Launching 10,000 ISPC tasks:**
ISPC tasks are backed by a thread pool (`tasksys.cpp` maintains a fixed set of worker threads, one per core). A `launch` call simply enqueues a task descriptor into a work queue — it does not create a new OS thread. The fixed pool of worker threads dequeues and executes tasks one by one. Launching 10,000 tasks costs only the memory for 10,000 lightweight task descriptors; the actual concurrency is always bounded by the number of cores.

**Key semantic differences:**
- `pthread_create/join`: programmer controls thread lifetime explicitly; each thread is an OS-level resource.
- `launch/sync`: tasks are logical units of work managed by a runtime scheduler; the programmer expresses *what* to compute, not *how many* OS threads to use.
- ISPC tasks are cheap to create in large numbers, enabling fine-grained load balancing without OS overhead. Threads are expensive and should be created sparingly.

---

**Platform note:** Results use NEON 4-wide on Apple Silicon. The assignment targets AVX2 8-wide on myth machines.

### Task 1: SIMD and Multi-core Speedup

Baseline with random input values in range [0.001, 2.999]:

| Version | Time (ms) | Speedup |
|---|---|---|
| Serial | 722.5 | 1.00x |
| ISPC (SIMD only) | 269.9 | **2.68x** |
| ISPC with tasks | 44.6 | **16.21x** |

- **SIMD speedup (2.68x):** NEON 4-wide theoretical max is 4x. The gap is due to non-uniform convergence — each value requires a different number of Newton iterations, causing idle lanes within each SIMD gang.
- **Multi-core speedup:** Task ISPC (16.21x) ÷ ISPC no-tasks (2.68x) ≈ **6x** from parallelism across 8 cores (capped by asymmetric P/E-core architecture).

### Task 2: Input That Maximizes Speedup

Setting all values to `1.0f` (initial guess = 1.0, so `sqrt(1.0) = 1.0` converges in 0 iterations):

```cpp
values[i] = 1.0f;
```

| Version | Time (ms) | Speedup |
|---|---|---|
| Serial | 7.7 | 1.00x |
| ISPC (SIMD only) | 2.2 | **3.49x** |
| ISPC with tasks | 1.9 | **4.12x** |

SIMD speedup improved from 2.68x → **3.49x** (closer to the 4x theoretical max) because all 4 lanes finish in the same number of iterations — zero lane divergence. Multi-core speedup dropped significantly (16x → 4x) because the workload is now so small that task scheduling overhead dominates over computation time.

### Task 3: Input That Minimizes Speedup

Setting 1 in every 4 elements to `2.999f` (slowest to converge) and the rest to `1.0f` (instant):

```cpp
values[i] = (i % 4 == 0) ? 2.999f : 1.0f;
```

| Version | Time (ms) | Speedup |
|---|---|---|
| Serial | 252.6 | 1.00x |
| ISPC (SIMD only) | 371.4 | **0.68x** |
| ISPC with tasks | 72.1 | **3.50x** |

ISPC is **slower than serial** (0.68x). Each SIMD gang of 4 has 3 lanes that finish immediately (1.0) and 1 lane that requires many iterations (2.999). The 3 fast lanes sit idle while waiting for the slow lane, giving only 25% SIMD utilization. The overhead of SIMD execution on top of 25% utilization pushes performance below serial.

---

## Program 5: BLAS `saxpy`

*[ To be completed ]*

### Task 1: ISPC Speedup and Analysis

---

## Program 6: Making K-Means Faster

*[ To be completed ]*

### Profiling: Identifying the Bottleneck

### Optimization: Parallelization Approach and Speedup
