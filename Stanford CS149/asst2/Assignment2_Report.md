# CS149 Assignment 2: Building a Task Execution Library

**Platform:** Apple MacBook Air, Apple Silicon (arm64, 8 execution contexts)  
**Development environment:** **Local Mac only** — AWS VM not used (see below)

---

## Local Development Instead of AWS

The assignment README and `cloud_readme.md` recommend an AWS `c7g.4xlarge` instance (16-core ARM Graviton3) for performance testing and official grading. **This project was completed on a local Mac instead of AWS** for the following reasons:

| | Official (Stanford) | This project |
|--|---------------------|--------------|
| Machine | AWS `c7g.4xlarge` | MacBook Air, Apple Silicon |
| CPU | 16-core ARM (Graviton3) | 8 cores (4 P + 4 E) |
| OS / reference binary | Linux ARM → `runtasks_ref_linux_arm` | macOS ARM → `runtasks_ref_osx_arm` |
| Thread flag | `-n 16` | `-n 8` |
| AWS student coupons | Provided to enrolled students | Not available (self-study) |

**Why local is sufficient for Part A development:**

- `cloud_readme.md` explicitly says to do **most development locally** and use AWS only for performance tuning.
- asst2 is a **CPU / pthread** assignment — no GPU required. The Mac already provides ARM64 + multiple cores, which is a reasonable stand-in for the grading machine architecture.
- Correctness and relative performance can be validated with the provided **`runtasks_ref_osx_arm`** reference binary on the same hardware.

**Tradeoffs of skipping AWS:**

- Absolute timing numbers will **not match** AWS grading (different core count, OS scheduler, reference binary).
- Performance targets (within 20% of reference) are measured against **`osx_arm` on 8 cores**, not `linux_arm` on 16 cores.
- A final pre-submission run on AWS would still be ideal if submitting to Gradescope; for HPC-Reboot self-study, local results are the primary record.

**Workflow used:**

```bash
cd part_a
make
./runtasks -n 8 simple_test_sync          # correctness
python3 ../tests/run_test_harness.py -n 8 # performance vs osx_arm reference
```

No EC2 instance was created; no AWS billing incurred for Assignment 2.

---

## Progress Summary

| Part | Step | Class | Status |
|------|------|-------|--------|
| A | 1 | `TaskSystemParallelSpawn` | **Done** |
| A | 2 | `TaskSystemParallelThreadPoolSpinning` | Planned (design below) |
| A | 3 | `TaskSystemParallelThreadPoolSleeping` | Not started |
| B | — | `runAsyncWithDeps` / `sync` | Not started |
| — | — | Custom test | Not started |

---

## Environment Setup

- **Development:** Local Mac only — **not AWS** (`part_a/`, `make` + `./runtasks`)
- **Reference binary:** `runtasks_ref_osx_arm` (not `runtasks_ref_linux_arm`)
- **Thread count:** `-n 8` (matches 8 local cores; AWS grading uses `-n 16`)
- **Files modified:** `part_a/tasksys.h`, `part_a/tasksys.cpp` only
- **Makefile:** Unchanged (per assignment requirement)

### Test harness invocation

```bash
cd part_a
make
./runtasks -n 8 simple_test_sync
python3 ../tests/run_test_harness.py -n 8 -t super_super_light super_light
```

### How tests call `run()`

1. `tests/main.cpp` maps test name `simple_test_sync` → function `simpleTestSync`
2. `simpleTestSync` → `simpleTest(t, false)` in `tests/tests.h`
3. `simpleTest` calls `t->run(&first, num_tasks)` and `t->run(&second, num_tasks)` twice
4. `main.cpp` runs all four task system implementations (Serial, Spawn, Spin, Sleep) and prints timing

---

## Part A

### Step 1: `TaskSystemParallelSpawn` — Done

**Goal:** Parallelize bulk task launch by spawning worker threads on every `run()` call. When `run()` returns, all tasks must be complete (synchronous).

**Approach:**

- Store `num_threads_` in constructor (from `-n` flag).
- On `run()`:
  - `num_workers = min(num_threads_, num_total_tasks)` — avoid spawning idle threads when tasks are fewer than workers.
  - If `num_workers <= 1`, fall back to serial loop.
  - Otherwise spawn `num_workers` `std::thread` workers.
  - **Static task assignment (round-robin):** thread `i` runs tasks `i, i + num_workers, i + 2*num_workers, ...`
  - `join()` all workers before returning.

**Why no mutex:** Each `task_id` is handled by exactly one thread; no shared mutable state inside the task system for this step.

**Example** (`num_workers = 4`, `num_total_tasks = 10`):

```
thread 0 → tasks 0, 4, 8
thread 1 → tasks 1, 5, 9
thread 2 → tasks 2, 6
thread 3 → tasks 3, 7
```

**Key code pattern:**

```cpp
auto worker = [&](int thread_id) {
    for (int task_id = thread_id; task_id < num_total_tasks; task_id += num_workers) {
        runnable->runTask(task_id, num_total_tasks);
    }
};
```

#### Results (Mac, `-n 8`, reference: `runtasks_ref_osx_arm`)

**Correctness:** `simple_test_sync` passes for all four implementations.

**Performance harness** (`super_super_light`, `super_light`):

| Test | Implementation | Student (ms) | Reference (ms) | PERF |
|------|----------------|-------------|----------------|------|
| super_super_light | Serial | 3.972 | 3.948 | 1.01 (OK) |
| super_super_light | Parallel + Always Spawn | 23.511 | 22.745 | 1.03 (OK) |
| super_light | Serial | 17.93 | 28.82 | 0.62 (OK) |
| super_light | Parallel + Always Spawn | 29.536 | 30.399 | 0.97 (OK) |

Spawn implementation meets the 20% performance threshold vs. reference on these tests. Thread pool variants (Spin, Sleep) still run serial starter code at this point — they appear fast only because reference pool implementations are slower on light workloads while student code is still serial.

**Limitation (motivation for Step 2):** Every `run()` creates and destroys threads. For tests with many cheap bulk launches (e.g. `ping_pong`, `spin_between_run_calls`), thread creation overhead dominates. Step 2 addresses this with a persistent thread pool.

---

### Step 2: `TaskSystemParallelThreadPoolSpinning` — Planned

**Goal:** Create worker threads once in the constructor; workers spin-wait for work instead of being recreated per `run()`.

**Planned design:**

| Component | Role |
|-----------|------|
| `std::vector<std::thread> workers_` | Pool threads, created in constructor |
| `std::mutex mutex_` | Protect launch state setup/teardown |
| `std::atomic<int> next_task_id_` | Dynamic task assignment (fetch-and-add) |
| `std::atomic<int> tasks_completed_` | Count finished tasks; main spins until `== num_total_tasks` |
| `bool work_available_` / `launch_active_` | Signal workers that a new bulk launch started |
| `bool shutdown_` | Destructor sets true; workers exit loop and join |

**Worker loop (spinning):**

1. Spin while no work and not shutdown.
2. Atomically grab next `task_id`; if `>= num_total_tasks`, continue (other workers may still have tasks).
3. Call `runnable_->runTask(task_id, num_total_tasks)`.
4. Increment `tasks_completed_`; loop back to wait for next launch.

**`run()` (main thread):**

1. Under mutex: set `runnable_`, `num_total_tasks_`, reset counters, set `work_available_ = true`.
2. Spin until `tasks_completed_ == num_total_tasks_` (synchronous return).
3. Under mutex: clear `work_available_`.

**Task assignment:** Dynamic (atomic counter) — better load balance when per-task cost varies.

**Step 3 preview:** Replace spin-waits with `std::condition_variable` so idle workers and the main thread sleep instead of burning CPU.

---

### Step 3: `TaskSystemParallelThreadPoolSleeping` — Not started

*To be filled after implementation.*

---

## Part B — Not started

Implement on top of Step 3 sleeping thread pool:

- `runAsyncWithDeps()` — async bulk launch with dependency vector
- `sync()` — block until all prior async launches complete
- Dependency tracking: waiting tasks vs. ready queue

*Writeup question on dependency tracking: TBD after implementation.*

---

## Writeup Questions (draft / TBD)

### 1. Task system implementation

**Thread management (so far):**

- Step 1: spawn-per-`run()` — no pool.
- Step 2 (planned): fixed pool in constructor, destroyed in destructor.

**Task assignment (so far):**

- Step 1: **static** round-robin by thread id.
- Step 2 (planned): **dynamic** via atomic `next_task_id_`.

**Part B dependencies:** TBD.

### 2. When simpler implementations win

*TBD after Steps 2–3 — compare Serial vs Spawn vs Thread Pool on tests like `super_super_light` (cheap tasks), `mandelbrot_chunked` (heavy tasks), `ping_pong` (many launches).*

### 3. Custom test

*Not yet implemented. Skeleton: `YourTask` / `yourTest()` in `tests/tests.h`.*

---

## Changelog

| Date | Update |
|------|--------|
| 2026-06-12 | Report created. Local Mac workflow documented (AWS skipped). Part A Step 1 implemented and tested. Step 2 design documented. |
