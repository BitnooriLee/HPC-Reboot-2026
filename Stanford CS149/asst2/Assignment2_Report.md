# CS149 Assignment 2: Building a Task Execution Library

**Platform:** Apple MacBook Air, Apple Silicon (arm64, 8 execution contexts)  
**Development environment:** **Local Mac only** — AWS VM not used (see below)

---

## Assignment Overview

Assignment 2 asks you to build a **C++ task execution library** — a small parallel runtime that applications use to launch many tasks efficiently on a multi-core CPU.

**Part A:** Implement synchronous **bulk task launch** via `run(runnable, num_total_tasks)`.

**Part B:** Extend the library with **async task graphs** via `runAsyncWithDeps()` and `sync()`.

The starter code provides four task system classes. You implement three of them (Serial is the baseline). Each step adds complexity and targets better performance on different workloads.

---

## Core Concepts

### Bulk task launch

One call to `run()` launches `num_total_tasks` instances of the same task:

```cpp
t->run(&myTask, 100);  // launches 100 tasks (task_id 0..99)
```

Each task is invoked as:

```cpp
runnable->runTask(task_id, num_total_tasks);
```

The application defines *what* each task does (`IRunnable`). Your task system defines *how* tasks are scheduled across threads.

### Synchronous `run()`

When `run()` returns, **all tasks in that bulk launch must be finished**. The calling thread blocks until completion. This is different from Part B's async API.

### Key files

| File | Role |
|------|------|
| `itasksys.h` | Interface (`IRunnable`, `ITaskSystem`) — **do not modify** |
| `tasksys.h` | Class declarations (your member variables go here) |
| `tasksys.cpp` | Implementations (your main work) |
| `tests/tests.h` | Test definitions — read to understand workloads |
| `tests/main.cpp` | Test driver — maps test names to functions |

`.h` = declarations ("what exists"). `.cpp` = definitions ("how it works"). Only modify `tasksys.h` and `tasksys.cpp`; do not change the `Makefile`.

---

## The Four Task System Variants

These names are **defined by the assignment starter code**, not universal industry terms. Each class implements `name()` with a fixed label used in test output:

| Step | Class | `name()` output | Idea |
|------|-------|-----------------|------|
| (baseline) | `TaskSystemSerial` | `Serial` | One thread, sequential `runTask` loop |
| 1 | `TaskSystemParallelSpawn` | `Parallel + Always Spawn` | New threads on every `run()`, then `join()` |
| 2 | `TaskSystemParallelThreadPoolSpinning` | `Parallel + Thread Pool + Spin` | Persistent thread pool; idle workers **busy-wait** |
| 3 | `TaskSystemParallelThreadPoolSleeping` | `Parallel + Thread Pool + Sleep` | Same pool; idle workers and main thread **sleep** on condition variables |

**Mapping to general terminology:**

- **Serial** → single-threaded execution
- **Spawn / Always Spawn** → per-call thread creation (assignment-specific label)
- **Thread pool** → standard term; workers created once, reused
- **Spin / spinning** → busy-wait loop while checking a flag (standard term)
- **Sleep** → block on `std::condition_variable` until notified (standard term)

Part B builds on **Step 3 (Sleeping)** only — you add `runAsyncWithDeps()` and `sync()` to that class.

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

The README says to run `./runtasks -n 8 simple_test_sync` but does not spell out the call chain. The path is:

```
./runtasks -n 8 simple_test_sync
        │
        ▼
tests/main.cpp
  - matches "simple_test_sync" → simpleTestSync()
  - creates each task system (Serial, Spawn, Spin, Sleep)
  - calls test function with ITaskSystem* t
        │
        ▼
tests/tests.h → simpleTestSync(t) → simpleTest(t, false)
        │
        ▼
t->run(&first, 3);
t->run(&second, 3);    ← two bulk launches, 3 tasks each
```

For debugging, add print statements inside `simpleTest()` in `tests/tests.h`.

### Performance grading

`run_test_harness.py` compares student `runtasks` vs `runtasks_ref_osx_arm`. **PERF** = student time / reference time. Values ≤ 1.0 mean you are at or faster than reference. Part A requires within **20%** of reference (PERF ≤ 1.2) for full performance points, on implementations that pass correctness.

---

## Part A

### Design progression (Steps 1 → 2 → 3)

```
Step 1 Spawn          Step 2 Spin Pool         Step 3 Sleep Pool
─────────────────     ───────────────────      ────────────────────
run() → create N      run() → signal pool      run() → signal pool
        threads               workers                  workers
        join()                  spin-wait                sleep on cv
        return                  spin-wait                sleep on cv
                                return                   return
(per run() overhead)  (no thread create)       (no thread create
                                                 + no CPU spin)
```

| Concern | Spawn | Spin Pool | Sleep Pool |
|---------|-------|-----------|------------|
| Thread creation per `run()` | Yes | No | No |
| Idle worker CPU usage | N/A (threads exit) | High (spin) | Low (blocked) |
| Main thread while waiting | Blocked on `join()` | Spins | Sleeps on cv |
| Task assignment | Static round-robin | Dynamic (planned) | Dynamic (planned) |
| Synchronization | None needed | mutex + atomic | mutex + cv + atomic |

---

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

Spawn meets the 20% threshold on these tests. Spin and Sleep still run serial starter code at this point — their student times look fast only because reference pool code is slower on light workloads while our implementations are still serial.

**Limitation (motivation for Step 2):** Every `run()` creates and destroys threads. Tests like `ping_pong` (400 bulk launches) and `spin_between_run_calls` amplify this overhead. Step 2 removes per-call thread creation.

---

### Step 2: `TaskSystemParallelThreadPoolSpinning` — Planned

**Goal:** Create worker threads once in the constructor; workers spin-wait for work instead of being recreated per `run()`.

#### Architecture

```
Constructor: spawn num_threads_ workers → each enters workerLoop() forever

run() [main thread]:
  1. Publish launch state (runnable, num_total_tasks, reset counters)
  2. Set work_available_ = true
  3. Spin until tasks_completed_ == num_total_tasks_
  4. Clear work_available_, return

workerLoop() [pool threads]:
  1. Spin until work_available_ || shutdown_
  2. Atomic fetch next_task_id_; if invalid, spin until launch ends
  3. runTask(); increment tasks_completed_
  4. Last task done → work_available_ = false

Destructor: shutdown_ = true, wake workers, join all
```

#### Planned `tasksys.h` members

| Member | Role |
|--------|------|
| `std::vector<std::thread> workers_` | Pool threads, created in constructor |
| `std::mutex mutex_` | Protect launch state setup |
| `std::atomic<bool> shutdown_` | Destructor signals workers to exit |
| `std::atomic<bool> work_available_` | New bulk launch is active |
| `IRunnable* runnable_` | Current task object |
| `int num_total_tasks_` | Tasks in current launch |
| `std::atomic<int> next_task_id_` | Dynamic assignment counter |
| `std::atomic<int> tasks_completed_` | Finished task count |
| `void workerLoop()` | Private worker entry point |

#### Worker loop detail

1. **Wait for work (spin):** `while (!work_available_ && !shutdown_) { }`
2. **Grab task (dynamic):** `task_id = next_task_id_++`
3. **No task left for this worker:** if `task_id >= num_total_tasks_`, spin until `!work_available_` (other workers may still be running), then loop back
4. **Execute:** `runnable_->runTask(task_id, num_total_tasks_)`
5. **Completion:** if `++tasks_completed_ == num_total_tasks_`, set `work_available_ = false`

**Why dynamic assignment (vs Step 1 static):** Tests like `ping_pong_unequal` give lower-index tasks more work. Static assignment can leave some threads idle while one thread is slow. An atomic task counter lets fast threads pick up remaining work.

#### Main thread synchronous wait

Main spins: `while (tasks_completed_ < num_total_tasks_) { }`

This is the answer to the README question: **`run()` does not return until the completion counter reaches `num_total_tasks_`.**

#### Edge cases to handle

- **`num_total_tasks == 0`:** return immediately, no state change
- **`num_threads_ <= 1`:** serial fallback (same as Step 1)
- **Workers > tasks:** extra workers get `task_id >= num_total_tasks`, spin until launch completes
- **Back-to-back `run()` calls:** reset all counters before setting `work_available_ = true`; previous launch must be fully done before main returns
- **Destructor:** set `shutdown_ = true` and ensure spinning workers can exit (may need to set `work_available_ = true` as a wake-up)

#### Tests to run after implementation

```bash
./runtasks -n 8 simple_test_sync
python3 ../tests/run_test_harness.py -n 8 -t ping_pong_equal super_light super_super_light
```

`ping_pong_equal` is the key test where thread pool should beat spawn.

---

### Step 3: `TaskSystemParallelThreadPoolSleeping` — Not started

**Goal:** Same thread pool structure as Step 2, but replace busy-wait loops with **`std::condition_variable`** so idle workers and the main thread do not consume CPU while waiting.

**Changes from Step 2:**

| Waiting situation | Step 2 (Spin) | Step 3 (Sleep) |
|-------------------|---------------|----------------|
| Worker, no work | `while (!work_available_) {}` | `workers_cv_.wait(lock, ...)` |
| Worker, no tasks left this launch | spin until `!work_available_` | `workers_cv_.wait(...)` |
| Main, tasks not done | `while (completed < total) {}` | `main_cv_.wait(lock, ...)` |
| New work arrives | set atomic flag | set flag + `notify_all()` |

**Additional members:** `std::condition_variable workers_cv_`, `std::condition_variable main_cv_`

**Why it matters:** On `super_light` / `ping_pong`, a spinning main thread competes with workers for CPU cycles. Sleeping yields the core to workers doing useful work.

**Part B note:** Step 3's class is extended with `runAsyncWithDeps()` and `sync()`. The dependency tracking (waiting queue vs ready queue) builds on this sleeping pool.

*Implementation details to be filled after Step 2 is complete.*

---

## Part B — Not started

Implement on top of Step 3 sleeping thread pool:

- **`runAsyncWithDeps(runnable, num_total_tasks, deps)`** — returns immediately with a `TaskID`; tasks may still be running
- **`sync()`** — blocks until all prior async launches complete
- **Dependency rule:** tasks in a launch cannot start until all tasks in every `deps` launch have finished

**Planned data structures:**

1. **Waiting set** — launches whose dependencies are not yet satisfied
2. **Ready queue** — tasks that can run now; workers pull from here

*Writeup question on dependency tracking: TBD after implementation.*

---

## Writeup Questions (draft / TBD)

### 1. Task system implementation

**Thread management (so far):**

- Step 1: spawn-per-`run()` — no pool.
- Step 2 (planned): fixed pool in constructor, destroyed in destructor.
- Step 3 (planned): same pool + condition variables for idle/synchronization.

**Task assignment (so far):**

- Step 1: **static** round-robin by thread id.
- Steps 2–3 (planned): **dynamic** via atomic `next_task_id_`.

**Part B dependencies:** TBD.

### 2. When simpler implementations win

Expected patterns (to confirm with measurements after Steps 2–3):

| Workload type | Example test | Likely winner | Why |
|---------------|----------------|---------------|-----|
| Very cheap tasks, few launches | `super_super_light` | **Serial** or Spawn | Pool/sync overhead exceeds useful work |
| Many cheap launches | `ping_pong`, `spin_between_run_calls` | **Thread pool** | Amortizes thread creation; spawn loses |
| Heavy per-task work | `mandelbrot_chunked`, `recursive_fibonacci` | **Parallel** (any) | Compute dominates; overhead negligible |
| Uneven task cost | `ping_pong_unequal` | **Pool + dynamic** | Static assignment leaves threads idle |

### 3. Custom test

*Not yet implemented. Skeleton: `YourTask` / `yourTest()` in `tests/tests.h`. Must register in `tests/main.cpp`.*

---

## Changelog

| Date | Update |
|------|--------|
| 2026-06-12 | Report created. Local Mac workflow documented (AWS skipped). Part A Step 1 implemented and tested. |
| 2026-06-12 | Expanded: assignment overview, core concepts, four-variant terminology, test call chain, Step 1–3 design comparison, detailed Step 2 plan, Step 3 preview, writeup draft for Q2. |
