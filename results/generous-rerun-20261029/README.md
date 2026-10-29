# Generous-budget complete rerun — 2026-10-29

**Stopped and finalized at the user's request.** Use the
[final report](finalized/REPORT.md) and [frozen timings](finalized/timings.csv)
for conclusions. There are 210 completed configurations; incomplete cases are
retained separately. The supervisor, cutoff monitors and database container are
stopped. The descriptions below preserve the campaign's execution history.

This is a fresh campaign. Consult `progress.json` for live status; this file does
not assert completion. Earlier results are preserved and are not pooled here.

On 2026-10-29, the user requested skipping the remaining 256-qubit full-reduce
cases. Case index 90 had already been interrupted; indices 91–104 are explicitly
skipped, including the partial attempt of index 91. The original plan is retained;
`skip-decisions.json` records this scope change. `continue_single_rules.py`
continues all 152 single-rule cases with the same measured sources and limits.
Its logs are `continuation.log` and `continuation-error.log`. Reopen the terminal
viewer to display skipped cases separately from failures.

The user subsequently requested skipping cases with very long warmups. The
independent `limit_warmups.py` monitor applies a 30-minute wall-time ceiling to
each remaining engine warmup, starting with case index 207. Exceeding it ends
that case and lets the original supervisor recover and continue. Measured
repetition budgets are unchanged. `warmup-policy.json` records the policy;
each affected case retains `warmup-skip.json` and all partial observations.
The supervisor records worker termination as an error; the updated viewer
uses these explicit decision files to classify such cases as skipped.
The limit includes verification before the engine completion log event.

The user then extended the 30-minute cutoff to measured repetitions as well.
`limit_engine_runs.py` supersedes the warmup-only monitor, covering each engine
invocation from case 210 onward. Case 209 had already completed and its results
remain intact. A cutoff interrupts the whole case; the supervisor recovers and
continues. `engine-run-policy.json` records the effective time and scope;
`engine-run-monitor-status.json` records the latest heartbeat. Each cutoff
retains `engine-run-skip.json` and partial measurements. Logs are
`engine-run-monitor.log` and `engine-run-monitor-error.log`. The monitor checks
every five seconds and includes verification before the completion log event.
Earlier two-hour limits remain as fallback watchdogs in the unchanged runner;
the external monitor now enforces the shorter wall-time cutoff.

The user subsequently shortened this cutoff to **five minutes (300 seconds)**
for both warmups and measured engine runs, effective during case 212. The new
monitor applies elapsed time already spent in the current run. Earlier completed
measurements remain intact. The superseded policy is preserved in
`engine-run-policy-1800s.json`; the current policy remains `engine-run-policy.json`.
The five-minute monitor writes `engine-run-monitor-300s.log` and
`engine-run-monitor-300s-error.log`. Reopen the viewer to update its cutoff label.

## Scope and execution

- All regressions and 223 correctness inputs in both optimized Cypher and native modes precede timing. A failed correctness gate stops the campaign for inspection.
- 257 performance configurations: the original rule grid, identity chains, the full large-rule extension, and all five full-reduce circuit families at 3, 8, 16, 32, 64, 128 and 256 qubits with three seed labels each.
- Three engines: stock PyZX, optimized Cypher, and hybrid Cypher+C++. One warmup and three measured repetitions each: 3,084 planned engine trials.
- Sparse rule sizes up to 16,384; disjoint motifs up to 16,384; dense fan sizes up to 1,024. Random circuit depth is 20 times qubit count; layered circuits retain the existing two-layer generator.
- Database watchdog/query budget: 7,200 seconds per engine. PyZX uses the existing cooperative 7,200-second deadline; a long internal PyZX call can exceed it. A hard 93,600-second (26-hour) subprocess limit bounds each complete case including verification and all repetitions. There is no overall campaign time limit.
- Cases run serially. A failed case retains its evidence, restarts only the dedicated container, cleans only its recorded graph ID, and does not suppress later sizes of its family. A failed repetition stops further repetitions of that case.
- Memgraph 3.14.0 runs in the new dedicated `zxdb-generous-20261029` container at `127.0.0.1:7692`. Docker reports 16 CPUs and 25,201,369,088 bytes of VM RAM (about 23.5 GiB). No additional container CPU/memory quota was imposed. Host RAM is about 31.9 GiB. These are available resources, not guaranteed peak usable graph memory.
- The native binary SHA-256 is `465ca8969b457c592a92e90a1f3d705024e11c1a8fbdddfecfb2d3b8f778cd78`, matching the recorded validated build. All native procedures were checked. Sources are hashed before the run and checked before every case; individual cases archive their sources.

## Evidence

`plan.json` records the exact grid and limits. `progress.json` records the active
case and every case outcome. `correctness/` contains regression and fixture
evidence. `cases/` retains raw observations, snapshots, and failure diagnostics.
`performance/REPORT.md` and `performance/timings.csv` are regenerated after each
case; incomplete cells have no speedup. Figures are generated when the campaign
finishes. `supervisor.log`, `supervisor-error.log`, and each case's `run.log` show
execution details. Source edits during execution stop the campaign.

Large-output semantic checks may remain inconclusive under the existing oracle
bounds. This campaign does not equate timing completion, structural agreement,
or a large workload with semantic correctness or a database win. Local-layered
seed labels can generate identical inputs and are not independent samples.

The supervisor is `evaluation/run_generous_campaign.py`. It was launched hidden
so that closing the chat does not itself end the process. The machine and Docker
must remain running. A stopped campaign must be inspected before a fresh attempt;
the supervisor deliberately refuses to overwrite or silently resume this folder.
