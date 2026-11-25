---
name: hpu-mps
description: >-
  Use when running, inspecting, benchmarking, or troubleshooting workloads on an
  Intel Gaudi2 HPU host with hpu-mps (gaudi2-mps) installed. Covers reading
  hpu-mps status vs hl-smi, sizing HPU_MPS_HBM_GB quotas, running concurrent
  processes, avoiding Slot 0 teardown, and switching between shared and
  exclusive modes.
---

# hpu-mps (Gaudi2 Multi-Process Service)

`hpu-mps` shares a single Intel Gaudi2 (`habanalabs`) compute context across up
to 3 processes (`Slot 0` Primary, `Slot 1` Attach-1, `Slot 2` Attach-2) via
`LD_PRELOAD` (`libhpu_mps.so`).

## 1. Check Card State First

Always run `hpu-mps status` before launching jobs or assuming the card is busy.
Do not rely on `hl-smi` alone.

```bash
hpu-mps status
hpu-mps top --once      # hardware telemetry + per-slot processes (or --json)
```

How to read the output:

- **Daemon + Zero-Prefix Mode** (`Primary Bringup Ready : YES`, `Zero-Prefix Mode : ENABLED`):
  - `Slot 0` is held by the background `scal-xfer serve` primary (`1.5 GiB` HBM,
    2 × 768 MiB DRAM pages). `hl-smi` will always show `scal-xfer` (or another
    PID sharing the single kernel context) — **never kill Slot 0**; it holds the
    shared device context.
  - `Slot 1` and `Slot 2` are available for up to **2 concurrent workloads**
    (default `36 GB` HBM each). Run HPU binaries directly without the `hpu-mps`
    prefix; `/etc/ld.so.preload` attaches them automatically.
  - If one workload slot is `ACTIVE` and the other is `FREE`, you can launch a
    second HPU process immediately without stopping the first.
  - Workloads are full only when both `Slot 1` and `Slot 2` are `ACTIVE`.
- **Daemon stopped, Zero-Prefix disabled** (`Primary Bringup Ready : NO`):
  - If `hl-smi` already shows a compute process, that process opened
    `/dev/accel/accel0` directly without MPS. A second process will fail with
    `EBUSY` until the first exits and MPS is started.
  - If the card is idle, either start the daemon (`systemctl start hpu-mps` or
    `hpu-mps start`) for zero-prefix mode, or launch up to 3 processes
    explicitly via `hpu-mps [-m <GB>] <cmd>` (default `24 GB` each).

## 2. Size HBM Quotas

Valid per-process quota is `1..90` GB. Each slot's physical HBM allocation
rounds up to Gaudi2's `768 MiB` DRAM page size, and the sum of physical
allocations across active slots must fit within usable HBM (~95.25 GiB on a
96 GiB card; in daemon mode `Slot 0` uses `1.5 GiB`, leaving ~93.75 GiB for
`Slot 1` + `Slot 2`, independent of `12 GiB` VA alignment).

```bash
# Override quota for one process (works in zero-prefix and prefixed mode)
HPU_MPS_HBM_GB=48 python train.py
hpu-mps -m 48 python train.py

# Target a specific slot or quiet [hpu-mps] stderr logs
HPU_MPS_SLOT=2 HPU_MPS_HBM_GB=16 HPU_MPS_QUIET=1 python serve.py

# Change default quota for a slot (persists in $HPU_MPS_RUN_DIR/hbm.state)
hpu-mps set-mem 1 56
hpu-mps set-mem 2 28
```

- Device VA ranges are 12 GiB-aligned while physical HBM allocates in 768 MiB
  DRAM pages (enforced in software only over a shared ASID/page table). Only
  share a card among trusted workloads and do not overcommit past physical HBM.
- Forked children (such as PyTorch `DataLoader(num_workers>0)`) automatically
  drop slot ownership and detach inherited `/dev/accel/*` fds in a
  `pthread_atfork` child handler so worker processes cannot pin `compute_ctx`
  or block driver resets if the parent exits.

## 3. Compute Contention and Benchmarking

- PDMA transfers run in parallel across slots; compute recipe launches are
  serialized across slots by a process-shared lock with time-sliced fairness.
- Concurrent training and serving work normally, but share MME/TPC compute and
  the board power cap.
- Before running latency or throughput benchmarks, confirm in `hpu-mps status`
  that all other workload slots are `FREE`.

## 4. Lifecycle and Switching to Exclusive Mode

Never kill `Slot 0` (`scal-xfer`) while `Slot 1` or `Slot 2` is `ACTIVE`; doing
so tears down the shared context and breaks attached processes.

```bash
# Clean restart when Slot 1 and Slot 2 are FREE
systemctl restart hpu-mps   # if systemd-managed
# or: hpu-mps stop && hpu-mps start

# Force recovery only if a slot is wedged
hpu-mps stop --force
```

To run a single exclusive workload (for example, needing >90 GB HBM or
debugging without `libhpu_mps.so`):

1. Wait until `Slot 1` and `Slot 2` are `FREE`.
2. Stop the primary (`systemctl stop hpu-mps` or `hpu-mps stop`), which also
   removes `libhpu_mps.so` from `/etc/ld.so.preload`.
3. Run the exclusive job (do not rely on `HPU_MPS_DISABLE=1` while the primary
   still holds `/dev/accel/accel0`, as a direct open will hit `EBUSY`).
4. Re-enable MPS (`systemctl start hpu-mps` or `hpu-mps start`) when finished.

## 5. Verification and Troubleshooting

Verify 2-process zero-prefix concurrency after install or restart:

```bash
/usr/local/lib/hpu-mps/scal-xfer loop 5 &
/usr/local/lib/hpu-mps/scal-xfer loop 5 &
wait
hpu-mps status
```

Common issues:

| Symptom | Cause | Fix |
|---|---|---|
| `EBUSY` opening `/dev/accel/accel0` | An earlier process opened the card without `libhpu_mps.so` | Wait for or stop the non-MPS process, then start `hpu-mps` (`systemctl start hpu-mps` or `hpu-mps start`) |
| `running without MPS` on stderr | `libscal.so` / `libSynapse.so` GNU build IDs not in `kVerifiedBuilds` | Check `readelf -n /usr/lib/habanalabs/lib{scal,Synapse}.so \| grep 'Build ID'`; test with `scal-xfer` before adding to `src/hpu_mps.cpp` |
| `Slot 1/2 still in use` on `hpu-mps stop` | Secondary processes are still running | Stop processes listed in `hpu-mps status` first, or use `hpu-mps stop --force` for recovery |
| OOM inside a workload despite free card HBM | Default slot quota (`36 GB` / `24 GB`) is smaller than model requirement | Re-launch with a higher `HPU_MPS_HBM_GB=<GB>` (within remaining usable HBM) |
