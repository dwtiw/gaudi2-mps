# gaudi2-mps

Run up to three Synapse processes on one Intel Gaudi2 card at the same time, in the style of NVIDIA MPS.
Programs need no changes, and the stock `habanalabs` kernel driver is used as-is.
(CLI binary and preload library keep the `hpu-mps` / `libhpu_mps.so` names.)

## Requirements

- Gaudi2, x86_64 Linux
- SynapseAI **1.24.2-481**. Other builds are detected at load time and run without MPS (see [Compatibility](#compatibility)).
- All processes sharing a card run as the same user.

## Build

```sh
git clone https://github.com/dwtiw/gaudi2-mps.git
cd gaudi2-mps
make
sudo make install            # PREFIX=/usr/local by default
```

`libhpu_mps.so` and `hpu-mps-top` need no SynapseAI headers or libraries at build time (`hpu-mps-top` loads `libhlml.so` at runtime). `scal-xfer` links against `libSynapse.so`
(`HABANA_LIB=/usr/lib/habanalabs`).

## Usage

With a background primary (recommended):

```sh
hpu-mps start                # Slot 0 primary, 1.5 GB HBM (2 x 768 MiB pages)
hpu-mps set-mem 1 48         # optional per-slot default quota
hpu-mps -m 36 python serve_a.py &
hpu-mps -m 36 python serve_b.py &
hpu-mps status
hpu-mps top                  # live TUI (AIP/HBM/power plot + per-slot processes)
hpu-mps top --once           # single snapshot (or --json for scripts/agents)
hpu-mps stop
```

As root, `start` also adds the library to `/etc/ld.so.preload`, so HPU programs can run without the `hpu-mps` prefix.
`stop` (or `hpu-mps disable-global`) removes it again.

Without a daemon, the first process becomes the primary:

```sh
hpu-mps -m 24 ./app0 &
hpu-mps -m 24 ./app1 &
hpu-mps -m 24 ./app2 &
```

Check that the card works with `scal-xfer`:

```sh
hpu-mps ./build/scal-xfer loop 10
```

| Variable | Effect |
|---|---|
| `HPU_MPS_HBM_GB` | HBM quota of this process (1..90) |
| `HPU_MPS_DAEMON_HBM_MB` | Override Slot 0 daemon HBM pool in MiB (768..4608, rounded up to 768 MiB, default 1536) |
| `HPU_MPS_SLOT` | Request slot 0, 1 or 2 |
| `HPU_MPS_QUIET=1` | No `[hpu-mps]` log lines |
| `HPU_MPS_DISABLE=1` | Pass-through for this process |
| `HPU_MPS_RUN_DIR` | State directory (tmpfs, mode 0700). Default `/run/hpu-mps` for root, `$XDG_RUNTIME_DIR/hpu-mps` otherwise |
| `HPU_MPS_ALLOW_UNVERIFIED=1` | Activate on an untested SynapseAI build |

## Agent Skill

To teach coding agents (Claude Code, Codex, Antigravity, etc.) how to check `hpu-mps status`, size `HPU_MPS_HBM_GB`, and avoid killing the Slot 0 primary daemon:

```sh
npx skills add dwtiw/gaudi2-mps
```

The skill lives at [`skills/hpu-mps/SKILL.md`](skills/hpu-mps/SKILL.md).

## How it works

On Gaudi2 the runtime's device bring-up switches the PDMA command fetcher from its boot queue to the ARC-managed
queue, and the switch cannot be undone. A second process repeating the bring-up submits to a queue the hardware no
longer reads, and times out. The driver also allows only one compute context per device.

`libhpu_mps.so` is loaded with `LD_PRELOAD` and intercepts calls into `libhl-thunk`, `libscal` and `libSynapse`:

- Slot 0 opens the device and does the full bring-up. It passes its device fd to Slots 1 and 2 over a Unix socket
  (`SCM_RIGHTS`), so all processes share one driver context.
- Slots 1 and 2 skip the one-time bring-up steps and attach to the running engines.
- Each slot gets its own stream queues, completion queues, sync objects, monitors, fences, host and HBM pools and
  device VA ranges.
- Compute recipe launches are serialized across slots by a process-shared lock with time-sliced fairness. PDMA
  transfers keep running in parallel.

## Limitations

- **No isolation between processes.** All slots share one ASID and page table. The HBM split is enforced in
  software only, so a faulty or malicious process can read or corrupt another slot's memory, or hang the card.
  Only share a card between programs you trust.
- At most 3 processes per card.
- The library relies on internal symbols and data layouts of the SynapseAI build listed above.
- The primary waits for Slots 1 and 2 to exit before it tears down. `hpu-mps stop --force` kills it anyway.

## Compatibility

At first use the library compares the GNU build IDs of the loaded `libscal.so` and `libSynapse.so` with a built-in
list (`kVerifiedBuilds` in `src/hpu_mps.cpp`). If they do not match, it prints a warning and every hook passes
straight through to the original function. `HPU_MPS_ALLOW_UNVERIFIED=1` overrides this; expect hangs.

To add a build, verify it with the `scal-xfer` tests above under three concurrent slots, then add its IDs:

```sh
readelf -n /usr/lib/habanalabs/libscal.so    | grep 'Build ID'
readelf -n /usr/lib/habanalabs/libSynapse.so | grep 'Build ID'
```

## Disclaimer

This project is not affiliated with or endorsed by Intel or Habana Labs. Intel, Gaudi and SynapseAI are trademarks
of Intel Corporation. SynapseAI is not included; obtain it from Intel and comply with its license. This software is
provided as is, without warranty; it can hang the device and require a reset.

## License

[Apache-2.0](LICENSE)
