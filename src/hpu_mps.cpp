// SPDX-License-Identifier: Apache-2.0
/*
 * hpu_mps.cpp -- CUDA-MPS-like multi-process HPU sharing interposer for Gaudi2.
 *
 * Allows up to 3 unmodified SCAL/Synapse processes (Slot 0, Slot 1, Slot 2) to
 * share a single Gaudi2 HPU concurrently on the stock habanalabs driver:
 *
 *   1. Zero-Prefix Transparent Execution:
 *      Safe for global /etc/ld.so.preload (zero C++ static constructors; inert
 *      for processes that do not load libscal.so, and for SynapseAI builds that
 *      are not in kVerifiedBuilds).
 *
 *   2. Dynamic & Asymmetric HBM Pool Slicing (<run dir>/hbm.state):
 *      - Background Primary daemon (Slot 0, HPU_MPS_DAEMON=1) reserves 1.5 GiB
 *        (2 x 768 MiB DRAM pages, zero rounding waste; tunable via HPU_MPS_DAEMON_HBM_MB).
 *      - Each client process can request any custom HBM quota (1..90 GiB) via
 *        `hpu-mps -m <GB>` / `HPU_MPS_HBM_GB=<GB>` or `hpu-mps set-mem <slot> <GB>`.
 *      - Physical capacity is checked against the sum of 768 MiB-rounded slot
 *        allocations, decoupled from the 12 GiB-aligned VA placement (preserving
 *        32MB page alignment, 1GB DMMU alignment, and 0x50000000 ARC offset).
 *
 *   3. Intra-Process Multi-Stream Virtualization:
 *      Clamps Synapse stream affinity masks to 0x1 so a single process can create
 *      an arbitrary number of synStreamHandle streams (e.g. PyTorch compute/DMA
 *      streams) safely multiplexed onto its own slot's dedicated hardware queues
 *      (*0 -> *{slot}) without touching other slots' queues.
 *
 * All processes sharing a card must run as the same user: state lives in a
 * private 0700 tmpfs directory (/run/hpu-mps for root, $XDG_RUNTIME_DIR/hpu-mps
 * otherwise, or $HPU_MPS_RUN_DIR).
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <ctype.h>
#include <dlfcn.h>
#include <elf.h>
#include <errno.h>
#include <fcntl.h>
#include <link.h>
#include <linux/magic.h>
#include <pthread.h>
#include <sched.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/eventfd.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/sysmacros.h>
#include <sys/types.h>
#include <sys/un.h>
#include <sys/vfs.h>
#include <time.h>
#include <unistd.h>

#include <string>

#include "mps_shm.h"

static constexpr uint64_t kSlotDramVaBase        = 0x1001100000000000ULL;
static constexpr uint64_t kSlotDramVaStride      = 0x0000100000000000ULL; // 16 TiB per slot
static constexpr uint64_t kDramVaAlign           = 0x40000000ULL;         // 1 GiB alignment
static constexpr uint64_t kSlotHostHugeVaBase    = 0xfff0800000000000ULL;
static constexpr uint64_t kSlotHostHugeVaStride  = 0x0000001000000000ULL; // 64 GiB per slot in HOST_HUGE range
static constexpr uint64_t kHostHugeVaAlign       = 0x200000ULL;           // 2 MiB hugepage alignment

static int g_slot = -1;
static int g_slot_lock_fd = -1;
static int g_slot_accel_fd = -1;
static int g_primary_accel_fd = -1;
static int g_fd_server_listen_fd = -1;
static bool g_fd_server_started = false;
static dev_t g_accel_dev_rdev = 0;
static bool g_has_accel_dev_rdev = false;
static dev_t g_accel_ctrl_rdev = 0;
static bool g_has_accel_ctrl_rdev = false;
static pid_t g_slot_owner_pid = -1;
static bool g_verbose = true;
static uint64_t g_my_hbm_cap_bytes = 0;
static uint64_t g_my_hbm_rel_offset = 0;
static bool g_my_hbm_offset_assigned = false;
static uint64_t g_card_hbm_bytes = 0;
static volatile uint64_t *g_cq_counters_host = nullptr;
static uint32_t g_my_compute_cq_idx = 0xffffffffu;
static constexpr uint64_t kNoHbmRange = UINT64_MAX;

static void lock_internal_mutexes_for_fork();
static void unlock_internal_mutexes_after_fork(bool reset_child_state);
static void set_coeff_table_state(uint32_t state, int32_t pid);

#define MPS_LOG(fmt, ...)                                                      \
    do {                                                                       \
        if (g_verbose) {                                                       \
            fprintf(stderr, "[hpu-mps slot=%d pid=%d] " fmt "\n",              \
                    g_slot, (int)getpid(), ##__VA_ARGS__);                     \
        }                                                                      \
    } while (0)

// ============================================================================
// Private runtime directory
// ============================================================================

// Leaves room for the longest file name below inside sockaddr_un::sun_path.
static constexpr size_t kRunDirMax = sizeof(((struct sockaddr_un *)nullptr)->sun_path) - 32;
static char g_run_dir[kRunDirMax];

static bool init_run_dir()
{
    const char *env = getenv("HPU_MPS_RUN_DIR");
    int n;
    if (env && env[0]) {
        n = snprintf(g_run_dir, sizeof(g_run_dir), "%s", env);
    } else if (geteuid() == 0) {
        n = snprintf(g_run_dir, sizeof(g_run_dir), "/run/hpu-mps");
    } else {
        const char *xdg = getenv("XDG_RUNTIME_DIR");
        if (xdg && xdg[0] == '/')
            n = snprintf(g_run_dir, sizeof(g_run_dir), "%s/hpu-mps", xdg);
        else
            n = snprintf(g_run_dir, sizeof(g_run_dir), "/run/user/%u/hpu-mps", (unsigned)geteuid());
    }
    if (n <= 0 || (size_t)n >= sizeof(g_run_dir) || g_run_dir[0] != '/') {
        fprintf(stderr, "[hpu-mps pid=%d] run dir path is not absolute or too long\n", (int)getpid());
        return false;
    }
    if (mkdir(g_run_dir, 0700) != 0 && errno != EEXIST) {
        fprintf(stderr, "[hpu-mps pid=%d] cannot create %s: %s\n", (int)getpid(), g_run_dir, strerror(errno));
        return false;
    }
    struct stat st;
    if (lstat(g_run_dir, &st) != 0 || !S_ISDIR(st.st_mode) || st.st_uid != geteuid() ||
        (st.st_mode & 077) != 0) {
        fprintf(stderr, "[hpu-mps pid=%d] %s must be a directory owned by uid %u with mode 0700\n",
                (int)getpid(), g_run_dir, (unsigned)geteuid());
        return false;
    }
    // The CQ counter page is DMA-pinned by the device; long-term pins of writable
    // file-backed pages are only allowed on shmem.
    struct statfs sfs;
    if (statfs(g_run_dir, &sfs) != 0 || sfs.f_type != TMPFS_MAGIC) {
        fprintf(stderr, "[hpu-mps pid=%d] %s must be on tmpfs\n", (int)getpid(), g_run_dir);
        return false;
    }
    return true;
}

static void run_path(char *buf, size_t len, const char *name)
{
    snprintf(buf, len, "%s/%s", g_run_dir, name);
}

static int open_run_file(const char *name, int flags)
{
    char path[sizeof(g_run_dir) + 32];
    run_path(path, sizeof(path), name);
    return open(path, flags | O_CLOEXEC | O_NOFOLLOW, 0600);
}

// Exclusive flock on <run dir>/<name>; returns the fd to pass to unlock_run_file.
static int lock_run_file(const char *name)
{
    int fd = open_run_file(name, O_RDWR | O_CREAT);
    if (fd >= 0)
        flock(fd, LOCK_EX);
    return fd;
}

static void unlock_run_file(int fd)
{
    if (fd < 0)
        return;
    flock(fd, LOCK_UN);
    close(fd);
}

static void lock_path_for_slot(int slot, char *buf, size_t len)
{
    char name[32];
    snprintf(name, sizeof(name), "slot-%d.lock", slot);
    run_path(buf, len, name);
}

static void unlink_run_file(const char *name)
{
    char path[sizeof(g_run_dir) + 32];
    run_path(path, sizeof(path), name);
    unlink(path);
}

// kReadyFile and kHbmStateFile are defined in mps_shm.h.
static constexpr const char *kFdSockFile = "fd.sock";

// ============================================================================
// Activation: only for verified SynapseAI builds
// ============================================================================

// Every raw struct offset and mangled symbol below was validated against these
// builds. Anything else falls back to pass-through instead of corrupting state.
struct VerifiedBuild {
    const char *release;
    const char *scal_build_id;
    const char *synapse_build_id;
};

static const VerifiedBuild kVerifiedBuilds[] = {
    {"1.24.2-481", "2c533aeb2023fb6ca40bd726ee16031ff6e087b3", "c02ac9c8860c05fb988fa69d07548478870c345b"},
};

struct LoadedLib {
    const char *soname;
    char path[256];
    char build_id[2 * 64 + 1];
};

static LoadedLib g_lib_scal = {"libscal.so", {0}, {0}};
static LoadedLib g_lib_synapse = {"libSynapse.so", {0}, {0}};
static LoadedLib g_lib_hlthunk = {"libhl-thunk.so", {0}, {0}};
static LoadedLib g_lib_hlml = {"libhlml.so", {0}, {0}};

static void read_build_id(const struct dl_phdr_info *info, char *out, size_t out_len)
{
    for (int i = 0; i < info->dlpi_phnum; ++i) {
        const ElfW(Phdr) *ph = &info->dlpi_phdr[i];
        if (ph->p_type != PT_NOTE)
            continue;
        const char *p = (const char *)(info->dlpi_addr + ph->p_vaddr);
        const char *end = p + ph->p_memsz;
        while (p + sizeof(ElfW(Nhdr)) <= end) {
            const ElfW(Nhdr) *nh = (const ElfW(Nhdr) *)p;
            const char *name = p + sizeof(*nh);
            const unsigned char *desc = (const unsigned char *)(name + ((nh->n_namesz + 3) & ~3u));
            if ((const char *)desc + nh->n_descsz > end)
                break;
            if (nh->n_type == NT_GNU_BUILD_ID && nh->n_namesz == 4 && memcmp(name, "GNU", 4) == 0 &&
                2 * nh->n_descsz < out_len) {
                for (uint32_t b = 0; b < nh->n_descsz; ++b)
                    snprintf(out + 2 * b, 3, "%02x", desc[b]);
                return;
            }
            p = (const char *)desc + ((nh->n_descsz + 3) & ~3u);
        }
    }
}

static int collect_loaded_lib(struct dl_phdr_info *info, size_t, void *)
{
    if (!info->dlpi_name || !info->dlpi_name[0])
        return 0;
    const char *slash = strrchr(info->dlpi_name, '/');
    const char *base = slash ? slash + 1 : info->dlpi_name;
    LoadedLib *libs[] = {&g_lib_scal, &g_lib_synapse, &g_lib_hlthunk, &g_lib_hlml};
    for (LoadedLib *lib : libs) {
        if (lib->path[0] || strncmp(base, lib->soname, strlen(lib->soname)) != 0)
            continue;
        snprintf(lib->path, sizeof(lib->path), "%s", info->dlpi_name);
        read_build_id(info, lib->build_id, sizeof(lib->build_id));
    }
    return 0;
}

static bool is_verified_build()
{
    for (const VerifiedBuild &b : kVerifiedBuilds) {
        if (strcmp(g_lib_scal.build_id, b.scal_build_id) == 0 &&
            strcmp(g_lib_synapse.build_id, b.synapse_build_id) == 0)
            return true;
    }
    return false;
}

static bool env_is_1(const char *name)
{
    const char *v = getenv(name);
    return v && v[0] == '1';
}

static pthread_once_t g_mps_once = PTHREAD_ONCE_INIT;
static bool g_mps_active = false;

static void decide_mps_active()
{
    dl_iterate_phdr(collect_loaded_lib, nullptr);
    if (env_is_1("HPU_MPS_DISABLE") || !g_lib_scal.path[0])
        return;
    if (!is_verified_build()) {
        if (!env_is_1("HPU_MPS_ALLOW_UNVERIFIED")) {
            fprintf(stderr,
                    "[hpu-mps pid=%d] SynapseAI build not verified (libscal %s, libSynapse %s); "
                    "running without MPS. Set HPU_MPS_ALLOW_UNVERIFIED=1 to try anyway.\n",
                    (int)getpid(), g_lib_scal.build_id[0] ? g_lib_scal.build_id : "?",
                    g_lib_synapse.build_id[0] ? g_lib_synapse.build_id : "?");
            return;
        }
        fprintf(stderr, "[hpu-mps pid=%d] WARNING: forcing MPS on an unverified SynapseAI build\n",
                (int)getpid());
    }
    if (!init_run_dir()) {
        fprintf(stderr, "[hpu-mps pid=%d] running without MPS\n", (int)getpid());
        return;
    }
    g_mps_active = true;
}

static bool is_mps_disabled()
{
    pthread_once(&g_mps_once, decide_mps_active);
    return !g_mps_active;
}

static int read_pid_file(const char *path)
{
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0)
        return -1;
    char buf[64] = {0};
    ssize_t n = read(fd, buf, sizeof(buf) - 1);
    close(fd);
    if (n <= 0)
        return -1;
    return atoi(buf);
}

static bool write_pid_file(const char *path, int pid)
{
    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC | O_NOFOLLOW, 0600);
    if (fd < 0)
        return false;
    char buf[64];
    int len = snprintf(buf, sizeof(buf), "%d\n", pid);
    ssize_t w = write(fd, buf, len);
    close(fd);
    return w == len;
}

// Slot ownership uses OFD locks so that probing (F_OFD_GETLK) never holds the lock.
static bool try_lock_slot_fd(int fd)
{
    struct flock fl = {};
    fl.l_type = F_WRLCK;
    fl.l_whence = SEEK_SET;
    return fcntl(fd, F_OFD_SETLK, &fl) == 0;
}

static bool slot_lock_held_elsewhere(int fd)
{
    struct flock fl = {};
    fl.l_type = F_WRLCK;
    fl.l_whence = SEEK_SET;
    return fcntl(fd, F_OFD_GETLK, &fl) == 0 && fl.l_type != F_UNLCK;
}

static bool is_slot_locked(int slot)
{
    if (slot == g_slot && g_slot_lock_fd >= 0 && g_slot_owner_pid == getpid())
        return true;
    char path[128];
    lock_path_for_slot(slot, path, sizeof(path));
    int fd = open(path, O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    if (fd < 0)
        return false;
    bool locked = slot_lock_held_elsewhere(fd);
    close(fd);
    return locked;
}

static bool any_secondary_slot_alive()
{
    for (int s = 1; s < kMaxSlots; ++s) {
        if (is_slot_locked(s))
            return true;
    }
    return false;
}

static int get_ready_slot0_pid()
{
    if (!is_slot_locked(0))
        return 0;
    char lock_path[128];
    lock_path_for_slot(0, lock_path, sizeof(lock_path));
    int lock_pid = read_pid_file(lock_path);
    char ready_path[sizeof(g_run_dir) + 32];
    run_path(ready_path, sizeof(ready_path), kReadyFile);
    int ready_pid = read_pid_file(ready_path);
    if (lock_pid <= 0 || ready_pid <= 0 || lock_pid != ready_pid)
        return 0;
    if (kill(lock_pid, 0) != 0 && errno == ESRCH)
        return 0;
    return lock_pid;
}

static bool is_slot0_ready()
{
    return get_ready_slot0_pid() > 0;
}

// ============================================================================
// Shared Dynamic HBM State Table (<run dir>/hbm.state)
// ============================================================================
// Text format (3 lines, one per slot 0..2):
//   <slot> <pid> <rel_offset_hex> <size_bytes_hex> <cfg_gb> <is_daemon>
struct HbmSlotEntry {
    int pid;
    uint64_t rel_offset;
    uint64_t size_bytes;
    uint64_t cfg_gb;
    int is_daemon;
};

static void load_hbm_state_locked(HbmSlotEntry entries[kMaxSlots])
{
    for (int s = 0; s < kMaxSlots; ++s) {
        entries[s].pid = 0;
        entries[s].rel_offset = 0;
        entries[s].size_bytes = 0;
        entries[s].cfg_gb = 0;
        entries[s].is_daemon = 0;
    }
    int state_fd = open_run_file(kHbmStateFile, O_RDONLY);
    FILE *fp = state_fd >= 0 ? fdopen(state_fd, "r") : nullptr;
    if (fp) {
        char line[256];
        while (fgets(line, sizeof(line), fp)) {
            int s = -1, pid = 0, is_daemon = 0;
            unsigned long long rel_off = 0, sz = 0, cfg = 0;
            if (sscanf(line, "%d %d %llx %llx %llu %d",
                       &s, &pid, &rel_off, &sz, &cfg, &is_daemon) >= 5) {
                if (s >= 0 && s < kMaxSlots) {
                    entries[s].pid = pid;
                    entries[s].rel_offset = rel_off;
                    entries[s].size_bytes = sz;
                    entries[s].cfg_gb = cfg;
                    entries[s].is_daemon = is_daemon;
                }
            }
        }
        fclose(fp);
    }
    // Prune any slot that is no longer locked
    for (int s = 0; s < kMaxSlots; ++s) {
        if (entries[s].pid > 0 && !is_slot_locked(s)) {
            entries[s].pid = 0;
            entries[s].rel_offset = 0;
            entries[s].size_bytes = 0;
            entries[s].is_daemon = 0;
        }
    }
}

static void save_hbm_state_locked(const HbmSlotEntry entries[kMaxSlots])
{
    char tmp_path[sizeof(g_run_dir) + 32];
    char final_path[sizeof(g_run_dir) + 32];
    run_path(tmp_path, sizeof(tmp_path), "hbm.state.tmp");
    run_path(final_path, sizeof(final_path), kHbmStateFile);

    int state_fd = open(tmp_path, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC | O_NOFOLLOW, 0600);
    FILE *fp = state_fd >= 0 ? fdopen(state_fd, "w") : nullptr;
    if (!fp) {
        if (state_fd >= 0)
            close(state_fd);
        return;
    }
    for (int s = 0; s < kMaxSlots; ++s) {
        fprintf(fp, "%d %d 0x%llx 0x%llx %llu %d\n",
                s,
                entries[s].pid,
                (unsigned long long)entries[s].rel_offset,
                (unsigned long long)entries[s].size_bytes,
                (unsigned long long)entries[s].cfg_gb,
                entries[s].is_daemon);
    }
    bool ok = (fflush(fp) == 0);
    if (fclose(fp) != 0)
        ok = false;
    if (ok)
        rename(tmp_path, final_path);
    else
        unlink(tmp_path);
}

static bool parse_strict_u64(const char *s, uint64_t &out)
{
    if (!s || !s[0] || s[0] == '-' || s[0] == '+' || isspace((unsigned char)s[0]))
        return false;
    errno = 0;
    char *end = nullptr;
    unsigned long long v = strtoull(s, &end, 10);
    if (errno != 0 || end == s || *end != '\0')
        return false;
    out = (uint64_t)v;
    return true;
}

static uint64_t determine_slot_hbm_cap_bytes()
{
    if (g_my_hbm_cap_bytes > 0)
        return g_my_hbm_cap_bytes;

    const char *is_daemon = getenv("HPU_MPS_DAEMON");
    if (is_daemon && is_daemon[0] == '1') {
        uint64_t daemon_bytes = kDaemonHbmCapBytes;
        const char *env_mb = getenv("HPU_MPS_DAEMON_HBM_MB");
        if (env_mb && env_mb[0]) {
            uint64_t mb = 0;
            constexpr uint64_t kMinMb = kDaemonHbmMinBytes >> 20;
            constexpr uint64_t kMaxMb = kDaemonHbmMaxBytes >> 20;
            if (parse_strict_u64(env_mb, mb) && mb >= kMinMb && mb <= kMaxMb) {
                daemon_bytes = align_up_dram_page(mb << 20);
            } else {
                fprintf(stderr, "[hpu-mps pid=%d] WARNING: ignoring invalid HPU_MPS_DAEMON_HBM_MB='%s' "
                                "(must be %llu..%llu MiB)\n",
                        (int)getpid(), env_mb,
                        (unsigned long long)kMinMb, (unsigned long long)kMaxMb);
            }
        }
        g_my_hbm_cap_bytes = daemon_bytes;
        return g_my_hbm_cap_bytes;
    }

    const char *env_gb = getenv("HPU_MPS_HBM_GB");
    if (env_gb && env_gb[0]) {
        uint64_t gb = 0;
        if (parse_strict_u64(env_gb, gb) && gb >= kMinClientHbmGb && gb <= kMaxClientHbmGb) {
            g_my_hbm_cap_bytes = gb << 30;
            return g_my_hbm_cap_bytes;
        }
    }

    int lock_fd = lock_run_file("hbm.lock");

    HbmSlotEntry entries[kMaxSlots];
    load_hbm_state_locked(entries);
    uint64_t cfg_gb = (g_slot >= 0 && g_slot < kMaxSlots) ? entries[g_slot].cfg_gb : 0;
    bool slot0_is_daemon = (entries[0].pid > 0 && entries[0].is_daemon != 0);

    unlock_run_file(lock_fd);

    if (cfg_gb >= kMinClientHbmGb && cfg_gb <= kMaxClientHbmGb) {
        g_my_hbm_cap_bytes = cfg_gb << 30;
    } else if (slot0_is_daemon && g_slot > 0) {
        g_my_hbm_cap_bytes = kDefaultDaemonClientGb << 30;
    } else {
        g_my_hbm_cap_bytes = kDefaultActiveSlotGb << 30;
    }
    return g_my_hbm_cap_bytes;
}

static uint64_t align_up_hbm(uint64_t x)
{
    if (x == 0)
        return kHbmAlignUnit;
    if (x > UINT64_MAX - (kHbmAlignUnit - 1))
        return UINT64_MAX;
    return ((x + kHbmAlignUnit - 1) / kHbmAlignUnit) * kHbmAlignUnit;
}

static uint64_t assign_slot_hbm_rel_offset(uint64_t size_bytes)
{
    if (g_my_hbm_offset_assigned)
        return g_my_hbm_rel_offset;

    uint64_t aligned_span = align_up_hbm(size_bytes);
    uint64_t my_phys = align_up_dram_page(size_bytes);
    if (aligned_span > kMaxHbmVaSpan || my_phys == UINT64_MAX)
        return kNoHbmRange;

    int lock_fd = lock_run_file("hbm.lock");

    HbmSlotEntry entries[kMaxSlots];
    load_hbm_state_locked(entries);

    bool slot0_recorded = (entries[0].pid > 0 && entries[0].size_bytes > 0);

    // Check actual physical HBM budget (each allocation rounds up to 768 MiB
    // DRAM pages in habanalabs) independently of 12 GiB VA alignment gaps.
    uint64_t total_phys = my_phys;
    if (g_slot > 0 && !slot0_recorded)
        total_phys += align_up_dram_page(kDaemonHbmCapBytes);
    for (int s = 0; s < kMaxSlots; ++s) {
        if (s == g_slot || entries[s].pid <= 0 || entries[s].size_bytes == 0)
            continue;
        uint64_t s_phys = align_up_dram_page(entries[s].size_bytes);
        if (s_phys > UINT64_MAX - total_phys) {
            total_phys = UINT64_MAX;
            break;
        }
        total_phys += s_phys;
    }
    if (g_card_hbm_bytes != 0 && total_phys > g_card_hbm_bytes) {
        unlock_run_file(lock_fd);
        fprintf(stderr, "[hpu-mps slot=%d pid=%d] ERROR: insufficient physical HBM for %llu MB "
                        "(%llu MB aligned; total %llu MB > card usable %llu MB); "
                        "lower -m/HPU_MPS_HBM_GB or the other slots' quotas\n",
                g_slot, (int)getpid(),
                (unsigned long long)(size_bytes >> 20),
                (unsigned long long)(my_phys >> 20),
                (unsigned long long)(total_phys >> 20),
                (unsigned long long)(g_card_hbm_bytes >> 20));
        return kNoHbmRange;
    }

    uint64_t candidate = 0;
    if (g_slot == 0) {
        candidate = 0;
    } else {
        // First-fit 12 GiB-aligned VA placement (preserves 768 MiB page alignment,
        // 1 GiB DMMU alignment, and lower 32-bit 0x50000000 ARC offset).
        while (candidate <= kMaxHbmVaSpan - aligned_span) {
            bool conflict = false;
            uint64_t cand_end = candidate + aligned_span;
            // Ensure we always reserve at least [0, 12 GiB) VA for Slot 0 if not yet recorded
            if (candidate == 0 && !slot0_recorded) {
                conflict = true;
                candidate = kHbmAlignUnit;
                continue;
            }
            for (int s = 0; s < kMaxSlots; ++s) {
                if (s == g_slot || entries[s].pid <= 0 || entries[s].size_bytes == 0)
                    continue;
                uint64_t s_start = entries[s].rel_offset;
                uint64_t s_span = align_up_hbm(entries[s].size_bytes);
                if (s_span == UINT64_MAX || s_start > kMaxHbmVaSpan || s_span > kMaxHbmVaSpan - s_start)
                    continue;
                uint64_t s_end = s_start + s_span;
                if (!(cand_end <= s_start || candidate >= s_end)) {
                    conflict = true;
                    candidate = align_up_hbm(s_end);
                    break;
                }
            }
            if (!conflict)
                break;
        }
    }

    if (candidate > kMaxHbmVaSpan - aligned_span) {
        unlock_run_file(lock_fd);
        fprintf(stderr, "[hpu-mps slot=%d pid=%d] ERROR: HBM VA span overflow (offset=0x%llx span=0x%llx)\n",
                g_slot, (int)getpid(), (unsigned long long)candidate, (unsigned long long)aligned_span);
        return kNoHbmRange;
    }

    const char *is_daemon = getenv("HPU_MPS_DAEMON");
    entries[g_slot].pid = (int)getpid();
    entries[g_slot].rel_offset = candidate;
    entries[g_slot].size_bytes = size_bytes;
    entries[g_slot].is_daemon = (is_daemon && is_daemon[0] == '1') ? 1 : 0;
    save_hbm_state_locked(entries);

    unlock_run_file(lock_fd);

    g_my_hbm_rel_offset = candidate;
    g_my_hbm_offset_assigned = true;
    return g_my_hbm_rel_offset;
}

static void release_slot_hbm_state()
{
    if (g_slot < 0 || g_slot >= kMaxSlots)
        return;
    int lock_fd = lock_run_file("hbm.lock");

    HbmSlotEntry entries[kMaxSlots];
    load_hbm_state_locked(entries);
    if (entries[g_slot].pid == (int)getpid()) {
        entries[g_slot].pid = 0;
        entries[g_slot].rel_offset = 0;
        entries[g_slot].size_bytes = 0;
        entries[g_slot].is_daemon = 0;
        save_hbm_state_locked(entries);
    }

    unlock_run_file(lock_fd);
}

struct LinuxDirent64 {
    uint64_t d_ino;
    int64_t d_off;
    unsigned short d_reclen;
    unsigned char d_type;
    char d_name[1];
};

static int open_eintr(const char *path, int flags)
{
    for (;;) {
        int fd = open(path, flags);
        if (fd >= 0 || errno != EINTR)
            return fd;
    }
}

static int fstat_eintr(int fd, struct stat *st)
{
    for (;;) {
        int rc = fstat(fd, st);
        if (rc == 0 || errno != EINTR)
            return rc;
    }
}

static void record_ctrl_rdev_for_minor(unsigned int min_num)
{
    char ctrl_path[64];
    snprintf(ctrl_path, sizeof(ctrl_path), "/dev/accel/accel_controlD%u", min_num);
    struct stat cst = {};
    if (stat(ctrl_path, &cst) == 0 && S_ISCHR(cst.st_mode)) {
        __atomic_store_n(&g_accel_ctrl_rdev, cst.st_rdev, __ATOMIC_RELEASE);
        __atomic_store_n(&g_has_accel_ctrl_rdev, true, __ATOMIC_RELEASE);
    } else if (stat("/dev/accel/accel_controlD0", &cst) == 0 && S_ISCHR(cst.st_mode)) {
        __atomic_store_n(&g_accel_ctrl_rdev, cst.st_rdev, __ATOMIC_RELEASE);
        __atomic_store_n(&g_has_accel_ctrl_rdev, true, __ATOMIC_RELEASE);
    } else if (stat("/dev/hl_controlD0", &cst) == 0 && S_ISCHR(cst.st_mode)) {
        __atomic_store_n(&g_accel_ctrl_rdev, cst.st_rdev, __ATOMIC_RELEASE);
        __atomic_store_n(&g_has_accel_ctrl_rdev, true, __ATOMIC_RELEASE);
    }
}

// Pre-populate exact dev_t values before hlthunk_open() so a concurrent fork()
// between real_open() and record_accel_fd_info() still detaches accel fds.
static void probe_accel_rdevs_from_path()
{
    if (__atomic_load_n(&g_has_accel_dev_rdev, __ATOMIC_ACQUIRE))
        return;
    struct stat st = {};
    if (stat("/dev/accel/accel0", &st) == 0 && S_ISCHR(st.st_mode)) {
        __atomic_store_n(&g_accel_dev_rdev, st.st_rdev, __ATOMIC_RELEASE);
        __atomic_store_n(&g_has_accel_dev_rdev, true, __ATOMIC_RELEASE);
        record_ctrl_rdev_for_minor(minor(st.st_rdev));
    } else if (stat("/dev/hl0", &st) == 0 && S_ISCHR(st.st_mode)) {
        __atomic_store_n(&g_accel_dev_rdev, st.st_rdev, __ATOMIC_RELEASE);
        __atomic_store_n(&g_has_accel_dev_rdev, true, __ATOMIC_RELEASE);
        record_ctrl_rdev_for_minor(minor(st.st_rdev));
    }
}

static void record_accel_fd_info(int fd)
{
    if (fd < 0)
        return;
    __atomic_store_n(&g_slot_accel_fd, fd, __ATOMIC_RELEASE);
    struct stat st = {};
    if (fstat_eintr(fd, &st) == 0 && S_ISCHR(st.st_mode)) {
        __atomic_store_n(&g_accel_dev_rdev, st.st_rdev, __ATOMIC_RELEASE);
        __atomic_store_n(&g_has_accel_dev_rdev, true, __ATOMIC_RELEASE);
        record_ctrl_rdev_for_minor(minor(st.st_rdev));
    }
}

static bool is_compute_accel_fd(int fd)
{
    if (fd < 0 || !__atomic_load_n(&g_has_accel_dev_rdev, __ATOMIC_ACQUIRE))
        return false;
    struct stat st = {};
    if (fstat_eintr(fd, &st) != 0 || !S_ISCHR(st.st_mode))
        return false;
    return st.st_rdev == __atomic_load_n(&g_accel_dev_rdev, __ATOMIC_ACQUIRE);
}

static bool is_accel_fd(int fd)
{
    if (fd < 0)
        return false;
    bool has_dev = __atomic_load_n(&g_has_accel_dev_rdev, __ATOMIC_ACQUIRE);
    bool has_ctrl = __atomic_load_n(&g_has_accel_ctrl_rdev, __ATOMIC_ACQUIRE);
    if (!has_dev && !has_ctrl)
        return false;
    struct stat st = {};
    if (fstat_eintr(fd, &st) != 0 || !S_ISCHR(st.st_mode))
        return false;
    if (has_dev && st.st_rdev == __atomic_load_n(&g_accel_dev_rdev, __ATOMIC_ACQUIRE))
        return true;
    if (has_ctrl && st.st_rdev == __atomic_load_n(&g_accel_ctrl_rdev, __ATOMIC_ACQUIRE))
        return true;
    return false;
}

// Async-signal-safe helper: replace an inherited /dev/accel/* fd with /dev/null
// so the kernel compute_ctx refcount drops immediately while the fd number stays
// occupied (preventing newly opened child files from colliding with Synapse's stale fd).
static void detach_inherited_fd(int fd, int devnull_fd)
{
    if (fd < 0 || fd == devnull_fd)
        return;
    if (devnull_fd >= 0) {
        for (;;) {
            if (dup3(devnull_fd, fd, O_CLOEXEC) >= 0)
                return;
            if (errno != EINTR)
                break;
        }
    }
    close(fd);
}

static void atfork_prepare()
{
    lock_internal_mutexes_for_fork();
}

static void atfork_parent()
{
    unlock_internal_mutexes_after_fork(false);
}

// A forked child (e.g. PyTorch DataLoader pt_data_worker) inherits the parent's
// slot lock and /dev/accel/* fds. Drop all of them so the child cannot pin
// compute_ctx across parent exit / hardware reset, and let the child acquire its
// own slot if it ever initializes HPU independently. Must stay async-signal-safe.
static void forget_inherited_slot()
{
    int saved_errno = errno;

    int slot_fd = __atomic_load_n(&g_slot_accel_fd, __ATOMIC_ACQUIRE);
    int primary_fd = __atomic_load_n(&g_primary_accel_fd, __ATOMIC_ACQUIRE);
    int listen_fd = __atomic_load_n(&g_fd_server_listen_fd, __ATOMIC_ACQUIRE);
    bool has_state = (g_slot >= 0 || g_slot_lock_fd >= 0 || slot_fd >= 0 ||
                      primary_fd >= 0 || listen_fd >= 0);

    if (!has_state) {
        unlock_internal_mutexes_after_fork(false);
        errno = saved_errno;
        return;
    }

    // Close our private fds first to free fd table headroom under EMFILE.
    if (g_slot_lock_fd >= 0) {
        close(g_slot_lock_fd);
        g_slot_lock_fd = -1;
    }
    if (listen_fd >= 0) {
        close(listen_fd);
        __atomic_store_n(&g_fd_server_listen_fd, -1, __ATOMIC_RELEASE);
    }
    if (primary_fd >= 0) {
        // When primary_fd is our private F_DUPFD_CLOEXEC copy (distinct from
        // slot_fd), Synapse does not know its fd number, so close it directly.
        if (primary_fd != slot_fd && is_accel_fd(primary_fd))
            close(primary_fd);
        __atomic_store_n(&g_primary_accel_fd, -1, __ATOMIC_RELEASE);
    }
    g_fd_server_started = false;

    int devnull_fd = open_eintr("/dev/null", O_RDONLY | O_CLOEXEC);
    if (slot_fd >= 0) {
        if (is_accel_fd(slot_fd))
            detach_inherited_fd(slot_fd, devnull_fd);
        __atomic_store_n(&g_slot_accel_fd, -1, __ATOMIC_RELEASE);
    }

    if (__atomic_load_n(&g_has_accel_dev_rdev, __ATOMIC_ACQUIRE) ||
        __atomic_load_n(&g_has_accel_ctrl_rdev, __ATOMIC_ACQUIRE)) {
        int dir_fd = open_eintr("/proc/self/fd", O_RDONLY | O_DIRECTORY | O_CLOEXEC);
        if (dir_fd >= 0) {
            alignas(LinuxDirent64) char buf[4096];
            constexpr size_t kNameOff = offsetof(LinuxDirent64, d_name);
            for (;;) {
                long nread = syscall(SYS_getdents64, dir_fd, buf, sizeof(buf));
                if (nread < 0) {
                    if (errno == EINTR)
                        continue;
                    break;
                }
                if (nread == 0)
                    break;
                for (long pos = 0; pos + (long)kNameOff <= nread;) {
                    LinuxDirent64 *d = (LinuxDirent64 *)(buf + pos);
                    if (d->d_reclen < kNameOff || pos + (long)d->d_reclen > nread)
                        break;
                    size_t max_name_len = (size_t)d->d_reclen - kNameOff;
                    pos += (long)d->d_reclen;

                    const char *p = d->d_name;
                    int fd = 0;
                    size_t i = 0;
                    bool ok = false;
                    for (; i < max_name_len && p[i] != '\0'; ++i) {
                        if (p[i] < '0' || p[i] > '9' || fd > 100000000) {
                            ok = false;
                            break;
                        }
                        fd = fd * 10 + (p[i] - '0');
                        ok = true;
                    }
                    if (!ok || i == max_name_len || p[i] != '\0' || fd == dir_fd || fd == devnull_fd)
                        continue;
                    if (is_accel_fd(fd))
                        detach_inherited_fd(fd, devnull_fd);
                }
            }
            close(dir_fd);
        } else {
            // Bounded fallback if /proc/self/fd is unavailable (e.g. EMFILE or no /proc).
            for (int fd = 0; fd < 1024; ++fd) {
                if (fd != devnull_fd && is_accel_fd(fd))
                    detach_inherited_fd(fd, devnull_fd);
            }
        }
    }

    if (devnull_fd >= 0)
        close(devnull_fd);

    g_slot = -1;
    g_slot_owner_pid = -1;
    g_my_hbm_cap_bytes = 0;
    g_my_hbm_rel_offset = 0;
    g_my_hbm_offset_assigned = false;
    __atomic_store_n(&g_cq_counters_host, nullptr, __ATOMIC_RELEASE);
    g_my_compute_cq_idx = 0xffffffffu;

    unlock_internal_mutexes_after_fork(true);
    errno = saved_errno;
}

static pthread_once_t g_atfork_once = PTHREAD_ONCE_INIT;

static void register_atfork()
{
    pthread_atfork(atfork_prepare, atfork_parent, forget_inherited_slot);
}

static void acquire_slot_if_needed()
{
    if (g_slot >= 0 && g_slot_owner_pid == getpid())
        return;

    const char *quiet = getenv("HPU_MPS_QUIET");
    if (quiet && quiet[0] == '1')
        g_verbose = false;

    const char *env_slot = getenv("HPU_MPS_SLOT");
    if (env_slot && env_slot[0] != '\0') {
        int requested = atoi(env_slot);
        if (requested >= 0 && requested < kMaxSlots) {
            char path[128];
            lock_path_for_slot(requested, path, sizeof(path));
            int fd = open(path, O_RDWR | O_CREAT | O_CLOEXEC | O_NOFOLLOW, 0600);
            bool locked = false;
            if (fd >= 0) {
                for (int retry = 0; retry < 50; ++retry) {
                    if (try_lock_slot_fd(fd)) {
                        locked = true;
                        break;
                    }
                    usleep(10000);
                }
            }
            if (locked) {
                g_slot = requested;
                g_slot_lock_fd = fd;
                g_slot_owner_pid = getpid();
            } else {
                if (fd >= 0)
                    close(fd);
                fprintf(stderr, "[hpu-mps pid=%d] ERROR: requested HPU_MPS_SLOT=%d is already in use\n",
                        (int)getpid(), requested);
                _exit(1);
            }
        }
    }

    if (g_slot < 0) {
        for (int s = 0; s < kMaxSlots; ++s) {
            char path[128];
            lock_path_for_slot(s, path, sizeof(path));
            int fd = open(path, O_RDWR | O_CREAT | O_CLOEXEC | O_NOFOLLOW, 0600);
            if (fd < 0)
                continue;
            if (try_lock_slot_fd(fd)) {
                g_slot = s;
                g_slot_lock_fd = fd;
                g_slot_owner_pid = getpid();
                break;
            }
            close(fd);
        }
    }

    pthread_once(&g_atfork_once, register_atfork);

    if (g_slot < 0) {
        fprintf(stderr, "[hpu-mps pid=%d] ERROR: all %d HPU MPS slots are busy\n",
                (int)getpid(), kMaxSlots);
        _exit(1);
    }

    if (g_slot_lock_fd >= 0) {
        if (ftruncate(g_slot_lock_fd, 0) == 0) {
            lseek(g_slot_lock_fd, 0, SEEK_SET);
            char buf[64];
            int len = snprintf(buf, sizeof(buf), "%d\n", (int)getpid());
            ssize_t w = write(g_slot_lock_fd, buf, len);
            (void)w;
        }
    }

    probe_accel_rdevs_from_path();

    if (g_slot == 0) {
        unlink_run_file(kReadyFile);
        set_coeff_table_state(kCoeffTablePending, (int32_t)getpid());
    }

    MPS_LOG("acquired HPU MPS slot %d (%s mode)",
            g_slot, g_slot == 0 ? "PRIMARY" : "SECONDARY ATTACH");
}

__attribute__((destructor))
static void hpu_mps_dtor()
{
    if (g_slot < 0 || g_slot_owner_pid != getpid())
        return;
    release_slot_hbm_state();
    if (g_slot == 0) {
        unlink_run_file(kReadyFile);
        set_coeff_table_state(kCoeffTableUnknown, 0);
    }
    if (g_slot_lock_fd >= 0) {
        close(g_slot_lock_fd);
        g_slot_lock_fd = -1;
    }
    g_slot = -1;
}

static void *open_loaded_lib(LoadedLib *lib, void **handle)
{
    if (*handle == nullptr) {
        char fallback[256];
        const char *path = lib->path;
        if (!path[0]) {
            snprintf(fallback, sizeof(fallback), "/usr/lib/habanalabs/%s", lib->soname);
            path = fallback;
        }
        *handle = dlopen(path, RTLD_LAZY | RTLD_NOLOAD);
    }
    return *handle;
}

static pthread_mutex_t g_sym_resolve_mu = PTHREAD_MUTEX_INITIALIZER;

static void *resolve_real_sym(const char *name)
{
    void *fn = dlsym(RTLD_NEXT, name);
    if (fn != nullptr)
        return fn;

    // When loaded via dlopen(..., RTLD_LOCAL) (e.g. Python, nvtop, ctypes),
    // libSynapse.so, libscal.so, libhl-thunk.so, and libhlml.so (which embeds
    // hlthunk_* symbols) live in a local link-map scope invisible to RTLD_NEXT.
    static void *h_synapse = nullptr;
    static void *h_scal    = nullptr;
    static void *h_hlthunk = nullptr;
    static void *h_hlml    = nullptr;
    pthread_once(&g_mps_once, decide_mps_active);

    pthread_mutex_lock(&g_sym_resolve_mu);
    // Refresh in case a library (such as libhlml.so) was dlopen'd after init.
    dl_iterate_phdr(collect_loaded_lib, nullptr);

    struct {
        LoadedLib *lib;
        void **handle;
    } order[] = {
        {&g_lib_hlthunk, &h_hlthunk},
        {&g_lib_hlml, &h_hlml},
        {&g_lib_scal, &h_scal},
        {&g_lib_synapse, &h_synapse},
    };
    // hlthunk_* first (libhl-thunk or libhlml); everything else in libscal/libSynapse.
    size_t first = strncmp(name, "hlthunk_", 8) == 0 ? 0 : 2;
    for (size_t i = 0; i < 4; ++i) {
        auto &o = order[(first + i) % 4];
        void *h = open_loaded_lib(o.lib, o.handle);
        if (h != nullptr && (fn = dlsym(h, name)) != nullptr) {
            pthread_mutex_unlock(&g_sym_resolve_mu);
            return fn;
        }
    }
    pthread_mutex_unlock(&g_sym_resolve_mu);

    fprintf(stderr, "[hpu-mps pid=%d] FATAL: failed to resolve symbol '%s'\n",
            (int)getpid(), name);
    abort();
}

static bool wait_for_slot0_ready()
{
    if (g_slot <= 0 || is_slot0_ready())
        return true;

    int wait_sec = 30;
    const char *env_wait = getenv("HPU_MPS_WAIT_SEC");
    if (env_wait && env_wait[0])
        wait_sec = atoi(env_wait);

    MPS_LOG("waiting up to %ds for Primary (slot 0) to complete hardware bringup...", wait_sec);
    int Loops = wait_sec * 100; // 10ms intervals
    for (int i = 0; i < Loops; ++i) {
        if (is_slot0_ready()) {
            MPS_LOG("Primary (slot 0) is ready; proceeding with Secondary Attach");
            return true;
        }
        usleep(10000);
    }
    fprintf(stderr, "[hpu-mps slot=%d pid=%d] ERROR: Primary (slot 0) not ready after %ds\n",
            g_slot, (int)getpid(), wait_sec);
    return false;
}

static void fd_sock_addr(struct sockaddr_un *addr)
{
    memset(addr, 0, sizeof(*addr));
    addr->sun_family = AF_UNIX;
    run_path(addr->sun_path, sizeof(addr->sun_path), kFdSockFile);
}

// Defense in depth on top of the 0700 run dir: only talk to our own uid.
static bool peer_is_same_user(int sfd)
{
    struct ucred cred = {};
    socklen_t len = sizeof(cred);
    return getsockopt(sfd, SOL_SOCKET, SO_PEERCRED, &cred, &len) == 0 && cred.uid == geteuid();
}

static void stop_fd_server()
{
    int listen_fd = __atomic_exchange_n(&g_fd_server_listen_fd, -1, __ATOMIC_ACQ_REL);
    if (listen_fd < 0)
        return;
    shutdown(listen_fd, SHUT_RDWR); // wakes the accept() in the server thread
    close(listen_fd);
    unlink_run_file(kFdSockFile);
}

static void *slot0_fd_server_thread(void *arg)
{
    int listen_fd = (int)(intptr_t)arg;
    while (listen_fd >= 0 && __atomic_load_n(&g_primary_accel_fd, __ATOMIC_ACQUIRE) >= 0) {
        int client_fd = accept(listen_fd, nullptr, nullptr);
        if (client_fd < 0) {
            if (errno == EINTR)
                continue;
            break;
        }
        if (!peer_is_same_user(client_fd)) {
            close(client_fd);
            continue;
        }
        int target_fd = __atomic_load_n(&g_primary_accel_fd, __ATOMIC_ACQUIRE);
        if (target_fd >= 0) {
            struct msghdr msg = {};
            char iov_byte = 'F';
            struct iovec iov = { &iov_byte, 1 };
            msg.msg_iov = &iov;
            msg.msg_iovlen = 1;

            union {
                char buf[CMSG_SPACE(sizeof(int))];
                struct cmsghdr align;
            } u = {};
            msg.msg_control = u.buf;
            msg.msg_controllen = sizeof(u.buf);

            struct cmsghdr *cmsg = CMSG_FIRSTHDR(&msg);
            cmsg->cmsg_level = SOL_SOCKET;
            cmsg->cmsg_type = SCM_RIGHTS;
            cmsg->cmsg_len = CMSG_LEN(sizeof(int));
            memcpy(CMSG_DATA(cmsg), &target_fd, sizeof(int));

            (void)sendmsg(client_fd, &msg, MSG_NOSIGNAL);
        }
        close(client_fd);
    }
    return nullptr;
}

static void start_slot0_fd_server_if_needed(int accel_fd)
{
    if (g_fd_server_started || accel_fd < 0)
        return;

    int dup_fd = fcntl(accel_fd, F_DUPFD_CLOEXEC, 3);
    int master_fd = (dup_fd >= 0) ? dup_fd : accel_fd;
    __atomic_store_n(&g_primary_accel_fd, master_fd, __ATOMIC_RELEASE);

    int sfd = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (sfd < 0)
        return;

    struct sockaddr_un addr;
    fd_sock_addr(&addr);
    unlink(addr.sun_path);

    if (bind(sfd, (struct sockaddr *)&addr, sizeof(addr)) != 0 ||
        listen(sfd, 16) != 0) {
        close(sfd);
        return;
    }

    __atomic_store_n(&g_fd_server_listen_fd, sfd, __ATOMIC_RELEASE);
    g_fd_server_started = true;

    pthread_t tid;
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
    if (pthread_create(&tid, &attr, slot0_fd_server_thread, (void *)(intptr_t)sfd) == 0) {
        MPS_LOG("started SCM_RIGHTS FD server on %s (master fd=%d)",
                addr.sun_path, master_fd);
    }
    pthread_attr_destroy(&attr);
}

static int recv_accel_fd_from_slot0()
{
    for (int attempt = 0; attempt < 100; ++attempt) {
        int sfd = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
        if (sfd < 0)
            return -1;

        struct sockaddr_un addr;
        fd_sock_addr(&addr);

        if (connect(sfd, (struct sockaddr *)&addr, sizeof(addr)) == 0 && peer_is_same_user(sfd)) {
            struct msghdr msg = {};
            char iov_byte = 0;
            struct iovec iov = { &iov_byte, 1 };
            msg.msg_iov = &iov;
            msg.msg_iovlen = 1;

            union {
                char buf[CMSG_SPACE(sizeof(int))];
                struct cmsghdr align;
            } u = {};
            msg.msg_control = u.buf;
            msg.msg_controllen = sizeof(u.buf);

            ssize_t n = recvmsg(sfd, &msg, 0);
            close(sfd);
            if (n > 0) {
                struct cmsghdr *cmsg = CMSG_FIRSTHDR(&msg);
                if (cmsg && cmsg->cmsg_level == SOL_SOCKET &&
                    cmsg->cmsg_type == SCM_RIGHTS &&
                    cmsg->cmsg_len == CMSG_LEN(sizeof(int))) {
                    int received_fd = -1;
                    memcpy(&received_fd, CMSG_DATA(cmsg), sizeof(int));
                    if (received_fd >= 0) {
                        int flags = fcntl(received_fd, F_GETFD);
                        if (flags >= 0)
                            fcntl(received_fd, F_SETFD, flags | FD_CLOEXEC);
                        return received_fd;
                    }
                }
            }
        } else {
            close(sfd);
        }
        usleep(10000);
    }
    return -1;
}

static void reclaim_slot_resources_on_shared_fd(int shared_fd, int slot);
static void record_slot_dram_handle(uint64_t handle);
static void remove_slot_dram_handle(uint64_t handle);
static void record_slot_mapped_va(uint64_t va);
static void remove_slot_mapped_va(uint64_t va);
static bool slot_owns_mapped_va(uint64_t va);

extern "C" int hlthunk_open(int device_name, const char *busid)
{
    if (is_mps_disabled()) {
        using fn_t = int (*)(int, const char *);
        static fn_t real_fn = (fn_t)resolve_real_sym("hlthunk_open");
        return real_fn(device_name, busid);
    }

    acquire_slot_if_needed();
    if (g_slot > 0) {
        // The driver allows one compute context, so a secondary can only run on
        // the primary's fd; opening the device itself would fail or hang.
        if (!wait_for_slot0_ready()) {
            errno = ENODEV;
            return -1;
        }
        int shared_fd = recv_accel_fd_from_slot0();
        if (shared_fd < 0) {
            fprintf(stderr, "[hpu-mps slot=%d pid=%d] ERROR: no device fd from Primary (slot 0)\n",
                    g_slot, (int)getpid());
            errno = ENODEV;
            return -1;
        }
        MPS_LOG("hlthunk_open: received shared device fd=%d from Primary (slot 0)", shared_fd);
        record_accel_fd_info(shared_fd);
        // Only our own slot: we hold its lock, so no live process can be using it.
        reclaim_slot_resources_on_shared_fd(shared_fd, g_slot);
        return shared_fd;
    }

    int open_lock_fd = lock_run_file("open.lock");

    using fn_t = int (*)(int, const char *);
    static fn_t real_fn = (fn_t)resolve_real_sym("hlthunk_open");
    int fd = real_fn(device_name, busid);

    if (g_slot == 0 && fd >= 0) {
        start_slot0_fd_server_if_needed(fd);
    }

    unlock_run_file(open_lock_fd);
    if (fd >= 0)
        record_accel_fd_info(fd);
    return fd;
}

extern "C" int hlthunk_close(int fd)
{
    if (!is_mps_disabled() && fd >= 0) {
        int slot_fd = __atomic_load_n(&g_slot_accel_fd, __ATOMIC_ACQUIRE);
        bool is_primary_compute = (slot_fd >= 0) ? (fd == slot_fd) : is_compute_accel_fd(fd);
        if (fd == slot_fd)
            __atomic_store_n(&g_slot_accel_fd, -1, __ATOMIC_RELEASE);

        if (g_slot == 0 && is_primary_compute) {
            // Secondaries lock their slot before asking for the fd, so once the server is
            // gone every process that can still hold the fd is visible to the wait below.
            unlink_run_file(kReadyFile);
            stop_fd_server();
            if (any_secondary_slot_alive()) {
                MPS_LOG("hlthunk_close: waiting for active Secondary slots before closing Primary fd...");
                while (any_secondary_slot_alive()) {
                    usleep(50000);
                }
            }
            int primary_fd = __atomic_exchange_n(&g_primary_accel_fd, -1, __ATOMIC_ACQ_REL);
            if (primary_fd >= 0 && primary_fd != fd) {
                close(primary_fd);
            }
        }
    }
    using fn_t = int (*)(int);
    static fn_t real_fn = (fn_t)resolve_real_sym("hlthunk_close");
    return real_fn(fd);
}

extern "C" int hlthunk_notifier_create(int fd)
{
    if (!is_mps_disabled() && g_slot > 0) {
        int efd = eventfd(0, EFD_CLOEXEC);
        if (efd >= 0)
            return efd;
        return -errno;
    }
    using fn_t = int (*)(int);
    static fn_t real_fn = (fn_t)resolve_real_sym("hlthunk_notifier_create");
    int rc = real_fn(fd);
    if (rc < 0 && !is_mps_disabled()) {
        int efd = eventfd(0, EFD_CLOEXEC);
        if (efd >= 0)
            return efd;
    }
    return rc;
}

extern "C" int hlthunk_notifier_release(int fd, int handle)
{
    if (!is_mps_disabled() && g_slot > 0) {
        if (handle >= 0)
            close(handle);
        return 0;
    }
    using fn_t = int (*)(int, int);
    static fn_t real_fn = (fn_t)resolve_real_sym("hlthunk_notifier_release");
    return real_fn(fd, handle);
}

extern "C" int _ZN4Scal5setupEv(void *self)
{
    if (!is_mps_disabled())
        acquire_slot_if_needed();
    using fn_t = int (*)(void *);
    static fn_t real_fn = (fn_t)resolve_real_sym("_ZN4Scal5setupEv");
    return real_fn(self);
}

// ============================================================================
// 1. Memory Pool Partitioning & Dynamic HBM Relocation
// ============================================================================

extern "C" int _ZN4Scal11fillRegionsEj(void *self, unsigned int region)
{
    if (is_mps_disabled()) {
        using fn_t = int (*)(void *, unsigned int);
        static fn_t real_fn = (fn_t)resolve_real_sym("_ZN4Scal11fillRegionsEj");
        return real_fn(self, region);
    }
    (void)self;
    (void)region;
    acquire_slot_if_needed();
    return 0;
}

extern "C" void *_ZSt18_Rb_tree_incrementPSt18_Rb_tree_node_base(void *node);

static constexpr uint64_t kHbmSharedSliceSize    = 128ULL << 20; // Full 128 MiB per slot
static constexpr uint64_t kHbmBinaryTotalReserve = 128ULL << 20; // 128 MiB in Slot 0 [0x50000000, 0x58000000)

struct Slot0PoolBases {
    uint64_t magic;
    uint64_t hbm_shared_dev;
    uint64_t hbm_shared_base;
    uint32_t hbm_shared_core;
    uint32_t pad0;
};

static constexpr const char *kSlot0PoolsFile = "pools-slot0.bin";

struct ScalPoolOffsets {
    size_t off_total_hbm = 0x28;
    size_t off_core_addr = 0x28;
    size_t off_dev_addr  = 0x30;
    size_t off_dev_base  = 0x38;
    size_t off_size      = 0x50;
    bool calibrated_post = false;
};

static ScalPoolOffsets g_pool_offs;
static bool g_secondary_attach_ok = false;

struct ScalMemoryPoolInfoV2Probe {
    void       *scal;
    const char *name;
    uint32_t    idx;
    uint32_t    pad0;
    uint64_t    device_base_address;
    void       *host_base_address;
    uint32_t    core_base_address;
    uint32_t    pad1;
    uint64_t    totalSize;
    uint64_t    freeSize;
    uint64_t    device_base_allocated_address;
};

static char *find_scal_pool(void *self, const char *target_name)
{
    if (!self || !target_name)
        return nullptr;

    // 1. Prefer libscal.so's own exported Scal::getPoolByName(std::string const&) const
    using get_pool_fn_t = char *(*)(const void *, const std::string &);
    static get_pool_fn_t real_get_pool =
        (get_pool_fn_t)resolve_real_sym("_ZNK4Scal13getPoolByNameERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE");
    if (real_get_pool) {
        std::string name_str(target_name);
        char *pool = real_get_pool(self, name_str);
        if (pool)
            return pool;
    }

    // 2. Fallback to RB-tree walk
    size_t target_len = strlen(target_name);
    char *end_node = (char *)self + 0x220;
    char *cur_node = *(char **)((char *)self + 0x230);
    while (cur_node && cur_node != end_node) {
        const char *key_ptr = *(const char **)(cur_node + 0x20);
        size_t key_len = *(size_t *)(cur_node + 0x28);
        if (key_ptr && key_len == target_len && memcmp(key_ptr, target_name, target_len) == 0) {
            return cur_node + 0x40;
        }
        cur_node = (char *)_ZSt18_Rb_tree_incrementPSt18_Rb_tree_node_base(cur_node);
    }
    return nullptr;
}

static bool is_plausible_hbm_bytes(uint64_t val)
{
    return val >= (16ULL << 30) && val <= (256ULL << 30) && ((val & 0x7ffffffULL) == 0);
}

static size_t calibrate_total_hbm_offset(void *self)
{
    uint64_t cur = *(uint64_t *)((char *)self + g_pool_offs.off_total_hbm);
    if (is_plausible_hbm_bytes(cur))
        return g_pool_offs.off_total_hbm;
    for (size_t off = 0x10; off <= 0x80; off += 8) {
        uint64_t v = *(uint64_t *)((char *)self + off);
        if (is_plausible_hbm_bytes(v)) {
            g_pool_offs.off_total_hbm = off;
            return off;
        }
    }
    return g_pool_offs.off_total_hbm;
}

static void calibrate_pool_size_offset_pre_init(char *pool_binary, char *pool_shared)
{
    if (!pool_binary || !pool_shared)
        return;
    if (*(uint64_t *)(pool_binary + g_pool_offs.off_size) == (32ULL << 20) &&
        *(uint64_t *)(pool_shared + g_pool_offs.off_size) == (128ULL << 20)) {
        return;
    }
    for (size_t off = 0x20; off <= 0x80; off += 8) {
        if (*(uint64_t *)(pool_binary + off) == (32ULL << 20) &&
            *(uint64_t *)(pool_shared + off) == (128ULL << 20)) {
            g_pool_offs.off_size = off;
            return;
        }
    }
}

static void calibrate_pool_offsets_post_init(char *pool_shared)
{
    if (!pool_shared || g_pool_offs.calibrated_post)
        return;
    using info_v2_fn_t = int (*)(void *, ScalMemoryPoolInfoV2Probe *);
    static info_v2_fn_t real_info_v2 = (info_v2_fn_t)resolve_real_sym("scal_pool_get_infoV2");
    if (!real_info_v2)
        return;

    ScalMemoryPoolInfoV2Probe info = {};
    if (real_info_v2(pool_shared, &info) != 0)
        return;

    for (size_t off = 0x18; off <= 0x70; off += 4) {
        if (info.core_base_address != 0 &&
            *(uint32_t *)(pool_shared + off) == info.core_base_address &&
            ((off + 4 <= 0x70 && *(uint32_t *)(pool_shared + off + 4) == 0) || off == g_pool_offs.off_core_addr)) {
            g_pool_offs.off_core_addr = off;
            break;
        }
    }
    for (size_t off = 0x18; off <= 0x70; off += 8) {
        uint64_t v = *(uint64_t *)(pool_shared + off);
        if (info.device_base_address != 0 && v == info.device_base_address) {
            g_pool_offs.off_dev_addr = off;
        } else if (info.device_base_allocated_address != 0 && v == info.device_base_allocated_address) {
            g_pool_offs.off_dev_base = off;
        } else if (info.totalSize != 0 && v == info.totalSize) {
            g_pool_offs.off_size = off;
        }
    }
    g_pool_offs.calibrated_post = true;
}

extern "C" int _ZN11Scal_Gaudi215initMemoryPoolsEv(void *self)
{
    if (is_mps_disabled()) {
        using fn_t = int (*)(void *);
        static fn_t real_fn = (fn_t)resolve_real_sym("_ZN11Scal_Gaudi215initMemoryPoolsEv");
        return real_fn(self);
    }

    acquire_slot_if_needed();
    if (g_slot > 0 && !wait_for_slot0_ready())
        return 1;

    uint64_t hbm_cap = determine_slot_hbm_cap_bytes();
    size_t off_total_hbm = calibrate_total_hbm_offset(self);
    uint64_t *p_total_hbm = (uint64_t *)((char *)self + off_total_hbm);
    uint64_t orig_hbm = *p_total_hbm;
    g_card_hbm_bytes = orig_hbm;
    if (assign_slot_hbm_rel_offset(hbm_cap) == kNoHbmRange)
        return 1;
    if (orig_hbm > hbm_cap) {
        *p_total_hbm = hbm_cap;
        MPS_LOG("initMemoryPools: configured slot-%d HBM pool size to %llu MB (phys %llu MB, card usable %llu MB)",
                g_slot,
                (unsigned long long)(hbm_cap >> 20),
                (unsigned long long)(align_up_dram_page(hbm_cap) >> 20),
                (unsigned long long)(orig_hbm >> 20));
    }

    char *pool_shared = find_scal_pool(self, "hbm_shared");
    char *pool_binary = find_scal_pool(self, "hbm_binary");
    calibrate_pool_size_offset_pre_init(pool_binary, pool_shared);
    if (g_slot == 0 && pool_binary) {
        *(uint64_t *)(pool_binary + g_pool_offs.off_size) = kHbmBinaryTotalReserve;
    }

    using fn_t = int (*)(void *);
    static fn_t real_fn = (fn_t)resolve_real_sym("_ZN11Scal_Gaudi215initMemoryPoolsEv");
    int rc = real_fn(self);
    if (rc != 0)
        return rc;

    pool_shared = find_scal_pool(self, "hbm_shared");
    calibrate_pool_offsets_post_init(pool_shared);

    if (g_slot == 0) {
        // Reserve 256 MiB immediately following Slot 0's hbm_shared ([0x60000000, 0x70000000))
        // from global_hbm so Slot 1 and Slot 2 each get a dedicated 128 MiB hbm_shared region
        // inside ARC region 6 (0x60000000..0x6fffffff).
        char *pool_global = find_scal_pool(self, "global_hbm");
        int arc = -1;
        if (pool_global) {
            using alloc_fn_t = int (*)(void *, uint64_t, void **);
            static alloc_fn_t real_alloc = (alloc_fn_t)resolve_real_sym("scal_allocate_buffer");
            void *region6_reserve = nullptr;
            arc = real_alloc(pool_global, (kMaxSlots - 1) * kHbmSharedSliceSize, &region6_reserve);
        }
        g_secondary_attach_ok = (arc == 0);
        if (g_secondary_attach_ok) {
            MPS_LOG("initMemoryPools (Primary): reserved %llu MB in ARC region 6 from global_hbm",
                    (unsigned long long)(((kMaxSlots - 1) * kHbmSharedSliceSize) >> 20));
        } else {
            fprintf(stderr, "[hpu-mps slot=0 pid=%d] ERROR: cannot reserve the secondaries' hbm_shared "
                            "(rc=%d); running as a single process\n", (int)getpid(), arc);
        }

        Slot0PoolBases bases = {};
        bases.magic           = 0x4850554d50535030ULL;
        bases.hbm_shared_dev  = pool_shared ? *(uint64_t *)(pool_shared + g_pool_offs.off_dev_addr)  : 0;
        bases.hbm_shared_base = pool_shared ? *(uint64_t *)(pool_shared + g_pool_offs.off_dev_base)  : 0;
        bases.hbm_shared_core = pool_shared ? *(uint32_t *)(pool_shared + g_pool_offs.off_core_addr) : 0;

        int fd = open_run_file(kSlot0PoolsFile, O_WRONLY | O_CREAT | O_TRUNC);
        if (fd >= 0) {
            ssize_t wn = write(fd, &bases, sizeof(bases));
            (void)wn;
            close(fd);
        }

        if (pool_shared) {
            // Temporarily set pool_shared->size = 384 MiB during configSchedulers/configEngines
            // so createCoreConfigQmanProgram maps both ARC region 5 and ARC region 6.
            *(uint64_t *)(pool_shared + g_pool_offs.off_size) = (uint64_t)kMaxSlots * kHbmSharedSliceSize;
        }
        MPS_LOG("initMemoryPools (Primary): hbm_shared dev=0x%llx core=0x%x (%llu MB per slot, 384 MB ARC window)",
                (unsigned long long)bases.hbm_shared_dev,
                bases.hbm_shared_core,
                (unsigned long long)(kHbmSharedSliceSize >> 20));
    } else {
        Slot0PoolBases bases = {};
        int fd = open_run_file(kSlot0PoolsFile, O_RDONLY);
        if (fd >= 0) {
            ssize_t rn = read(fd, &bases, sizeof(bases));
            (void)rn;
            close(fd);
        }
        if (bases.magic != 0x4850554d50535030ULL) {
            bases.hbm_shared_dev  = 0x1001600158000000ULL;
            bases.hbm_shared_base = kHbmPoolVaBase;
            bases.hbm_shared_core = 0x58000000U;
        }
        if (pool_shared) {
            uint64_t off = (uint64_t)g_slot * kHbmSharedSliceSize;
            *(uint32_t *)(pool_shared + g_pool_offs.off_core_addr) = bases.hbm_shared_core + (uint32_t)off;
            *(uint64_t *)(pool_shared + g_pool_offs.off_dev_addr)  = bases.hbm_shared_dev + off;
            *(uint64_t *)(pool_shared + g_pool_offs.off_dev_base)  = bases.hbm_shared_base;
            *(uint64_t *)(pool_shared + g_pool_offs.off_size)      = kHbmSharedSliceSize;
        }
        MPS_LOG("initMemoryPools (Secondary slot %d): hbm_shared dev=0x%llx core=0x%x (%llu MB)",
                g_slot,
                pool_shared ? *(unsigned long long *)(pool_shared + g_pool_offs.off_dev_addr) : 0ULL,
                pool_shared ? *(uint32_t *)(pool_shared + g_pool_offs.off_core_addr) : 0U,
                (unsigned long long)(kHbmSharedSliceSize >> 20));
    }

    return 0;
}

extern "C" int _ZN11Scal_Gaudi219allocateHostPoolMemEmPPvPmm(
    void *self,
    unsigned long size,
    void **host_va,
    unsigned long *dev_va,
    unsigned long offset)
{
    if (is_mps_disabled()) {
        using fn_t = int (*)(void *, unsigned long, void **, unsigned long *, unsigned long);
        static fn_t real_fn = (fn_t)resolve_real_sym("_ZN11Scal_Gaudi219allocateHostPoolMemEmPPvPmm");
        return real_fn(self, size, host_va, dev_va, offset);
    }

    acquire_slot_if_needed();
    unsigned long orig_offset = offset;
    unsigned long new_offset = offset + (unsigned long)g_slot * kHostSharedSliceSize;
    size = kHostSharedSliceSize;
    dev_va[4] = kHostSharedSliceSize; // pool->size at 0x50(pool)
    MPS_LOG("allocateHostPoolMem(host_shared): offset 0x%lx -> 0x%lx, size=%lu MB",
            orig_offset, new_offset, size >> 20);

    using fn_t = int (*)(void *, unsigned long, void **, unsigned long *, unsigned long);
    static fn_t real_fn = (fn_t)resolve_real_sym("_ZN11Scal_Gaudi219allocateHostPoolMemEmPPvPmm");
    return real_fn(self, size, host_va, dev_va, new_offset);
}

extern "C" int _ZN11Scal_Gaudi218allocateHBMPoolMemEmPmS0_mbb(
    void *self,
    unsigned long size,
    unsigned long *handle,
    unsigned long *dev_va,
    unsigned long offset,
    bool shared,
    bool contiguous)
{
    if (is_mps_disabled()) {
        using fn_t = int (*)(void *, unsigned long, unsigned long *, unsigned long *, unsigned long, bool, bool);
        static fn_t real_fn = (fn_t)resolve_real_sym("_ZN11Scal_Gaudi218allocateHBMPoolMemEmPmS0_mbb");
        return real_fn(self, size, handle, dev_va, offset, shared, contiguous);
    }

    acquire_slot_if_needed();
    uint64_t rel_off = assign_slot_hbm_rel_offset((uint64_t)size);
    if (rel_off == kNoHbmRange)
        return 1;
    unsigned long new_offset = offset + (unsigned long)rel_off;
    MPS_LOG("allocateHBMPoolMem: offset 0x%lx -> 0x%lx (+%llu GB, size=%lu MB, shared=%d)",
            offset, new_offset, (unsigned long long)(rel_off >> 30), size >> 20, (int)shared);

    using fn_t = int (*)(void *, unsigned long, unsigned long *, unsigned long *, unsigned long, bool, bool);
    static fn_t real_fn = (fn_t)resolve_real_sym("_ZN11Scal_Gaudi218allocateHBMPoolMemEmPmS0_mbb");
    return real_fn(self, size, handle, dev_va, new_offset, shared, contiguous);
}

// POD mutex and fixed table to avoid C++ static constructors in global preload mode
static pthread_mutex_t g_dram_alloc_mu = PTHREAD_MUTEX_INITIALIZER;
struct DramHandleEntry {
    uint64_t handle;
    uint64_t size;
};
static DramHandleEntry g_dram_handles[256];
static uint64_t g_dram_va_bump = 0;

extern "C" uint64_t hlthunk_device_memory_alloc(
    int fd, uint64_t size, uint64_t page_size, bool contiguous, bool shared)
{
    if (is_mps_disabled()) {
        using fn_t = uint64_t (*)(int, uint64_t, uint64_t, bool, bool);
        static fn_t real_fn = (fn_t)resolve_real_sym("hlthunk_device_memory_alloc");
        return real_fn(fd, size, page_size, contiguous, shared);
    }

    acquire_slot_if_needed();
    using fn_t = uint64_t (*)(int, uint64_t, uint64_t, bool, bool);
    static fn_t real_fn = (fn_t)resolve_real_sym("hlthunk_device_memory_alloc");
    uint64_t handle = real_fn(fd, size, page_size, contiguous, shared);
    if (handle == 0) {
        MPS_LOG("ERROR: hlthunk_device_memory_alloc(size=%llu MB, page=%llu, contig=%d, shared=%d) failed (errno=%d)",
                (unsigned long long)(size >> 20), (unsigned long long)page_size, (int)contiguous, (int)shared, errno);
    } else {
        pthread_mutex_lock(&g_dram_alloc_mu);
        for (int i = 0; i < 256; ++i) {
            if (g_dram_handles[i].handle == 0 || g_dram_handles[i].handle == handle) {
                g_dram_handles[i].handle = handle;
                g_dram_handles[i].size = size;
                break;
            }
        }
        pthread_mutex_unlock(&g_dram_alloc_mu);
        record_slot_dram_handle(handle);
    }
    return handle;
}

extern "C" int hlthunk_device_memory_free(int fd, uint64_t handle)
{
    if (!is_mps_disabled() && handle != 0) {
        pthread_mutex_lock(&g_dram_alloc_mu);
        for (int i = 0; i < 256; ++i) {
            if (g_dram_handles[i].handle == handle) {
                g_dram_handles[i].handle = 0;
                g_dram_handles[i].size = 0;
                break;
            }
        }
        pthread_mutex_unlock(&g_dram_alloc_mu);
        remove_slot_dram_handle(handle);
    }
    using fn_t = int (*)(int, uint64_t);
    static fn_t real_fn = (fn_t)resolve_real_sym("hlthunk_device_memory_free");
    return real_fn(fd, handle);
}

extern "C" uint64_t hlthunk_device_memory_map(int fd, uint64_t handle, uint64_t hint_addr)
{
    if (is_mps_disabled()) {
        using fn_t = uint64_t (*)(int, uint64_t, uint64_t);
        static fn_t real_fn = (fn_t)resolve_real_sym("hlthunk_device_memory_map");
        return real_fn(fd, handle, hint_addr);
    }

    acquire_slot_if_needed();
    using fn_t = uint64_t (*)(int, uint64_t, uint64_t);
    static fn_t real_fn = (fn_t)resolve_real_sym("hlthunk_device_memory_map");

    if (hint_addr == 0) {
        uint64_t size = kDramVaAlign;
        pthread_mutex_lock(&g_dram_alloc_mu);
        for (int i = 0; i < 256; ++i) {
            if (g_dram_handles[i].handle == handle && g_dram_handles[i].size > 0) {
                size = g_dram_handles[i].size;
                break;
            }
        }
        uint64_t aligned_size = (size + kDramVaAlign - 1) & ~(kDramVaAlign - 1);
        hint_addr = kSlotDramVaBase + (uint64_t)g_slot * kSlotDramVaStride + g_dram_va_bump;
        g_dram_va_bump += aligned_size;
        pthread_mutex_unlock(&g_dram_alloc_mu);

        MPS_LOG("hlthunk_device_memory_map(handle=%llu): assigned slot-%d hint_addr=0x%llx (size=%llu MB)",
                (unsigned long long)handle,
                g_slot,
                (unsigned long long)hint_addr,
                (unsigned long long)(size >> 20));
    }

    uint64_t mapped_va = real_fn(fd, handle, hint_addr);
    if (mapped_va == 0 && hint_addr != 0) {
        if (hint_addr >= 0x1001600000000000ULL && hint_addr < 0x1001800000000000ULL) {
            // All slots share one ASID: never unmap a VA another slot may be using.
            if (slot_owns_mapped_va(hint_addr)) {
                using unmap_fn_t = int (*)(int, uint64_t);
                static unmap_fn_t real_unmap = (unmap_fn_t)resolve_real_sym("hlthunk_memory_unmap");
                real_unmap(fd, hint_addr);
                remove_slot_mapped_va(hint_addr);
                mapped_va = real_fn(fd, handle, hint_addr);
            }
        } else {
            mapped_va = real_fn(fd, handle, 0);
        }
    }
    if (mapped_va == 0) {
        MPS_LOG("ERROR: hlthunk_device_memory_map(handle=%llu, hint=0x%llx) failed (errno=%d)",
                (unsigned long long)handle, (unsigned long long)hint_addr, errno);
    } else {
        record_slot_mapped_va(mapped_va);
    }
    return mapped_va;
}

static pthread_mutex_t g_host_huge_mu = PTHREAD_MUTEX_INITIALIZER;
static uint64_t g_host_huge_va_bump = 0;

static uint64_t alloc_slot_host_huge_hint(uint64_t host_size)
{
    pthread_mutex_lock(&g_host_huge_mu);
    uint64_t aligned_size = (host_size + kHostHugeVaAlign - 1) & ~(kHostHugeVaAlign - 1);
    if (aligned_size == 0)
        aligned_size = kHostHugeVaAlign;
    uint64_t hint_addr = kSlotHostHugeVaBase + (uint64_t)g_slot * kSlotHostHugeVaStride + g_host_huge_va_bump;
    g_host_huge_va_bump += aligned_size;
    pthread_mutex_unlock(&g_host_huge_mu);
    return hint_addr;
}

static uint64_t assign_host_huge_hint_if_needed(void *host_virt_addr, uint64_t hint_addr, uint64_t host_size)
{
    (void)host_virt_addr;
    if ((hint_addr == 0 ||
         (hint_addr >= kSlotHostHugeVaBase &&
          hint_addr < kSlotHostHugeVaBase + kSlotHostHugeVaStride)) &&
        host_size >= kHostHugeVaAlign &&
        (host_size & (kHostHugeVaAlign - 1)) == 0) {
        hint_addr = alloc_slot_host_huge_hint(host_size);
        MPS_LOG("hlthunk_host_memory_map(size=%llu MB): assigned slot-%d hint_addr=0x%llx",
                (unsigned long long)(host_size >> 20),
                g_slot,
                (unsigned long long)hint_addr);
    }
    return hint_addr;
}

extern "C" uint64_t hlthunk_host_memory_map(
    int fd, void *host_virt_addr, uint64_t hint_addr, uint64_t host_size)
{
    if (!is_mps_disabled()) {
        acquire_slot_if_needed();
        hint_addr = assign_host_huge_hint_if_needed(host_virt_addr, hint_addr, host_size);
    }
    using fn_t = uint64_t (*)(int, void *, uint64_t, uint64_t);
    static fn_t real_fn = (fn_t)resolve_real_sym("hlthunk_host_memory_map");
    uint64_t va = real_fn(fd, host_virt_addr, hint_addr, host_size);
    if (va == 0 && !is_mps_disabled()) {
        uint64_t retry_hint = alloc_slot_host_huge_hint(host_size);
        MPS_LOG("hlthunk_host_memory_map(size=%llu KB): retrying with slot-%d huge hint 0x%llx",
                (unsigned long long)(host_size >> 10), g_slot, (unsigned long long)retry_hint);
        va = real_fn(fd, host_virt_addr, retry_hint, host_size);
        if (va == 0) {
            va = real_fn(fd, host_virt_addr, 0, host_size);
        }
    }
    if (!is_mps_disabled() && va != 0) {
        record_slot_mapped_va(va);
    }
    return va;
}

extern "C" uint64_t hlthunk_host_memory_map_flags(
    int fd, void *host_virt_addr, uint64_t hint_addr, uint64_t host_size, uint32_t flags)
{
    if (!is_mps_disabled()) {
        acquire_slot_if_needed();
        hint_addr = assign_host_huge_hint_if_needed(host_virt_addr, hint_addr, host_size);
    }
    using fn_t = uint64_t (*)(int, void *, uint64_t, uint64_t, uint32_t);
    static fn_t real_fn = (fn_t)resolve_real_sym("hlthunk_host_memory_map_flags");
    uint64_t va = real_fn(fd, host_virt_addr, hint_addr, host_size, flags);
    if (va == 0 && !is_mps_disabled()) {
        uint64_t retry_hint = alloc_slot_host_huge_hint(host_size);
        MPS_LOG("hlthunk_host_memory_map_flags(size=%llu KB, flags=0x%x): retrying with slot-%d huge hint 0x%llx",
                (unsigned long long)(host_size >> 10), flags, g_slot, (unsigned long long)retry_hint);
        va = real_fn(fd, host_virt_addr, retry_hint, host_size, flags);
        if (va == 0) {
            va = real_fn(fd, host_virt_addr, 0, host_size, 0);
        }
    }
    if (!is_mps_disabled() && va != 0) {
        record_slot_mapped_va(va);
    }
    return va;
}

extern "C" int hlthunk_memory_unmap(int fd, uint64_t device_virt_addr)
{
    if (!is_mps_disabled() && g_slot > 0) {
        if (device_virt_addr >= kHostSharedVaBase &&
            device_virt_addr <  kHostSharedVaBase + 0x10000000ULL) {
            // Ensure any in-flight ARC CCB/CQ access finishes before unmapping Secondary host_shared
            usleep(50000);
        }
        remove_slot_mapped_va(device_virt_addr);
    }
    using fn_t = int (*)(int, uint64_t);
    static fn_t real_fn = (fn_t)resolve_real_sym("hlthunk_memory_unmap");
    return real_fn(fd, device_virt_addr);
}

// ============================================================================
// 2. Per-Slot Completion Queue Hardware Isolation
// ============================================================================

extern "C" void *_ZSt18_Rb_tree_incrementPSt18_Rb_tree_node_base(void *node);

static bool cg_belongs_to_slot(int slot, uint32_t dcore, const char *key, size_t key_len)
{
    if (key_len == 0)
        return slot == 0;
    char last = key[key_len - 1];
    if (slot == 0) {
        if (dcore != 3)
            return true;
        return last == '0';
    }
    char target = (char)('0' + slot);
    if (dcore != 3)
        return false;
    if (strncmp(key, "network_", 8) == 0)
        return false;
    return last == target;
}

static void write_cq_if_regs(
    volatile uint32_t *cq_if,
    uint32_t cq_idx,
    uint64_t ctr_dev_addr,
    uint32_t isr_idx,
    bool enable_isr)
{
    cq_if[(0x050 + cq_idx * 4) / 4] = (uint32_t)(ctr_dev_addr & 0xffffffffu);
    cq_if[(0x150 + cq_idx * 4) / 4] = (uint32_t)(ctr_dev_addr >> 32);
    cq_if[(0x250 + cq_idx * 4) / 4] = 0x3u;
    if (enable_isr && isr_idx != 0xffffffffu) {
        cq_if[(0x550 + cq_idx * 4) / 4] = 0xffff0000u;
        cq_if[(0x650 + cq_idx * 4) / 4] = 0xfff077ffu;
        cq_if[(0x750 + cq_idx * 4) / 4] = isr_idx;
    }
    cq_if[(0x850 + cq_idx * 4) / 4] = 0x0u;
    cq_if[(0x850 + cq_idx * 4) / 4] = 0x1u;
    __sync_synchronize();
}

// ComputeLaunchSyncShm, kComputeSyncMagic, kComputeSyncFile are in mps_shm.h.
static ComputeLaunchSyncShm *g_compute_sync_shm = nullptr;
static pthread_mutex_t g_compute_sync_init_mu = PTHREAD_MUTEX_INITIALIZER;

static ComputeLaunchSyncShm *get_compute_sync_shm()
{
    ComputeLaunchSyncShm *shm = __atomic_load_n(&g_compute_sync_shm, __ATOMIC_ACQUIRE);
    if (shm != nullptr)
        return shm;

    pthread_mutex_lock(&g_compute_sync_init_mu);
    shm = g_compute_sync_shm;
    if (shm == nullptr) {
        int fd = open_run_file(kComputeSyncFile, O_RDWR | O_CREAT);
        if (fd >= 0) {
            flock(fd, LOCK_EX);
            if (ftruncate(fd, sizeof(ComputeLaunchSyncShm)) == 0) {
                void *map = mmap(nullptr, sizeof(ComputeLaunchSyncShm),
                                 PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
                if (map != MAP_FAILED) {
                    shm = (ComputeLaunchSyncShm *)map;
                    if (__atomic_load_n(&shm->magic, __ATOMIC_ACQUIRE) != kComputeSyncMagic) {
                        memset((void *)shm->cq_page, 0, sizeof(shm->cq_page));
                        pthread_mutexattr_t attr;
                        pthread_mutexattr_init(&attr);
                        pthread_mutexattr_setpshared(&attr, PTHREAD_PROCESS_SHARED);
                        pthread_mutexattr_setrobust(&attr, PTHREAD_MUTEX_ROBUST);
                        pthread_mutex_init(&shm->mu, &attr);
                        pthread_mutex_init(&shm->bk_mu, &attr);
                        pthread_mutexattr_destroy(&attr);
                        __atomic_store_n(&shm->cq_page_dev_va, 0, __ATOMIC_RELEASE);
                        __atomic_store_n(&shm->active_slot, -1, __ATOMIC_RELEASE);
                        __atomic_store_n(&shm->active_slot_pid, 0, __ATOMIC_RELEASE);
                        __atomic_store_n(&shm->active_cq_idx, 0xffffffffu, __ATOMIC_RELEASE);
                        __atomic_store_n(&shm->burst_count, 0u, __ATOMIC_RELEASE);
                        __atomic_store_n(&shm->active_target_val, 0, __ATOMIC_RELEASE);
                        __atomic_store_n(&shm->last_slot_switch_ns, 0, __ATOMIC_RELEASE);
                        __atomic_store_n(&shm->waiting_mask, 0u, __ATOMIC_RELEASE);
                        __atomic_store_n(&shm->cfg_burst_limit, 32u, __ATOMIC_RELEASE);
                        __atomic_store_n(&shm->cfg_max_inflight, 12u, __ATOMIC_RELEASE);
                        __atomic_store_n(&shm->cfg_pdma_overlap, 1u, __ATOMIC_RELEASE);
                        __atomic_store_n(&shm->cfg_step_yield, 0u, __ATOMIC_RELEASE);
                        __atomic_store_n(&shm->slot_switches, 0u, __ATOMIC_RELEASE);
                        __atomic_store_n(&shm->last_enqueue_ns, 0ULL, __ATOMIC_RELEASE);
                        __atomic_store_n(&shm->cfg_idle_steal_ns, 1000000ULL, __ATOMIC_RELEASE);
                        for (int i = 0; i < 8; ++i) {
                            __atomic_store_n(&shm->slot_burst_limit[i], 0u, __ATOMIC_RELEASE);
                        }
                        __atomic_store_n(&shm->active_rx_base, 0ULL, __ATOMIC_RELEASE);
                        __atomic_store_n(&shm->cfg_quantum_ns, 1800000ULL, __ATOMIC_RELEASE);
                        __atomic_store_n(&shm->cfg_min_burst, 6u, __ATOMIC_RELEASE);
                        __atomic_store_n(&shm->coeff_table_state, kCoeffTableUnknown, __ATOMIC_RELEASE);
                        __atomic_store_n(&shm->coeff_table_pid, 0, __ATOMIC_RELEASE);
                        __atomic_store_n(&shm->reserved_cfg, 0u, __ATOMIC_RELEASE);
                        memset((void *)shm->slot_dram_handles, 0, sizeof(shm->slot_dram_handles));
                        memset((void *)shm->slot_mapped_vas, 0, sizeof(shm->slot_mapped_vas));
                        __atomic_store_n(&shm->magic, kComputeSyncMagic, __ATOMIC_RELEASE);
                    }
                    __atomic_store_n(&g_compute_sync_shm, shm, __ATOMIC_RELEASE);
                }
            }
            flock(fd, LOCK_UN);
            close(fd);
        }
    }
    pthread_mutex_unlock(&g_compute_sync_init_mu);
    return shm;
}

static void lock_internal_mutexes_for_fork()
{
    pthread_mutex_lock(&g_sym_resolve_mu);
    pthread_mutex_lock(&g_compute_sync_init_mu);
    pthread_mutex_lock(&g_dram_alloc_mu);
    pthread_mutex_lock(&g_host_huge_mu);
}

static void unlock_internal_mutexes_after_fork(bool reset_child_state)
{
    if (reset_child_state) {
        memset(g_dram_handles, 0, sizeof(g_dram_handles));
        g_dram_va_bump = 0;
        g_host_huge_va_bump = 0;
        g_secondary_attach_ok = false;
    }
    pthread_mutex_unlock(&g_host_huge_mu);
    pthread_mutex_unlock(&g_dram_alloc_mu);
    pthread_mutex_unlock(&g_compute_sync_init_mu);
    pthread_mutex_unlock(&g_sym_resolve_mu);
}

static void set_coeff_table_state(uint32_t state, int32_t pid)
{
    ComputeLaunchSyncShm *shm = get_compute_sync_shm();
    if (!shm)
        return;
    __atomic_store_n(&shm->coeff_table_pid, pid, __ATOMIC_RELEASE);
    __atomic_store_n(&shm->coeff_table_state, state, __ATOMIC_RELEASE);
}

static void lock_robust(pthread_mutex_t *mu)
{
    if (pthread_mutex_lock(mu) == EOWNERDEAD)
        pthread_mutex_consistent(mu);
}

static bool slot_owns_mapped_va(uint64_t va)
{
    if (g_slot <= 0 || g_slot >= 8 || va == 0)
        return false;
    ComputeLaunchSyncShm *shm = get_compute_sync_shm();
    if (!shm)
        return false;
    bool found = false;
    lock_robust(&shm->bk_mu);
    for (int i = 0; i < 256 && !found; ++i)
        found = shm->slot_mapped_vas[g_slot][i] == va;
    pthread_mutex_unlock(&shm->bk_mu);
    return found;
}

static void record_slot_dram_handle(uint64_t handle)
{
    if (g_slot <= 0 || g_slot >= 8 || handle == 0)
        return;
    ComputeLaunchSyncShm *shm = get_compute_sync_shm();
    if (!shm)
        return;
    lock_robust(&shm->bk_mu);
    for (int i = 0; i < 64; ++i) {
        if (shm->slot_dram_handles[g_slot][i] == handle) {
            pthread_mutex_unlock(&shm->bk_mu);
            return;
        }
    }
    for (int i = 0; i < 64; ++i) {
        if (shm->slot_dram_handles[g_slot][i] == 0) {
            shm->slot_dram_handles[g_slot][i] = handle;
            break;
        }
    }
    pthread_mutex_unlock(&shm->bk_mu);
}

static void remove_slot_dram_handle(uint64_t handle)
{
    if (g_slot <= 0 || g_slot >= 8 || handle == 0)
        return;
    ComputeLaunchSyncShm *shm = get_compute_sync_shm();
    if (!shm)
        return;
    lock_robust(&shm->bk_mu);
    for (int i = 0; i < 64; ++i) {
        if (shm->slot_dram_handles[g_slot][i] == handle) {
            shm->slot_dram_handles[g_slot][i] = 0;
        }
    }
    pthread_mutex_unlock(&shm->bk_mu);
}

static void record_slot_mapped_va(uint64_t va)
{
    if (g_slot <= 0 || g_slot >= 8 || va == 0)
        return;
    ComputeLaunchSyncShm *shm = get_compute_sync_shm();
    if (!shm)
        return;
    lock_robust(&shm->bk_mu);
    for (int i = 0; i < 256; ++i) {
        if (shm->slot_mapped_vas[g_slot][i] == va) {
            pthread_mutex_unlock(&shm->bk_mu);
            return;
        }
    }
    for (int i = 0; i < 256; ++i) {
        if (shm->slot_mapped_vas[g_slot][i] == 0) {
            shm->slot_mapped_vas[g_slot][i] = va;
            break;
        }
    }
    pthread_mutex_unlock(&shm->bk_mu);
}

static void remove_slot_mapped_va(uint64_t va)
{
    if (g_slot <= 0 || g_slot >= 8 || va == 0)
        return;
    ComputeLaunchSyncShm *shm = get_compute_sync_shm();
    if (!shm)
        return;
    lock_robust(&shm->bk_mu);
    for (int i = 0; i < 256; ++i) {
        if (shm->slot_mapped_vas[g_slot][i] == va) {
            shm->slot_mapped_vas[g_slot][i] = 0;
        }
    }
    pthread_mutex_unlock(&shm->bk_mu);
}

static void reclaim_slot_resources_on_shared_fd(int shared_fd, int slot)
{
    if (shared_fd < 0 || slot <= 0 || slot >= 8)
        return;
    ComputeLaunchSyncShm *shm = get_compute_sync_shm();
    if (!shm)
        return;

    uint64_t vas_to_unmap[256];
    uint64_t handles_to_free[64];
    int n_vas = 0;
    int n_handles = 0;

    int rc = pthread_mutex_lock(&shm->mu);
    if (rc == EOWNERDEAD)
        pthread_mutex_consistent(&shm->mu);

    uint32_t my_bit = (1u << slot);
    __atomic_fetch_and(&shm->waiting_mask, ~my_bit, __ATOMIC_ACQ_REL);
    if (__atomic_load_n(&shm->active_slot, __ATOMIC_ACQUIRE) == slot) {
        __atomic_store_n(&shm->active_target_val, 0, __ATOMIC_RELEASE);
        __atomic_store_n(&shm->active_slot, -1, __ATOMIC_RELEASE);
        __atomic_store_n(&shm->burst_count, 0u, __ATOMIC_RELEASE);
    }
    pthread_mutex_unlock(&shm->mu);

    lock_robust(&shm->bk_mu);

    for (int i = 0; i < 256; ++i) {
        uint64_t va = shm->slot_mapped_vas[slot][i];
        if (va != 0) {
            vas_to_unmap[n_vas++] = va;
            shm->slot_mapped_vas[slot][i] = 0;
        }
    }
    for (int i = 0; i < 64; ++i) {
        uint64_t h = shm->slot_dram_handles[slot][i];
        if (h != 0) {
            handles_to_free[n_handles++] = h;
            shm->slot_dram_handles[slot][i] = 0;
        }
    }

    pthread_mutex_unlock(&shm->bk_mu);

    if (n_vas == 0 && n_handles == 0)
        return;

    // Wait briefly so any in-flight DMA/ARC commands from a killed process drain before unmapping
    usleep(50000);

    using unmap_fn_t = int (*)(int, uint64_t);
    using free_fn_t  = int (*)(int, uint64_t);
    static unmap_fn_t real_unmap = (unmap_fn_t)resolve_real_sym("hlthunk_memory_unmap");
    static free_fn_t  real_free  = (free_fn_t)resolve_real_sym("hlthunk_device_memory_free");

    int unmapped_ok = 0;
    for (int i = 0; i < n_vas; ++i) {
        if (real_unmap && real_unmap(shared_fd, vas_to_unmap[i]) == 0) {
            unmapped_ok++;
        }
    }
    int freed_ok = 0;
    for (int i = 0; i < n_handles; ++i) {
        if (real_free && real_free(shared_fd, handles_to_free[i]) == 0) {
            freed_ok++;
        }
    }
    MPS_LOG("reclaim_slot_resources_on_shared_fd(slot=%d): unmapped %d/%d VAs, freed %d/%d HBM handles",
            slot, unmapped_ok, n_vas, freed_ok, n_handles);
}

static void discover_slot_compute_cq_idx(void *self)
{
    char *end_node = (char *)self + 0x2b0;
    char *cur_node = *(char **)((char *)self + 0x2c0);
    char target_char = (char)('0' + g_slot);
    while (cur_node != end_node) {
        const char *key_ptr = *(const char **)(cur_node + 0x20);
        size_t key_len = *(size_t *)(cur_node + 0x28);
        char *sm = *(char **)(cur_node + 0x138);
        uint32_t dcore = sm ? *(uint32_t *)(sm + 0x8) : 0;
        uint32_t cq_idx = *(uint32_t *)(cur_node + 0x148);
        uint32_t global_cq = dcore * 64 + cq_idx;
        if (key_ptr != nullptr && key_len == 25 &&
            strncmp(key_ptr, "compute_completion_queue", 24) == 0 &&
            key_ptr[24] == target_char) {
            g_my_compute_cq_idx = global_cq;
            break;
        }
        cur_node = (char *)_ZSt18_Rb_tree_incrementPSt18_Rb_tree_node_base(cur_node);
    }
}

extern "C" int _ZN11Scal_Gaudi212configureCQsEv(void *self)
{
    if (is_mps_disabled()) {
        using fn_t = int (*)(void *);
        static fn_t real_fn = (fn_t)resolve_real_sym("_ZN11Scal_Gaudi212configureCQsEv");
        return real_fn(self);
    }

    acquire_slot_if_needed();

    int cq_lock_fd = lock_run_file("cq.lock");

    ComputeLaunchSyncShm *shm = get_compute_sync_shm();

    if (g_slot == 0) {
        uint64_t *orig_host = *(uint64_t **)((char *)self + 0x5a8);
        uint64_t orig_dev   = *(uint64_t *)((char *)self + 0x5b0);
        uint64_t shm_dev_va = 0;
        int scal_fd = (g_primary_accel_fd >= 0) ? g_primary_accel_fd : *(int *)((char *)self + 0x8);
        if (shm != nullptr && scal_fd >= 0) {
            using map_fn_t = uint64_t (*)(int, void *, uint64_t, uint64_t);
            static map_fn_t real_host_map =
                (map_fn_t)resolve_real_sym("hlthunk_host_memory_map");
            memset((void *)shm->cq_page, 0, sizeof(shm->cq_page));
            memset((void *)shm->slot_dram_handles, 0, sizeof(shm->slot_dram_handles));
            memset((void *)shm->slot_mapped_vas, 0, sizeof(shm->slot_mapped_vas));
            __atomic_store_n(&shm->active_target_val, 0, __ATOMIC_RELEASE);
            __atomic_store_n(&shm->active_slot, -1, __ATOMIC_RELEASE);
            __atomic_store_n(&shm->active_slot_pid, 0, __ATOMIC_RELEASE);
            __atomic_store_n(&shm->active_cq_idx, 0xffffffffu, __ATOMIC_RELEASE);
            shm_dev_va = real_host_map(scal_fd, (void *)shm->cq_page, 0, sizeof(shm->cq_page));
            __atomic_store_n(&shm->cq_page_dev_va, shm_dev_va, __ATOMIC_RELEASE);
            if (shm_dev_va != 0) {
                *(uint64_t **)((char *)self + 0x5a8) = (uint64_t *)shm->cq_page;
                *(uint64_t *)((char *)self + 0x5b0)  = shm_dev_va;
            }
        }
        set_coeff_table_state(kCoeffTablePending, (int32_t)getpid());

        using fn_t = int (*)(void *);
        static fn_t real_fn = (fn_t)resolve_real_sym("_ZN11Scal_Gaudi212configureCQsEv");
        int rc = real_fn(self);

        *(uint64_t **)((char *)self + 0x5a8) = orig_host;
        *(uint64_t *)((char *)self + 0x5b0)  = orig_dev;

        volatile uint64_t *active_cq_host =
            (shm != nullptr && shm_dev_va != 0) ? shm->cq_page : (volatile uint64_t *)orig_host;
        __atomic_store_n(&g_cq_counters_host, active_cq_host, __ATOMIC_RELEASE);
        discover_slot_compute_cq_idx(self);

        unlock_run_file(cq_lock_fd);
        MPS_LOG("configureCQs (Primary): initialized all CQs via QMAN (rc=%d, cq_host=%p, shm_dev_va=0x%llx, compute_cq=%u)",
                rc, (const void *)g_cq_counters_host, (unsigned long long)shm_dev_va, g_my_compute_cq_idx);
        return rc;
    }

    char *end_node = (char *)self + 0x2b0;
    char *cur_node = *(char **)((char *)self + 0x2c0);
    uint64_t *cq_counters_host = *(uint64_t **)((char *)self + 0x5a8);
    uint64_t cq_counters_dev   = *(uint64_t *)((char *)self + 0x5b0);
    bool using_shm_cq_page = false;
    if (shm != nullptr) {
        uint64_t shm_dev_va = __atomic_load_n(&shm->cq_page_dev_va, __ATOMIC_ACQUIRE);
        if (shm_dev_va != 0) {
            cq_counters_host = (uint64_t *)shm->cq_page;
            cq_counters_dev  = shm_dev_va;
            using_shm_cq_page = true;
        }
        if (__atomic_load_n(&shm->active_slot, __ATOMIC_ACQUIRE) == g_slot) {
            __atomic_store_n(&shm->active_target_val, 0, __ATOMIC_RELEASE);
        }
    }
    __atomic_store_n(&g_cq_counters_host, (volatile uint64_t *)cq_counters_host, __ATOMIC_RELEASE);
    discover_slot_compute_cq_idx(self);

    int owned_count = 0;
    int total_count = 0;

    while (cur_node != end_node) {
        const char *key_ptr = *(const char **)(cur_node + 0x20);
        size_t key_len = *(size_t *)(cur_node + 0x28);
        char *sm = *(char **)(cur_node + 0x138);
        uint32_t dcore = sm ? *(uint32_t *)(sm + 0x8) : 0;
        volatile uint32_t *cq_if = sm ? *(volatile uint32_t **)(sm + 0x40) : nullptr;

        uint32_t isr_idx = *(uint32_t *)(cur_node + 0x78);
        uint64_t **p_ctr_host = (uint64_t **)(cur_node + 0x88);
        uint8_t long_sob = *(uint8_t *)(cur_node + 0x90);
        uint32_t tdr_cq_idx = *(uint32_t *)(cur_node + 0xb0);
        uint64_t **p_tdr_ctr_host = (uint64_t **)(cur_node + 0xb8);
        uint32_t cq_idx = *(uint32_t *)(cur_node + 0x148);

        uint32_t global_cq = dcore * 64 + cq_idx;
        if (cq_counters_host) {
            *p_ctr_host = cq_counters_host + global_cq;
            if (long_sob) {
                uint32_t global_tdr_cq = dcore * 64 + tdr_cq_idx;
                *p_tdr_ctr_host = cq_counters_host + global_tdr_cq;
            }
        }

        if (cg_belongs_to_slot(g_slot, dcore, key_ptr, key_len) && cq_if != nullptr) {
            if (!using_shm_cq_page) {
                if (cq_counters_host) {
                    cq_counters_host[global_cq] = 0;
                }
                uint64_t ctr_dev_addr = cq_counters_dev + (uint64_t)global_cq * 8;
                write_cq_if_regs(cq_if, cq_idx, ctr_dev_addr, isr_idx, true);

                if (long_sob) {
                    uint32_t global_tdr_cq = dcore * 64 + tdr_cq_idx;
                    if (cq_counters_host) {
                        cq_counters_host[global_tdr_cq] = 0;
                    }
                    uint64_t tdr_dev_addr = cq_counters_dev + (uint64_t)global_tdr_cq * 8;
                    write_cq_if_regs(cq_if, tdr_cq_idx, tdr_dev_addr, 0xffffffffu, false);
                }
            }
            owned_count++;
        }

        total_count++;
        cur_node = (char *)_ZSt18_Rb_tree_incrementPSt18_Rb_tree_node_base(cur_node);
    }

    unlock_run_file(cq_lock_fd);

    MPS_LOG("configureCQs (Secondary): programmed %d owned CQs via LBW MMIO (isolated %d non-owned CQs, cq_dev=0x%llx, compute_cq=%u)",
            owned_count, total_count - owned_count,
            (unsigned long long)cq_counters_dev, g_my_compute_cq_idx);
    return 0;
}

// ============================================================================
// 3. Hardware Bringup & Teardown Lifecycle Hooks
// ============================================================================

#define DEFINE_SLOT0_ONLY_HOOK(sym_name, label)                                \
    extern "C" int sym_name(void *self)                                        \
    {                                                                          \
        if (!is_mps_disabled()) {                                              \
            acquire_slot_if_needed();                                          \
            if (g_slot > 0) {                                                  \
                return 0;                                                      \
            }                                                                  \
        }                                                                      \
        using fn_t = int (*)(void *);                                          \
        static fn_t real_fn = (fn_t)resolve_real_sym(#sym_name);               \
        return real_fn(self);                                                  \
    }

DEFINE_SLOT0_ONLY_HOOK(_ZN11Scal_Gaudi212warmupHbmTlbEv,      "warmupHbmTlb")
DEFINE_SLOT0_ONLY_HOOK(_ZN11Scal_Gaudi210resetQMANsEv,        "resetQMANs")
DEFINE_SLOT0_ONLY_HOOK(_ZN11Scal_Gaudi211loadFWImageEv,       "loadFWImage")
DEFINE_SLOT0_ONLY_HOOK(_ZN11Scal_Gaudi216loadTPCNopKernelEv,  "loadTPCNopKernel")
DEFINE_SLOT0_ONLY_HOOK(_ZN11Scal_Gaudi211checkCanaryEv,       "checkCanary")
DEFINE_SLOT0_ONLY_HOOK(_ZN11Scal_Gaudi216configSchedulersEv,  "configSchedulers")
DEFINE_SLOT0_ONLY_HOOK(_ZN11Scal_Gaudi213configEnginesEv,     "configEngines")
DEFINE_SLOT0_ONLY_HOOK(_ZN11Scal_Gaudi217configureMonitorsEv, "configureMonitors")
DEFINE_SLOT0_ONLY_HOOK(_ZN11Scal_Gaudi216configureStreamsEv,  "configureStreams")

extern "C" int _ZN11Scal_Gaudi215activateEnginesEv(void *self)
{
    if (!is_mps_disabled()) {
        acquire_slot_if_needed();
        if (g_slot > 0) {
            return 0;
        }
    }
    using fn_t = int (*)(void *);
    static fn_t real_fn = (fn_t)resolve_real_sym("_ZN11Scal_Gaudi215activateEnginesEv");
    int rc = real_fn(self);
    if (!is_mps_disabled() && rc == 0) {
        char *pool_shared = find_scal_pool(self, "hbm_shared");
        if (pool_shared) {
            *(uint64_t *)(pool_shared + g_pool_offs.off_size) = kHbmSharedSliceSize;
        }
        char ready_path[sizeof(g_run_dir) + 32];
        run_path(ready_path, sizeof(ready_path), kReadyFile);
        if (g_secondary_attach_ok) {
            write_pid_file(ready_path, (int)getpid());
            MPS_LOG("Primary hardware bringup complete; ready for Secondary Attach");
        }
    }
    return rc;
}

extern "C" int _ZN11Scal_Gaudi29haltCoresEv(void *self)
{
    if (!is_mps_disabled()) {
        acquire_slot_if_needed();
        if (g_slot > 0) {
            release_slot_hbm_state();
            MPS_LOG("haltCores: skipped in Secondary Attach mode");
            return 0;
        }

        unlink_run_file(kReadyFile);
        stop_fd_server();
        if (any_secondary_slot_alive()) {
            MPS_LOG("haltCores: waiting for active Secondary slots to finish before Primary teardown...");
            while (any_secondary_slot_alive()) {
                usleep(50000);
            }
            MPS_LOG("haltCores: all Secondary slots finished; proceeding with Primary teardown");
        }

        set_coeff_table_state(kCoeffTableUnknown, 0);
        release_slot_hbm_state();
    }
    using fn_t = int (*)(void *);
    static fn_t real_fn = (fn_t)resolve_real_sym("_ZN11Scal_Gaudi29haltCoresEv");
    return real_fn(self);
}

extern "C" int _ZN6common14CoeffTableConf29submitCoeffTableConfigurationEv(void *self)
{
    using fn_t = int (*)(void *);
    if (is_mps_disabled()) {
        static fn_t real_fn = (fn_t)resolve_real_sym(
            "_ZN6common14CoeffTableConf29submitCoeffTableConfigurationEv");
        return real_fn(self);
    }

    acquire_slot_if_needed();
    if (g_slot == 0) {
        static fn_t real_fn = (fn_t)resolve_real_sym(
            "_ZN6common14CoeffTableConf29submitCoeffTableConfigurationEv");
        int rc = real_fn(self);
        set_coeff_table_state(rc == 0 ? kCoeffTableReady : kCoeffTableFailed, (int32_t)getpid());
        return rc;
    }

    // Slot 0 already uploads the immutable TPC special-function LUT into its permanent
    // HBM and programs all 24 TPCs' global LUT base registers. Letting a secondary slot
    // overwrite those global registers with its own temporary HBM causes HMMU page faults
    // across all other slots as soon as that secondary exits and unmaps its HBM pool.
    ComputeLaunchSyncShm *shm = get_compute_sync_shm();
    if (!shm) {
        fprintf(stderr, "[hpu-mps slot=%d pid=%d] ERROR: compute-sync SHM unavailable while waiting for Primary LUT\n",
                g_slot, (int)getpid());
        return 1; // synFail
    }

    int wait_sec = 30;
    const char *env_wait = getenv("HPU_MPS_WAIT_SEC");
    if (env_wait && env_wait[0]) {
        int v = atoi(env_wait);
        if (v > 0)
            wait_sec = v;
    }

    int loops = wait_sec * 100; // 10ms intervals
    for (int i = 0; i < loops; ++i) {
        int slot0_pid = get_ready_slot0_pid();
        if (slot0_pid <= 0) {
            fprintf(stderr, "[hpu-mps slot=%d pid=%d] ERROR: Primary (slot 0) exited before TPC coeff table was ready\n",
                    g_slot, (int)getpid());
            return 1; // synFail
        }
        uint32_t state = __atomic_load_n(&shm->coeff_table_state, __ATOMIC_ACQUIRE);
        int32_t owner_pid = __atomic_load_n(&shm->coeff_table_pid, __ATOMIC_ACQUIRE);
        if (owner_pid == slot0_pid) {
            if (state == kCoeffTableReady) {
                MPS_LOG("CoeffTableConf::submitCoeffTableConfiguration: skipped in Secondary Attach mode (reusing Primary LUT)");
                return 0; // synSuccess
            }
            if (state == kCoeffTableFailed) {
                fprintf(stderr, "[hpu-mps slot=%d pid=%d] ERROR: Primary (slot 0, pid=%d) failed to initialize TPC coeff table\n",
                        g_slot, (int)getpid(), slot0_pid);
                return 1; // synFail
            }
        }
        usleep(10000);
    }

    fprintf(stderr, "[hpu-mps slot=%d pid=%d] ERROR: timed out after %ds waiting for Primary (slot 0) TPC coeff table\n",
            g_slot, (int)getpid(), wait_sec);
    return 1; // synFail
}

// ============================================================================
// 4. Transparent Stream & Completion Group Name Redirection + Pool Isolation
// ============================================================================

static bool is_partitioned_resource(const char *s)
{
    return strncmp(s, "compute", 7) == 0 ||
           strncmp(s, "pdma_rx", 7) == 0 ||
           strncmp(s, "pdma_tx", 7) == 0 ||
           strncmp(s, "pdma_device2device", 18) == 0;
}

static std::string rewrite_resource_name(const char *name)
{
    if (!name)
        return std::string();
    std::string s(name);
    if (g_slot <= 0 || s.empty() || !is_partitioned_resource(s.c_str()))
        return s;
    char last = s.back();
    if (last >= '0' && last <= '2') {
        int idx = last - '0';
        s.back() = (char)('0' + ((idx + g_slot) % kMaxSlots));
    }
    return s;
}

static bool stream_handle_owned_by_slot(void *stream_handle)
{
    if (!stream_handle)
        return true;
    const char *real_name = *(const char **)((char *)stream_handle + 0x8);
    if (!real_name || !real_name[0])
        return true;
    if (!is_partitioned_resource(real_name)) {
        static const bool enable_hcl = env_is_1("HPU_MPS_ENABLE_HCL");
        return enable_hcl && g_slot == 0;
    }
    char last = real_name[strlen(real_name) - 1];
    if (last >= '0' && last <= '2') {
        return last == (char)('0' + g_slot);
    }
    return g_slot == 0;
}

extern "C" int scal_get_stream_handle_by_name(
    void *scal_handle, const char *stream_name, void *stream_handle)
{
    if (is_mps_disabled()) {
        using fn_t = int (*)(void *, const char *, void *);
        static fn_t real_fn = (fn_t)resolve_real_sym("scal_get_stream_handle_by_name");
        return real_fn(scal_handle, stream_name, stream_handle);
    }
    acquire_slot_if_needed();
    std::string redirected = rewrite_resource_name(stream_name);
    if (redirected != (stream_name ? stream_name : "")) {
        MPS_LOG("scal_get_stream_handle_by_name: %s -> %s", stream_name, redirected.c_str());
    }
    using fn_t = int (*)(void *, const char *, void *);
    static fn_t real_fn = (fn_t)resolve_real_sym("scal_get_stream_handle_by_name");
    return real_fn(scal_handle, redirected.c_str(), stream_handle);
}

extern "C" int scal_get_streamset_handle_by_name(
    void *scal_handle, const char *streamset_name, void *streamset_handle)
{
    if (is_mps_disabled()) {
        using fn_t = int (*)(void *, const char *, void *);
        static fn_t real_fn = (fn_t)resolve_real_sym("scal_get_streamset_handle_by_name");
        return real_fn(scal_handle, streamset_name, streamset_handle);
    }
    acquire_slot_if_needed();
    std::string redirected = rewrite_resource_name(streamset_name);
    using fn_t = int (*)(void *, const char *, void *);
    static fn_t real_fn = (fn_t)resolve_real_sym("scal_get_streamset_handle_by_name");
    return real_fn(scal_handle, redirected.c_str(), streamset_handle);
}

extern "C" int scal_get_completion_group_handle_by_name(
    void *scal_handle, const char *cg_name, void *cg_handle)
{
    if (is_mps_disabled()) {
        using fn_t = int (*)(void *, const char *, void *);
        static fn_t real_fn = (fn_t)resolve_real_sym("scal_get_completion_group_handle_by_name");
        return real_fn(scal_handle, cg_name, cg_handle);
    }
    acquire_slot_if_needed();
    std::string redirected = rewrite_resource_name(cg_name);
    if (redirected != (cg_name ? cg_name : "")) {
        MPS_LOG("scal_get_completion_group_handle_by_name: %s -> %s", cg_name, redirected.c_str());
    }
    using fn_t = int (*)(void *, const char *, void *);
    static fn_t real_fn = (fn_t)resolve_real_sym("scal_get_completion_group_handle_by_name");
    return real_fn(scal_handle, redirected.c_str(), cg_handle);
}

extern "C" int scal_stream_set_commands_buffer(void *stream_handle, void *buffer_handle)
{
    if (!is_mps_disabled()) {
        acquire_slot_if_needed();
        if (!stream_handle_owned_by_slot(stream_handle)) {
            *(void **)((char *)stream_handle + 0x40) = buffer_handle;
            return 0;
        }
        const char *real_name = stream_handle ? *(const char **)((char *)stream_handle + 0x8) : "?";
        MPS_LOG("scal_stream_set_commands_buffer(%s)", real_name ? real_name : "?");
    }
    using fn_t = int (*)(void *, void *);
    static fn_t real_fn = (fn_t)resolve_real_sym("scal_stream_set_commands_buffer");
    return real_fn(stream_handle, buffer_handle);
}

extern "C" int scal_stream_set_priority(void *stream_handle, unsigned int priority)
{
    if (!is_mps_disabled()) {
        acquire_slot_if_needed();
        if (!stream_handle_owned_by_slot(stream_handle)) {
            *(uint32_t *)((char *)stream_handle + 0x4c) = priority;
            return 0;
        }
    }
    using fn_t = int (*)(void *, unsigned int);
    static fn_t real_fn = (fn_t)resolve_real_sym("scal_stream_set_priority");
    return real_fn(stream_handle, priority);
}

extern "C" int scal_stream_submit(void *stream_handle, unsigned int pi, unsigned int submission_alignment)
{
    if (!is_mps_disabled()) {
        acquire_slot_if_needed();
        if (!stream_handle_owned_by_slot(stream_handle)) {
            return 0;
        }
    }
    using fn_t = int (*)(void *, unsigned int, unsigned int);
    static fn_t real_fn = (fn_t)resolve_real_sym("scal_stream_submit");
    return real_fn(stream_handle, pi, submission_alignment);
}

extern "C" int _ZN23ScalCompletionGroupBase4initEv(void *self)
{
    using fn_t = int (*)(void *);
    static fn_t real_fn = (fn_t)resolve_real_sym("_ZN23ScalCompletionGroupBase4initEv");
    int rc = real_fn(self);
    if (!is_mps_disabled() && rc == 0 && self != nullptr && g_slot > 0) {
        uint64_t cur_val = *(uint64_t *)((char *)self + 0x78);
        if (cur_val != 0) {
            *(uint64_t *)((char *)self + 0xe8) = cur_val;
            *(uint64_t *)((char *)self + 0xf0) = cur_val;
            void *cg_handle = *(void **)((char *)self + 0x30);
            if (cg_handle != nullptr) {
                using set_exp_fn_t = int (*)(void *, uint64_t);
                static set_exp_fn_t real_set_exp =
                    (set_exp_fn_t)resolve_real_sym("scal_completion_group_set_expected_ctr");
                if (real_set_exp)
                    real_set_exp(cg_handle, cur_val);
            }
            MPS_LOG("ScalCompletionGroupBase::init: synced slot-%d CG target counter to %llu",
                    g_slot, (unsigned long long)cur_val);
        }
    }
    return rc;
}

extern "C" void _ZN22StreamCyclicBufferBase4initEP23ScalCompletionGroupBasePhmt(
    void *self, void *cg, unsigned char *host_virt_addr, unsigned long stream_handle, unsigned short cmd_align)
{
    using fn_t = void (*)(void *, void *, unsigned char *, unsigned long, unsigned short);
    static fn_t real_fn =
        (fn_t)resolve_real_sym("_ZN22StreamCyclicBufferBase4initEP23ScalCompletionGroupBasePhmt");
    real_fn(self, cg, host_virt_addr, stream_handle, cmd_align);

    if (!is_mps_disabled() && g_slot > 0 && self != nullptr && stream_handle != 0) {
        void *s = reinterpret_cast<void *>(stream_handle);
        if (stream_handle_owned_by_slot(s)) {
            const char *name = *(const char **)((char *)s + 0x08);
            uint32_t stream_id = *(uint32_t *)((char *)s + 0x10);
            uint32_t pi_gran = 64u;
            volatile uint32_t *pi_addr = *(volatile uint32_t **)((char *)s + 0x30);
            if (pi_addr != nullptr) {
                uint32_t hw_pi = *pi_addr;
                uint32_t hw_ci = pi_addr[64];
                if (hw_pi != 0) {
                    uint64_t resume_pi = (uint64_t)hw_pi * (uint64_t)pi_gran;
                    uint64_t ccb_size = *(uint32_t *)((char *)self + 0x78);
                    if (ccb_size == 0)
                        ccb_size = 0x200000ULL;
                    uint64_t off_in_ccb = resume_pi % ccb_size;
                    *(uint64_t *)((char *)self + 0x30) = off_in_ccb;
                    *(uint64_t *)((char *)self + 0x38) = off_in_ccb;
                    *(uint64_t *)((char *)self + 0x40) = resume_pi;
                    MPS_LOG("StreamCyclicBufferBase::init(%s, id=%u): synced hw_pi=%u (hw_ci=%u, gran=%u) -> m_pi=0x%llx, offset=0x%llx",
                            name ? name : "?", stream_id, hw_pi, hw_ci, pi_gran,
                            (unsigned long long)resume_pi, (unsigned long long)off_in_ccb);
                }
            }
        }
    }
}

// ============================================================================
// 5. Intra-Process Multi-Stream Virtualization (Synapse Stream Affinity Hooks)
// ============================================================================
// Clamp Synapse stream affinity mask to 0x1 so multiple synStreamCreateGeneric
// streams inside the same process safely multiplex onto physical queue offset 0
// (which scal_get_stream_handle_by_name redirected to this slot's owned queue).

extern "C" int synDeviceGetAffinityMaskRange(uint32_t devId, uint64_t *streamAffinityMaskRange)
{
    using fn_t = int (*)(uint32_t, uint64_t *);
    static fn_t real_fn = (fn_t)resolve_real_sym("synDeviceGetAffinityMaskRange");
    int rc = real_fn(devId, streamAffinityMaskRange);
    if (!is_mps_disabled() && rc == 0 && streamAffinityMaskRange != nullptr) {
        *streamAffinityMaskRange = 0x1ULL;
    }
    return rc;
}

extern "C" int synDeviceGetNextStreamAffinity(uint32_t devId, uint64_t *streamAffinityMask)
{
    using fn_t = int (*)(uint32_t, uint64_t *);
    static fn_t real_fn = (fn_t)resolve_real_sym("synDeviceGetNextStreamAffinity");
    int rc = real_fn(devId, streamAffinityMask);
    if (!is_mps_disabled() && rc == 0 && streamAffinityMask != nullptr) {
        *streamAffinityMask = 0x1ULL;
    }
    return rc;
}

extern "C" int synStreamSetAffinity(uint32_t devId, void *streamHandle, uint64_t streamAffinityMask)
{
    if (!is_mps_disabled()) {
        streamAffinityMask = 0x1ULL;
    }
    using fn_t = int (*)(uint32_t, void *, uint64_t);
    static fn_t real_fn = (fn_t)resolve_real_sym("synStreamSetAffinity");
    return real_fn(devId, streamHandle, streamAffinityMask);
}

extern "C" int _ZN16StreamsContainer23setStreamAffinityByMaskEP6Streamm(
    void *self, void *stream, unsigned long mask)
{
    if (!is_mps_disabled()) {
        mask = 0x1UL;
    }
    using fn_t = int (*)(void *, void *, unsigned long);
    static fn_t real_fn = (fn_t)resolve_real_sym("_ZN16StreamsContainer23setStreamAffinityByMaskEP6Streamm");
    return real_fn(self, stream, mask);
}

extern "C" int _ZN16StreamsContainer27getDeviceNextStreamAffinityERm(
    void *self, unsigned long *mask)
{
    using fn_t = int (*)(void *, unsigned long *);
    static fn_t real_fn = (fn_t)resolve_real_sym("_ZN16StreamsContainer27getDeviceNextStreamAffinityERm");
    int rc = real_fn(self, mask);
    if (!is_mps_disabled() && rc == 0 && mask != nullptr) {
        *mask = 0x1UL;
    }
    return rc;
}

// ============================================================================
// 6. Monitor & Host Fence Counter Partitioning
// ============================================================================

struct scal_monitor_pool_info_layout {
    void *scal_handle;       // +0x00
    uint32_t sm_index;       // +0x08
    uint32_t pad0;           // +0x0c
    void *sm_base_addr;      // +0x10
    const char *name;        // +0x18
    uint32_t total_size;     // +0x20 (MonitorsPool + 0x28)
    uint32_t base_idx;       // +0x24 (MonitorsPool + 0x20)
    uint32_t align;          // +0x28 (MonitorsPool + 0x2c)
};

extern "C" int scal_monitor_pool_get_info(void *mon_pool_handle, void *info_ptr)
{
    if (!is_mps_disabled())
        acquire_slot_if_needed();
    using fn_t = int (*)(void *, void *);
    static fn_t real_fn = (fn_t)resolve_real_sym("scal_monitor_pool_get_info");
    int rc = real_fn(mon_pool_handle, info_ptr);
    if (!is_mps_disabled() && rc == 0 && info_ptr != nullptr) {
        auto *info = (scal_monitor_pool_info_layout *)info_ptr;
        if (info->name && strcmp(info->name, "compute_gp_monitors") == 0 && info->total_size >= 24) {
            uint32_t slice = (info->total_size / kMaxSlots) & ~3u;
            uint32_t orig_base = info->base_idx;
            uint32_t orig_size = info->total_size;
            info->base_idx = orig_base + (uint32_t)g_slot * slice;
            info->total_size = slice;
            MPS_LOG("scal_monitor_pool_get_info(%s): partitioned [%u, +%u) -> slot %d [%u, +%u)",
                    info->name,
                    orig_base, orig_size, g_slot, info->base_idx, info->total_size);
        }
    }
    return rc;
}

extern "C" int scal_so_pool_get_info(void *so_pool_handle, void *info_ptr)
{
    if (!is_mps_disabled())
        acquire_slot_if_needed();
    using fn_t = int (*)(void *, void *);
    static fn_t real_fn = (fn_t)resolve_real_sym("scal_so_pool_get_info");
    int rc = real_fn(so_pool_handle, info_ptr);
    if (!is_mps_disabled() && rc == 0 && info_ptr != nullptr) {
        auto *info = (scal_monitor_pool_info_layout *)info_ptr;
        if (info->name) {
            if (strcmp(info->name, "sos_pool_long_monitor_wa") == 0) {
                uint32_t orig_base = info->base_idx;
                info->base_idx = orig_base + (uint32_t)g_slot;
                MPS_LOG("scal_so_pool_get_info(%s): shifted base_idx %u -> %u for slot %d",
                        info->name, orig_base, info->base_idx, g_slot);
            } else if (strcmp(info->name, "compute_gp_sos") == 0 && info->total_size >= (uint32_t)kMaxSlots) {
                uint32_t slice = info->total_size / kMaxSlots;
                uint32_t orig_base = info->base_idx;
                uint32_t orig_size = info->total_size;
                info->base_idx = orig_base + (uint32_t)g_slot * slice;
                info->total_size = slice;
                MPS_LOG("scal_so_pool_get_info(%s): partitioned [%u, +%u) -> slot %d [%u, +%u)",
                        info->name, orig_base, orig_size, g_slot, info->base_idx, info->total_size);
            }
        }
    }
    return rc;
}

extern "C" void _ZN17ScalStreamsFences4initEjj(void *self, unsigned int base, unsigned int size)
{
    if (!is_mps_disabled()) {
        acquire_slot_if_needed();
        // compute_media_scheduler supports 64 hardware fence counters (0..63, 6-bit fence_id).
        // Partition into 3 disjoint slices of 20 fences per slot:
        // Slot 0: [0, 20), Slot 1: [20, 40), Slot 2: [40, 60).
        unsigned int orig_base = base;
        unsigned int orig_size = size;
        base = (unsigned int)g_slot * 20u;
        size = 20u;
        MPS_LOG("ScalStreamsFences::init: partitioned [%u, +%u) -> slot %d [%u, +%u)",
                orig_base, orig_size, g_slot, base, size);
    }
    using fn_t = void (*)(void *, unsigned int, unsigned int);
    static fn_t real_fn = (fn_t)resolve_real_sym("_ZN17ScalStreamsFences4initEjj");
    real_fn(self, base, size);
}

extern "C" int scal_get_host_fence_counter_handle_by_name(
    void *scal_handle, const char *fence_name, void *fence_handle)
{
    if (is_mps_disabled()) {
        using fn_t = int (*)(void *, const char *, void *);
        static fn_t real_fn = (fn_t)resolve_real_sym("scal_get_host_fence_counter_handle_by_name");
        return real_fn(scal_handle, fence_name, fence_handle);
    }
    acquire_slot_if_needed();
    std::string redirected(fence_name ? fence_name : "");
    if (g_slot > 0 && redirected.rfind("host_fence_counters_", 0) == 0 && redirected.size() >= 22) {
        char c = redirected[20];
        if (c >= '0' && c <= '2') {
            int idx = c - '0';
            redirected[20] = (char)('0' + ((idx + g_slot) % kMaxSlots));
        }
    }
    using fn_t = int (*)(void *, const char *, void *);
    static fn_t real_fn = (fn_t)resolve_real_sym("scal_get_host_fence_counter_handle_by_name");
    return real_fn(scal_handle, redirected.c_str(), fence_handle);
}

// ============================================================================
// 7. Cross-Process Compute Recipe Serialization (ARC Scheduler Isolation)
// ============================================================================
// On Gaudi2, compute_media_scheduler (ARCFARM_0) serves its DCCM stream queues
// round-robin, one command per queue per pass. Because a single recipe launch on compute{0,1,2} emits a multi-command sequence
// (FENCE_WAIT -> UPDATE_RECIPE_BASE_V2 -> ALLOC_BARRIER_V2 ->
// DISPATCH_COMPUTE_ECB_LIST_V3 -> DISPATCH_BARRIER -> LONG_SO -> FENCE_INC),
// having two compute streams active concurrently causes compute_media_scheduler
// to interleave their UPDATE_RECIPE_BASE_V2 and DISPATCH_COMPUTE_ECB_LIST_V3
// packets into the shared engine ARC queues (TPC/MME/EDMA), overwriting each
// other's recipe base address registers mid-recipe.
//
// Serializing RecipeLauncher::launch + RecipeLauncher::checkCompletionCompute
// across slots via a robust process-shared mutex in the run dir ensures that only
// one compute stream dispatches and executes a recipe on the shared compute
// engines at a time, while PDMA streams continue to run in parallel.


static __thread bool g_timerslack_set = false;

static inline uint64_t mps_monotonic_ns()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static int userspace_wait_cq_counter(
    volatile uint64_t *ctr,
    uint64_t target_value,
    uint64_t timeout_us,
    uint32_t *status,
    uint64_t *timestamp)
{
    if (!status)
        return -EINVAL;

    if (__atomic_load_n(ctr, __ATOMIC_ACQUIRE) >= target_value) {
        *status = 0; // HL_WAIT_CS_STATUS_COMPLETED
        if (timestamp)
            *timestamp = mps_monotonic_ns();
        return 0;
    }

    if (timeout_us == 0) {
        *status = 1; // HL_WAIT_CS_STATUS_BUSY
        return 0;
    }

    // Short initial spin (~25 us) for immediate completion
    for (int i = 0; i < 4000; ++i) {
        if (__atomic_load_n(ctr, __ATOMIC_ACQUIRE) >= target_value) {
            *status = 0; // HL_WAIT_CS_STATUS_COMPLETED
            if (timestamp)
                *timestamp = mps_monotonic_ns();
            return 0;
        }
        __builtin_ia32_pause();
    }

    if (!g_timerslack_set) {
        prctl(PR_SET_TIMERSLACK, 1UL, 0UL, 0UL, 0UL);
        g_timerslack_set = true;
    }

    ComputeLaunchSyncShm *shm = __atomic_load_n(&g_compute_sync_shm, __ATOMIC_RELAXED);
    bool is_compute_cq = (shm != nullptr && ctr >= &shm->cq_page[192] && ctr <= &shm->cq_page[197]);

    uint64_t start_ns = mps_monotonic_ns();
    uint64_t timeout_ns = (timeout_us > (UINT64_MAX / 1000ULL))
                              ? UINT64_MAX
                              : timeout_us * 1000ULL;

    uint32_t iter = 0;
    while (true) {
        if (__atomic_load_n(ctr, __ATOMIC_ACQUIRE) >= target_value) {
            *status = 0; // HL_WAIT_CS_STATUS_COMPLETED
            if (timestamp)
                *timestamp = mps_monotonic_ns();
            return 0;
        }
        if ((iter & 1023u) == 0 && (mps_monotonic_ns() - start_ns) >= timeout_ns) {
            *status = 1; // HL_WAIT_CS_STATUS_BUSY
            return 0;
        }
        ++iter;

        // Spin tightly when waiting on a compute CQ or when our slot is currently
        // active on the HPU and nearing the end of its burst; yield/sleep lightly
        // when another slot owns the HPU so background D2H threads don't steal CPU.
        bool spin_now = is_compute_cq;
        if (!spin_now && shm != nullptr) {
            int32_t act = __atomic_load_n(&shm->active_slot, __ATOMIC_RELAXED);
            if (act < 0 || act == g_slot) {
                spin_now = true;
            }
        }
        if (spin_now && iter < 500000) {
            __builtin_ia32_pause();
        } else if (iter < 2000) {
            sched_yield();
        } else {
            usleep(5);
        }
    }
}

static __thread int g_compute_lock_depth = 0;
static __thread bool g_in_recipe_launch = false;
static __thread bool g_hpu_phase_locked = false;

static void lock_compute_launch()
{
    if (g_compute_lock_depth++ > 0)
        return;
    ComputeLaunchSyncShm *shm = get_compute_sync_shm();
    if (!shm)
        return;
    int rc = pthread_mutex_lock(&shm->mu);
    if (rc == EOWNERDEAD) {
        pthread_mutex_consistent(&shm->mu);
    }
}

static void unlock_compute_launch()
{
    if (--g_compute_lock_depth > 0)
        return;
    ComputeLaunchSyncShm *shm = get_compute_sync_shm();
    if (!shm)
        return;
    pthread_mutex_unlock(&shm->mu);
}

static constexpr uint32_t kMaxSameSlotBurst = 18;
static constexpr uint64_t kMaxSameSlotInFlight = 12;

static inline uint32_t get_slot_burst_limit(ComputeLaunchSyncShm *shm, int32_t slot)
{
    if (shm) {
        if (slot >= 0 && slot < 8) {
            uint32_t sv = __atomic_load_n(&shm->slot_burst_limit[slot], __ATOMIC_RELAXED);
            if (sv > 0)
                return sv;
        }
        uint32_t v = __atomic_load_n(&shm->cfg_burst_limit, __ATOMIC_RELAXED);
        if (v > 0)
            return v;
    }
    return 32u;
}

static inline uint32_t get_burst_limit(ComputeLaunchSyncShm *shm)
{
    return get_slot_burst_limit(shm, g_slot);
}

static inline bool has_slot_quantum_expired(ComputeLaunchSyncShm *shm, int32_t slot, uint32_t burst)
{
    if (shm && slot >= 0 && slot < 8) {
        uint32_t sv = __atomic_load_n(&shm->slot_burst_limit[slot], __ATOMIC_RELAXED);
        if (sv > 0)
            return burst >= sv;
    }
    if (burst >= 32u)
        return true;
    if (!shm)
        return false;
    uint32_t min_burst = __atomic_load_n(&shm->cfg_min_burst, __ATOMIC_RELAXED);
    if (min_burst == 0)
        min_burst = 6u;
    if (burst < min_burst)
        return false;
    uint64_t quantum_ns = __atomic_load_n(&shm->cfg_quantum_ns, __ATOMIC_RELAXED);
    if (quantum_ns > 0) {
        uint64_t sw_ns = __atomic_load_n(&shm->last_slot_switch_ns, __ATOMIC_RELAXED);
        uint64_t now_ns = mps_monotonic_ns();
        if (sw_ns != 0 && now_ns >= sw_ns && (now_ns - sw_ns) >= quantum_ns)
            return true;
    }
    return false;
}

static inline int32_t pick_next_waiting_slot(int32_t act_slot, uint32_t waiting_mask)
{
    if (waiting_mask == 0)
        return -1;
    int32_t start = (act_slot >= 0 && act_slot < 16) ? ((act_slot + 1) & 15) : 0;
    for (int32_t i = 0; i < 16; ++i) {
        int32_t cand = (start + i) & 15;
        if (waiting_mask & (1u << cand))
            return cand;
    }
    return -1;
}

static inline uint64_t get_max_inflight(ComputeLaunchSyncShm *shm)
{
    uint32_t v = shm ? __atomic_load_n(&shm->cfg_max_inflight, __ATOMIC_RELAXED) : 0;
    return (v > 0) ? (uint64_t)v : kMaxSameSlotInFlight;
}

static void wait_for_same_slot_inflight()
{
    ComputeLaunchSyncShm *shm = get_compute_sync_shm();
    if (!shm || g_my_compute_cq_idx >= 512)
        return;
    int32_t act_slot = __atomic_load_n(&shm->active_slot, __ATOMIC_ACQUIRE);
    if (act_slot != g_slot)
        return;
    uint64_t target = __atomic_load_n(&shm->active_target_val, __ATOMIC_ACQUIRE);
    if (target == 0)
        return;
    uint64_t have = __atomic_load_n(&shm->cq_page[g_my_compute_cq_idx], __ATOMIC_ACQUIRE);
    uint64_t cfg_inf = get_max_inflight(shm);
    if (target <= have || (target - have) < cfg_inf)
        return;
    uint32_t my_bit = (g_slot >= 0 && g_slot < 16) ? (1u << g_slot) : 0u;
    uint32_t others = __atomic_load_n(&shm->waiting_mask, __ATOMIC_RELAXED) & ~my_bit;
    uint64_t now_ns = mps_monotonic_ns();
    uint64_t last_switch = __atomic_load_n(&shm->last_slot_switch_ns, __ATOMIC_RELAXED);
    uint64_t max_inflight = (others == 0 && (now_ns - last_switch) > 50000000ULL) ? 32ULL : cfg_inf;
    if ((target - have) < max_inflight)
        return;
    uint64_t wait_target = target - (max_inflight - 1);
    uint32_t status = 0;
    userspace_wait_cq_counter(&shm->cq_page[g_my_compute_cq_idx], wait_target, 5000000ULL, &status, nullptr);
}

static void wait_for_active_compute_recipe()
{
    ComputeLaunchSyncShm *shm = get_compute_sync_shm();
    if (!shm)
        return;
    uint64_t target = __atomic_load_n(&shm->active_target_val, __ATOMIC_ACQUIRE);
    if (target == 0)
        return;
    uint32_t cq_idx = __atomic_load_n(&shm->active_cq_idx, __ATOMIC_ACQUIRE);
    if (cq_idx >= 512)
        return;
    int32_t act_slot = __atomic_load_n(&shm->active_slot, __ATOMIC_ACQUIRE);

    uint64_t wait_target = target;
    if (act_slot == g_slot) {
        // Same-slot pipelining: compute{g_slot} executes in strict FIFO order
        // with hardware barriers between recipes, so we do not need to drain
        // compute{g_slot} between consecutive recipes of the same slot.
        uint64_t have = __atomic_load_n(&shm->cq_page[cq_idx], __ATOMIC_ACQUIRE);
        uint64_t cfg_inf = get_max_inflight(shm);
        if (target <= have || (target - have) < cfg_inf) {
            return;
        }
        uint32_t my_bit = (g_slot >= 0 && g_slot < 16) ? (1u << g_slot) : 0u;
        uint32_t others = __atomic_load_n(&shm->waiting_mask, __ATOMIC_RELAXED) & ~my_bit;
        uint64_t now_ns = mps_monotonic_ns();
        uint64_t last_switch = __atomic_load_n(&shm->last_slot_switch_ns, __ATOMIC_RELAXED);
        uint64_t max_inflight = (others == 0 && (now_ns - last_switch) > 50000000ULL) ? 32ULL : cfg_inf;
        if ((target - have) < max_inflight) {
            return;
        }
        wait_target = target - (max_inflight - 1);
    }

    int32_t pid = __atomic_load_n(&shm->active_slot_pid, __ATOMIC_ACQUIRE);
    if (pid > 0 && kill((pid_t)pid, 0) != 0 && errno == ESRCH) {
        __atomic_store_n(&shm->active_target_val, 0, __ATOMIC_RELEASE);
        return;
    }
    uint32_t status = 0;
    userspace_wait_cq_counter(&shm->cq_page[cq_idx], wait_target, 5000000ULL, &status, nullptr);
    if (status != 0) {
        MPS_LOG("WARNING: wait_for_active_compute_recipe timed out (cq=%u, have=%llu, target=%llu)",
                cq_idx,
                (unsigned long long)__atomic_load_n(&shm->cq_page[cq_idx], __ATOMIC_ACQUIRE),
                (unsigned long long)wait_target);
        __atomic_store_n(&shm->active_target_val, 0, __ATOMIC_RELEASE);
    }
}

static void record_active_compute_recipe(uint64_t target_val)
{
    if (target_val == 0 || g_my_compute_cq_idx >= 512)
        return;
    ComputeLaunchSyncShm *shm = get_compute_sync_shm();
    if (!shm)
        return;
    uint64_t now_ns = mps_monotonic_ns();
    int32_t prev_slot = __atomic_load_n(&shm->active_slot, __ATOMIC_RELAXED);
    if (prev_slot == g_slot) {
        uint32_t b = __atomic_load_n(&shm->burst_count, __ATOMIC_RELAXED);
        __atomic_store_n(&shm->burst_count, b + 1, __ATOMIC_RELAXED);
    } else {
        __atomic_store_n(&shm->last_slot_switch_ns, now_ns, __ATOMIC_RELAXED);
        __atomic_fetch_add(&shm->slot_switches, 1u, __ATOMIC_RELAXED);
        __atomic_store_n(&shm->burst_count, 1u, __ATOMIC_RELAXED);
        __atomic_store_n(&shm->active_slot, g_slot, __ATOMIC_RELAXED);
    }
    uint32_t rx_cq = 198u + 2u * (uint32_t)g_slot;
    if (rx_cq < 512) {
        uint64_t rx_now = __atomic_load_n(&shm->cq_page[rx_cq], __ATOMIC_RELAXED);
        uint64_t tagged_rx = (((uint64_t)(uint32_t)g_slot & 0xffULL) << 56) | (rx_now & 0x00ffffffffffffffULL);
        __atomic_store_n(&shm->active_rx_base, tagged_rx, __ATOMIC_RELAXED);
    }
    __atomic_store_n(&shm->last_enqueue_ns, now_ns, __ATOMIC_RELAXED);
    __atomic_store_n(&shm->active_slot_pid, (int32_t)getpid(), __ATOMIC_RELAXED);
    __atomic_store_n(&shm->active_cq_idx, g_my_compute_cq_idx, __ATOMIC_RELAXED);
    __atomic_store_n(&shm->active_target_val, target_val, __ATOMIC_RELEASE);
}

static bool is_active_slot_still_in_burst(ComputeLaunchSyncShm *shm, int32_t act_slot, uint32_t burst, uint64_t target, uint32_t cq_idx)
{
    if (act_slot < 0 || act_slot == g_slot)
        return false;
    if (has_slot_quantum_expired(shm, act_slot, burst))
        return false;
    if (target == 0 || cq_idx >= 512)
        return false;
    if (__atomic_load_n(&shm->cq_page[cq_idx], __ATOMIC_ACQUIRE) < target)
        return true;
    // All enqueued compute recipes of act_slot have completed on the HPU.
    // Check if act_slot's end-of-step pdma_rx D2H token copy has also completed!
    uint32_t rx_cq = 198u + 2u * (uint32_t)act_slot;
    if (rx_cq < 512) {
        uint64_t tagged_rx = __atomic_load_n(&shm->active_rx_base, __ATOMIC_ACQUIRE);
        if (tagged_rx != 0 && (int32_t)(tagged_rx >> 56) == act_slot) {
            uint64_t rx_base = tagged_rx & 0x00ffffffffffffffULL;
            if (__atomic_load_n(&shm->cq_page[rx_cq], __ATOMIC_ACQUIRE) > rx_base)
                return false;
        }
    }
    uint64_t idle_steal_ns = __atomic_load_n(&shm->cfg_idle_steal_ns, __ATOMIC_RELAXED);
    if (idle_steal_ns > 0) {
        uint64_t last_enq = __atomic_load_n(&shm->last_enqueue_ns, __ATOMIC_ACQUIRE);
        uint64_t now_ns = mps_monotonic_ns();
        if (last_enq != 0 && now_ns >= last_enq && (now_ns - last_enq) < idle_steal_ns)
            return true;
    }
    return false;
}

static void enter_compute_hpu_phase()
{
    if (!g_in_recipe_launch || g_hpu_phase_locked)
        return;

    ComputeLaunchSyncShm *shm = get_compute_sync_shm();
    uint32_t my_bit = (g_slot >= 0 && g_slot < 16) ? (1u << g_slot) : 0u;

    // Fast path when our slot already owns the HPU and no other slot is waiting
    if (shm != nullptr &&
        __atomic_load_n(&shm->active_slot, __ATOMIC_ACQUIRE) == g_slot &&
        (__atomic_load_n(&shm->waiting_mask, __ATOMIC_ACQUIRE) & ~my_bit) == 0) {
        lock_compute_launch();
        if (__atomic_load_n(&shm->active_slot, __ATOMIC_ACQUIRE) == g_slot &&
            (__atomic_load_n(&shm->waiting_mask, __ATOMIC_ACQUIRE) & ~my_bit) == 0) {
            g_hpu_phase_locked = true;
            wait_for_active_compute_recipe();
            return;
        }
        unlock_compute_launch();
    }

    if (shm != nullptr && my_bit != 0) {
        __atomic_fetch_or(&shm->waiting_mask, my_bit, __ATOMIC_ACQ_REL);
    }

    while (true) {
        if (shm != nullptr) {
            uint32_t self_yield_spins = 0;
            uint32_t rr_yield_spins = 0;
            while (true) {
                int32_t act_slot = __atomic_load_n(&shm->active_slot, __ATOMIC_ACQUIRE);
                uint32_t burst = __atomic_load_n(&shm->burst_count, __ATOMIC_ACQUIRE);
                if (act_slot < 0)
                    break;
                if (act_slot == g_slot) {
                    uint32_t others = __atomic_load_n(&shm->waiting_mask, __ATOMIC_ACQUIRE) & ~my_bit;
                    if (others != 0 && has_slot_quantum_expired(shm, g_slot, burst) && self_yield_spins < 4000) {
                        self_yield_spins++;
                        __builtin_ia32_pause();
                        continue;
                    }
                    break;
                }
                uint64_t target = __atomic_load_n(&shm->active_target_val, __ATOMIC_ACQUIRE);
                uint32_t cq_idx = __atomic_load_n(&shm->active_cq_idx, __ATOMIC_ACQUIRE);
                if (is_active_slot_still_in_burst(shm, act_slot, burst, target, cq_idx)) {
                    __builtin_ia32_pause();
                    continue;
                }
                uint32_t wmask = __atomic_load_n(&shm->waiting_mask, __ATOMIC_ACQUIRE);
                int32_t pref = pick_next_waiting_slot(act_slot, wmask);
                if (pref >= 0 && pref != g_slot && rr_yield_spins < 2000) {
                    rr_yield_spins++;
                    __builtin_ia32_pause();
                    continue;
                }
                break;
            }
        }

        lock_compute_launch();

        if (shm != nullptr) {
            int32_t act_slot = __atomic_load_n(&shm->active_slot, __ATOMIC_ACQUIRE);
            uint32_t burst = __atomic_load_n(&shm->burst_count, __ATOMIC_ACQUIRE);
            uint64_t target = __atomic_load_n(&shm->active_target_val, __ATOMIC_ACQUIRE);
            uint32_t cq_idx = __atomic_load_n(&shm->active_cq_idx, __ATOMIC_ACQUIRE);
            if (is_active_slot_still_in_burst(shm, act_slot, burst, target, cq_idx)) {
                unlock_compute_launch();
                __builtin_ia32_pause();
                continue;
            }
        }
        break;
    }

    if (shm != nullptr && my_bit != 0) {
        __atomic_fetch_and(&shm->waiting_mask, ~my_bit, __ATOMIC_ACQ_REL);
    }

    g_hpu_phase_locked = true;
    wait_for_active_compute_recipe();
}

static void leave_compute_hpu_phase()
{
    if (!g_hpu_phase_locked)
        return;
    g_hpu_phase_locked = false;
    unlock_compute_launch();
}

extern "C" int _ZN14RecipeLauncher12pdmaDownloadERK18MemorySectionsScal(
    void *self, const void *sections)
{
    using fn_t = int (*)(void *, const void *);
    static fn_t real_fn =
        (fn_t)resolve_real_sym("_ZN14RecipeLauncher12pdmaDownloadERK18MemorySectionsScal");
    if (!is_mps_disabled()) {
        ComputeLaunchSyncShm *shm = get_compute_sync_shm();
        uint32_t overlap = shm ? __atomic_load_n(&shm->cfg_pdma_overlap, __ATOMIC_RELAXED) : 1u;
        if (overlap) {
            wait_for_same_slot_inflight();
        } else {
            enter_compute_hpu_phase();
        }
    }
    return real_fn(self, sections);
}

extern "C" int _ZN14RecipeLauncher11scalEnqueueEm(
    void *self, unsigned long arg)
{
    using fn_t = int (*)(void *, unsigned long);
    static fn_t real_fn =
        (fn_t)resolve_real_sym("_ZN14RecipeLauncher11scalEnqueueEm");
    if (is_mps_disabled()) {
        return real_fn(self, arg);
    }

    enter_compute_hpu_phase();
    int rc = real_fn(self, arg);
    if (rc == 0 && self != nullptr) {
        uint64_t target_val = *(uint64_t *)((char *)self + 0x130);
        record_active_compute_recipe(target_val);
    }
    return rc;
}

extern "C" int _ZN14RecipeLauncher6launchERK10LaunchInfo(
    void *self, const void *launch_info)
{
    using launch_fn_t = int (*)(void *, const void *);
    static launch_fn_t real_launch =
        (launch_fn_t)resolve_real_sym("_ZN14RecipeLauncher6launchERK10LaunchInfo");
    if (is_mps_disabled()) {
        return real_launch(self, launch_info);
    }

    bool prev_in_launch = g_in_recipe_launch;
    g_in_recipe_launch = true;
    int rc = real_launch(self, launch_info);
    leave_compute_hpu_phase();
    g_in_recipe_launch = prev_in_launch;
    return rc;
}

// ============================================================================
// 8. Userspace CQ Completion Polling (Eliminates drm_file_update_pid RCU Stalls)
// ============================================================================

extern "C" int _ZN4Scal19completionGroupWaitEPKNS_24CompletionGroupInterfaceEmmb(
    void *self, const void *cg, unsigned long target, unsigned long timeout_us, bool always_interrupt)
{
    if (!is_mps_disabled() && self != nullptr) {
        if (__atomic_load_n(&g_cq_counters_host, __ATOMIC_ACQUIRE) == nullptr) {
            volatile uint64_t *cq_host = *(volatile uint64_t **)((char *)self + 0x5a8);
            if (cq_host != nullptr) {
                __atomic_store_n(&g_cq_counters_host, cq_host, __ATOMIC_RELEASE);
            }
        }
        if (!g_in_recipe_launch) {
            ComputeLaunchSyncShm *shm = get_compute_sync_shm();
            if (shm != nullptr &&
                __atomic_load_n(&shm->cfg_step_yield, __ATOMIC_RELAXED) != 0 &&
                __atomic_load_n(&shm->active_slot, __ATOMIC_RELAXED) == g_slot) {
                __atomic_store_n(&shm->burst_count, get_burst_limit(shm), __ATOMIC_RELEASE);
            }
        }
    }
    using fn_t = int (*)(void *, const void *, unsigned long, unsigned long, bool);
    static fn_t real_fn =
        (fn_t)resolve_real_sym("_ZN4Scal19completionGroupWaitEPKNS_24CompletionGroupInterfaceEmmb");
    return real_fn(self, cg, target, timeout_us, always_interrupt);
}

extern "C" int hlthunk_wait_for_interrupt_by_handle_with_timestamp(
    int fd,
    uint64_t cq_counters_handle,
    uint64_t cq_counters_offset,
    uint64_t target_value,
    uint32_t interrupt_id,
    uint64_t timeout_us,
    uint32_t *status,
    uint64_t *timestamp)
{
    if (!is_mps_disabled()) {
        volatile uint64_t *cq_base = __atomic_load_n(&g_cq_counters_host, __ATOMIC_ACQUIRE);
        if (cq_base != nullptr && cq_counters_offset < 512) {
            return userspace_wait_cq_counter(
                &cq_base[cq_counters_offset], target_value, timeout_us, status, timestamp);
        }
    }
    using fn_t = int (*)(int, uint64_t, uint64_t, uint64_t, uint32_t, uint64_t, uint32_t *, uint64_t *);
    static fn_t real_fn =
        (fn_t)resolve_real_sym("hlthunk_wait_for_interrupt_by_handle_with_timestamp");
    return real_fn(fd, cq_counters_handle, cq_counters_offset, target_value,
                   interrupt_id, timeout_us, status, timestamp);
}

extern "C" int hlthunk_wait_for_interrupt_by_handle(
    int fd,
    uint64_t cq_counters_handle,
    uint64_t cq_counters_offset,
    uint64_t target_value,
    uint32_t interrupt_id,
    uint64_t timeout_us,
    uint32_t *status)
{
    return hlthunk_wait_for_interrupt_by_handle_with_timestamp(
        fd, cq_counters_handle, cq_counters_offset, target_value,
        interrupt_id, timeout_us, status, nullptr);
}

extern "C" int hlthunk_wait_for_interrupt_with_timestamp(
    int fd,
    void *addr,
    uint64_t target_value,
    uint32_t interrupt_id,
    uint64_t timeout_us,
    uint32_t *status,
    uint64_t *timestamp)
{
    if (!is_mps_disabled() && addr != nullptr) {
        return userspace_wait_cq_counter(
            (volatile uint64_t *)addr, target_value, timeout_us, status, timestamp);
    }
    using fn_t = int (*)(int, void *, uint64_t, uint32_t, uint64_t, uint32_t *, uint64_t *);
    static fn_t real_fn =
        (fn_t)resolve_real_sym("hlthunk_wait_for_interrupt_with_timestamp");
    return real_fn(fd, addr, target_value, interrupt_id, timeout_us, status, timestamp);
}

extern "C" int hlthunk_wait_for_interrupt(
    int fd,
    void *addr,
    uint64_t target_value,
    uint32_t interrupt_id,
    uint64_t timeout_us,
    uint32_t *status)
{
    return hlthunk_wait_for_interrupt_with_timestamp(
        fd, addr, target_value, interrupt_id, timeout_us, status, nullptr);
}


