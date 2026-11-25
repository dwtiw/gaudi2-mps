// SPDX-License-Identifier: Apache-2.0
/*
 * hpu_mps_top.cpp -- Real-time TUI / CLI monitor for Intel Gaudi2 + hpu-mps.
 *
 * Combines live hardware telemetry from libhlml.so (via dlopen, no SDK headers
 * required at build time) with per-slot hpu-mps state from $HPU_MPS_RUN_DIR
 * (slot locks, HBM quotas/physical footprint, and compute-sync scheduler stats).
 */

#include <ctype.h>
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <termios.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <deque>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "mps_shm.h"

// ============================================================================
// Minimal HLML ABI definitions (matches /usr/include/habanalabs/hlml.h)
// ============================================================================

namespace hlml {

constexpr int kPciDomainLen = 9;
constexpr int kPciAddrLen = kPciDomainLen + 10;
constexpr int kPciLinkInfoLen = 10;

constexpr unsigned long long kThrottlePower = 1ULL << 0;
constexpr unsigned long long kThrottleThermal = 1ULL << 1;

enum Return {
    kSuccess = 0,
    kErrorTimeout = 10,
    kErrorAipIsLost = 15,
    kErrorNoData = 21,
};

static std::string read_trimmed_sysfs(const std::string &path)
{
    int fd = -1;
    do {
        fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
    } while (fd < 0 && errno == EINTR);
    if (fd < 0)
        return "";
    char buf[128];
    ssize_t n = -1;
    do {
        n = read(fd, buf, sizeof(buf) - 1);
    } while (n < 0 && errno == EINTR);
    close(fd);
    if (n <= 0)
        return "";
    buf[n] = '\0';
    size_t start = 0;
    while (start < (size_t)n && (buf[start] == ' ' || buf[start] == '\t' || buf[start] == '\r' || buf[start] == '\n'))
        ++start;
    size_t end = (size_t)n;
    while (end > start && (buf[end - 1] == ' ' || buf[end - 1] == '\t' || buf[end - 1] == '\r' || buf[end - 1] == '\n'))
        --end;
    std::string out;
    out.reserve(end - start);
    for (size_t i = start; i < end; ++i) {
        unsigned char c = (unsigned char)buf[i];
        if (c >= 0x20 && c < 0x7f)
            out.push_back((char)c);
    }
    return out;
}

static bool sanitize_pci_bus_id(const std::string &raw, std::string &out)
{
    if (raw.empty() || raw.size() > 31)
        return false;
    out.clear();
    out.reserve(raw.size());
    for (unsigned char c : raw) {
        if ((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || c == ':' || c == '.') {
            out.push_back((char)c);
        } else if (c >= 'A' && c <= 'F') {
            out.push_back((char)(c - 'A' + 'a'));
        } else {
            return false;
        }
    }
    return true;
}

static std::string read_device_status(const std::string &bus_id, unsigned int dev_index)
{
    std::string clean_bus;
    if (sanitize_pci_bus_id(bus_id, clean_bus)) {
        std::string s = read_trimmed_sysfs("/sys/bus/pci/devices/" + clean_bus + "/status");
        if (!s.empty())
            return s;
    }
    char path[128];
    snprintf(path, sizeof(path), "/sys/class/accel/accel%u/device/status", dev_index);
    return read_trimmed_sysfs(path);
}

enum ClockType {
    kClockSoc = 0,
};

enum TempSensor {
    kTempOnAip = 0,
    kTempOnBoard = 1,
};

struct PciCap {
    char link_speed[kPciLinkInfoLen];
    char link_width[kPciLinkInfoLen];
    char link_max_speed[kPciLinkInfoLen];
    char link_max_width[kPciLinkInfoLen];
};

struct PciInfo {
    unsigned int bus;
    char bus_id[kPciAddrLen];
    unsigned int device;
    unsigned int domain;
    unsigned int pci_device_id;
    PciCap caps;
    unsigned int pci_rev;
    unsigned int pci_subsys_id;
};

struct Utilization {
    unsigned int aip;
    unsigned int memory;
};

struct Memory {
    unsigned long long free;
    unsigned long long total;
    unsigned long long used;
};

using Device = void *;

class Client {
public:
    bool init(unsigned int dev_index) {
        dev_index_ = dev_index;
        if (!lib_) {
            // Use RTLD_GLOBAL so libhlml.so's embedded hlthunk_* symbols remain
            // visible to RTLD_NEXT if an older libhpu_mps.so is in /etc/ld.so.preload.
            lib_ = dlopen("/usr/lib/habanalabs/libhlml.so", RTLD_LAZY | RTLD_GLOBAL);
            if (!lib_)
                lib_ = dlopen("libhlml.so", RTLD_LAZY | RTLD_GLOBAL);
            if (!lib_)
                return false;

            load_sym(fn_init_, "hlml_init");
            load_sym(fn_shutdown_, "hlml_shutdown");
            load_sym(fn_get_count_, "hlml_device_get_count");
            load_sym(fn_get_handle_, "hlml_device_get_handle_by_index");
            load_sym(fn_get_name_, "hlml_device_get_name");
            load_sym(fn_get_pci_, "hlml_device_get_pci_info");
            load_sym(fn_get_driver_, "hlml_get_driver_version");
            load_sym(fn_get_util_, "hlml_device_get_utilization_rates");
            load_sym(fn_get_mem_, "hlml_device_get_memory_info");
            load_sym(fn_get_temp_, "hlml_device_get_temperature");
            load_sym(fn_get_power_, "hlml_device_get_power_usage");
            load_sym(fn_get_power_limit_, "hlml_device_get_power_management_limit");
            load_sym(fn_get_clock_, "hlml_device_get_clock_info");
            load_sym(fn_get_max_clock_, "hlml_device_get_max_clock_info");
            load_sym(fn_get_throttle_, "hlml_device_get_current_clocks_throttle_reasons");
            load_sym(fn_get_pcie_gen_, "hlml_device_get_curr_pcie_link_generation");
            load_sym(fn_get_pcie_width_, "hlml_device_get_curr_pcie_link_width");
        }

        if (inited_)
            shutdown_session();

        if (!fn_init_ || !fn_get_count_ || !fn_get_handle_) {
            init_err_ = -1;
            return false;
        }
        int rc = fn_init_();
        if (rc != kSuccess) {
            init_err_ = rc;
            return false;
        }
        inited_ = true;

        unsigned int count = 0;
        rc = fn_get_count_(&count);
        if (rc != kSuccess || count == 0 || dev_index >= count) {
            init_err_ = (rc != kSuccess) ? rc : kErrorNoData;
            dev_count_ = (rc == kSuccess) ? count : 0;
            shutdown_session();
            return false;
        }
        dev_count_ = count;
        rc = fn_get_handle_(dev_index, &dev_);
        if (rc != kSuccess || !dev_) {
            init_err_ = (rc != kSuccess) ? rc : kErrorNoData;
            shutdown_session();
            return false;
        }
        init_err_ = 0;
        return true;
    }

    bool reinit(unsigned int dev_index) {
        shutdown_session();
        return init(dev_index);
    }

    ~Client() {
        shutdown_session();
        if (lib_)
            dlclose(lib_);
    }

    bool ready() const { return inited_ && dev_ != nullptr; }
    bool has_lib() const { return lib_ != nullptr; }
    unsigned int device_count() const { return dev_count_; }

    struct Telemetry {
        bool valid = false;
        bool lib_loaded = false;
        bool operational = false;
        std::string device_status;
        int last_err = 0;
        std::string name = "HLML unavailable";
        std::string bus_id = "-";
        std::string driver = "-";
        bool has_util = false;
        unsigned int aip_util = 0;
        bool has_mem = false;
        uint64_t mem_used = 0;
        uint64_t mem_total = 0;
        bool has_temp_aip = false;
        unsigned int temp_aip = 0;
        unsigned int temp_board = 0;
        bool has_temp_board = false;
        bool has_power = false;
        double power_w = 0.0;
        bool has_power_limit = false;
        double power_limit_w = 0.0;
        bool has_clock = false;
        unsigned int clock_soc = 0;
        bool has_clock_max = false;
        unsigned int clock_soc_max = 0;
        bool has_throttle = false;
        unsigned long long throttle = 0;
        unsigned int pcie_gen = 0;
        unsigned int pcie_width = 0;
    };

    Telemetry sample() {
        Telemetry t;
        t.lib_loaded = (lib_ != nullptr);
        if (!last_name_.empty())
            t.name = last_name_;
        if (!last_bus_id_.empty())
            t.bus_id = last_bus_id_;
        if (!last_driver_.empty())
            t.driver = last_driver_;

        if (!ready()) {
            t.last_err = init_err_;
            t.device_status = read_device_status(t.bus_id, dev_index_);
            return t;
        }
        t.valid = true;
        if (last_name_.empty())
            t.name = "Gaudi2";

        auto record_err = [&](int rc) {
            if (rc != kSuccess && (t.last_err == 0 || rc == kErrorAipIsLost))
                t.last_err = rc;
        };

        // Check sysfs status early (zero-blocking) so a device in reset/malfunction
        // does not stall the TUI across 10+ serial HLML calls.
        t.device_status = read_device_status(t.bus_id, dev_index_);
        bool status_bad = !t.device_status.empty() && t.device_status != "Operational";

        char buf[128] = {};
        if (!status_bad && fn_get_name_ && fn_get_name_(dev_, buf, sizeof(buf)) == kSuccess) {
            buf[sizeof(buf) - 1] = '\0';
            if (buf[0]) {
                t.name = buf;
                last_name_ = buf;
            }
        }
        memset(buf, 0, sizeof(buf));
        if (!status_bad && fn_get_driver_ && fn_get_driver_(buf, sizeof(buf)) == kSuccess) {
            buf[sizeof(buf) - 1] = '\0';
            if (buf[0]) {
                t.driver = buf;
                last_driver_ = buf;
            }
        }
        PciInfo pci = {};
        if (!status_bad && fn_get_pci_ && fn_get_pci_(dev_, &pci) == kSuccess) {
            pci.bus_id[kPciAddrLen - 1] = '\0';
            if (pci.bus_id[0]) {
                t.bus_id = pci.bus_id;
                last_bus_id_ = pci.bus_id;
            }
        }

        // Re-read sysfs status via discovered PCI bus_id if not yet checked with it.
        if (t.device_status.empty() && t.bus_id != "-") {
            t.device_status = read_device_status(t.bus_id, dev_index_);
            status_bad = !t.device_status.empty() && t.device_status != "Operational";
        }

        if (status_bad) {
            t.operational = false;
            return t;
        }

        bool lost = false;
        Utilization util = {};
        if (!lost && fn_get_util_) {
            int rc = fn_get_util_(dev_, &util);
            if (rc == kSuccess) {
                t.aip_util = util.aip;
                t.has_util = true;
            } else {
                record_err(rc);
                if (rc == kErrorAipIsLost)
                    lost = true;
            }
        }
        Memory mem = {};
        if (!lost && fn_get_mem_) {
            int rc = fn_get_mem_(dev_, &mem);
            if (rc == kSuccess) {
                t.mem_used = mem.used;
                t.mem_total = mem.total;
                t.has_mem = true;
            } else {
                record_err(rc);
                if (rc == kErrorAipIsLost)
                    lost = true;
            }
        }
        unsigned int v = 0;
        if (!lost && fn_get_temp_) {
            int rc = fn_get_temp_(dev_, kTempOnAip, &v);
            if (rc == kSuccess) {
                t.temp_aip = v;
                t.has_temp_aip = true;
            } else {
                record_err(rc);
                if (rc == kErrorAipIsLost)
                    lost = true;
            }
            if (!lost && fn_get_temp_(dev_, kTempOnBoard, &v) == kSuccess) {
                t.temp_board = v;
                t.has_temp_board = true;
            }
        }
        if (!lost && fn_get_power_) {
            int rc = fn_get_power_(dev_, &v);
            if (rc == kSuccess) {
                t.power_w = v / 1000.0;
                t.has_power = true;
            } else {
                record_err(rc);
                if (rc == kErrorAipIsLost)
                    lost = true;
            }
        }
        if (!lost && fn_get_power_limit_) {
            int rc = fn_get_power_limit_(dev_, &v);
            if (rc == kSuccess) {
                t.power_limit_w = v / 1000.0;
                t.has_power_limit = true;
            } else {
                record_err(rc);
                if (rc == kErrorAipIsLost)
                    lost = true;
            }
        }
        if (!lost && fn_get_clock_) {
            int rc = fn_get_clock_(dev_, kClockSoc, &v);
            if (rc == kSuccess) {
                t.clock_soc = v;
                t.has_clock = true;
            } else {
                record_err(rc);
                if (rc == kErrorAipIsLost)
                    lost = true;
            }
        }
        if (!lost && fn_get_max_clock_) {
            int rc = fn_get_max_clock_(dev_, kClockSoc, &v);
            if (rc == kSuccess) {
                t.clock_soc_max = v;
                t.has_clock_max = true;
            } else {
                record_err(rc);
                if (rc == kErrorAipIsLost)
                    lost = true;
            }
        }
        unsigned long long thr = 0;
        if (!lost && fn_get_throttle_) {
            int rc = fn_get_throttle_(dev_, &thr);
            if (rc == kSuccess) {
                t.throttle = thr;
                t.has_throttle = true;
            } else {
                record_err(rc);
                if (rc == kErrorAipIsLost)
                    lost = true;
            }
        }
        if (!lost && fn_get_pcie_gen_ && fn_get_pcie_gen_(dev_, &v) == kSuccess)
            t.pcie_gen = v;
        if (!lost && fn_get_pcie_width_ && fn_get_pcie_width_(dev_, &v) == kSuccess)
            t.pcie_width = v;

        if (lost) {
            // Re-read sysfs status in case the card entered reset mid-sample.
            t.device_status = read_device_status(t.bus_id, dev_index_);
        }
        status_bad = !t.device_status.empty() && t.device_status != "Operational";
        bool no_live_metrics = !t.has_mem && !t.has_util && !t.has_power && !t.has_clock;
        bool bad_mem = t.has_mem && t.mem_total == 0;
        t.operational = !status_bad && !lost && !no_live_metrics && !bad_mem;
        return t;
    }

private:
    void shutdown_session() {
        if (inited_ && fn_shutdown_)
            fn_shutdown_();
        inited_ = false;
        dev_ = nullptr;
    }

    template <typename T>
    void load_sym(T &out, const char *name) {
        out = reinterpret_cast<T>(dlsym(lib_, name));
    }

    void *lib_ = nullptr;
    bool inited_ = false;
    int init_err_ = 0;
    unsigned int dev_index_ = 0;
    unsigned int dev_count_ = 0;
    Device dev_ = nullptr;
    std::string last_name_;
    std::string last_bus_id_;
    std::string last_driver_;

    int (*fn_init_)() = nullptr;
    int (*fn_shutdown_)() = nullptr;
    int (*fn_get_count_)(unsigned int *) = nullptr;
    int (*fn_get_handle_)(unsigned int, Device *) = nullptr;
    int (*fn_get_name_)(Device, char *, unsigned int) = nullptr;
    int (*fn_get_pci_)(Device, PciInfo *) = nullptr;
    int (*fn_get_driver_)(char *, unsigned int) = nullptr;
    int (*fn_get_util_)(Device, Utilization *) = nullptr;
    int (*fn_get_mem_)(Device, Memory *) = nullptr;
    int (*fn_get_temp_)(Device, int, unsigned int *) = nullptr;
    int (*fn_get_power_)(Device, unsigned int *) = nullptr;
    int (*fn_get_power_limit_)(Device, unsigned int *) = nullptr;
    int (*fn_get_clock_)(Device, int, unsigned int *) = nullptr;
    int (*fn_get_max_clock_)(Device, int, unsigned int *) = nullptr;
    int (*fn_get_throttle_)(Device, unsigned long long *) = nullptr;
    int (*fn_get_pcie_gen_)(Device, unsigned int *) = nullptr;
    int (*fn_get_pcie_width_)(Device, unsigned int *) = nullptr;
};

} // namespace hlml

// ============================================================================
// MPS State & /proc Process Reader (100% read-only, SIGBUS-safe pread)
// ============================================================================

struct SlotSnapshot {
    int slot = 0;
    bool active = false;
    bool is_daemon = false;
    bool compute_active = false;
    bool compute_waiting = false;
    int pid = 0;
    uint64_t rel_offset = 0;
    uint64_t quota_bytes = 0;
    uint64_t phys_bytes = 0;
    uint64_t cfg_gb = 0;
    double cpu_pct = -1.0;
    uint64_t rss_bytes = 0;
    std::string cmdline;
};

struct MpsSnapshot {
    std::string run_dir;
    bool primary_ready = false;
    bool zero_prefix = false;
    int primary_pid = 0;
    bool has_shm = false;
    int active_compute_slot = -1; // -1 when idle
    int active_compute_pid = 0;
    uint32_t waiting_mask = 0;
    uint32_t slot_switches = 0;
    double switches_per_sec = 0.0;
    uint64_t reserved_phys_bytes = 0;
    SlotSnapshot slots[kMaxSlots];
};

static uint64_t monotonic_ns()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static double monotonic_sec()
{
    return (double)monotonic_ns() * 1e-9;
}

static std::string resolve_run_dir()
{
    const char *env = getenv("HPU_MPS_RUN_DIR");
    if (env && env[0] == '/')
        return env;
    if (geteuid() == 0)
        return "/run/hpu-mps";
    const char *xdg = getenv("XDG_RUNTIME_DIR");
    if (xdg && xdg[0] == '/')
        return std::string(xdg) + "/hpu-mps";
    char buf[64];
    snprintf(buf, sizeof(buf), "/run/user/%u/hpu-mps", (unsigned)geteuid());
    return buf;
}

static bool check_zero_prefix_enabled()
{
    FILE *fp = fopen("/etc/ld.so.preload", "r");
    if (!fp)
        return false;
    char line[512];
    bool found = false;
    while (fgets(line, sizeof(line), fp)) {
        if (strstr(line, "libhpu_mps.so")) {
            found = true;
            break;
        }
    }
    fclose(fp);
    return found;
}

static int read_pid_from_file(const std::string &path)
{
    FILE *fp = fopen(path.c_str(), "r");
    if (!fp)
        return 0;
    int pid = 0;
    if (fscanf(fp, "%d", &pid) != 1)
        pid = 0;
    fclose(fp);
    return pid;
}

static std::unordered_set<std::string> scan_ofd_write_locks()
{
    std::unordered_set<std::string> keys;
    FILE *fp = fopen("/proc/locks", "r");
    if (!fp)
        return keys;
    char line[512];
    while (fgets(line, sizeof(line), fp)) {
        char f1[32], f2[32], f3[32], f4[32], f5[32], f6[64];
        if (sscanf(line, "%31s %31s %31s %31s %31s %63s", f1, f2, f3, f4, f5, f6) == 6) {
            if (strcmp(f2, "OFDLCK") == 0 && strcmp(f4, "WRITE") == 0)
                keys.insert(f6);
        }
    }
    fclose(fp);
    return keys;
}

static bool file_is_ofd_locked(const std::string &path, const std::unordered_set<std::string> &locks)
{
    struct stat st;
    if (stat(path.c_str(), &st) != 0)
        return false;
    char key[64];
    snprintf(key, sizeof(key), "%02x:%02x:%llu",
             major(st.st_dev), minor(st.st_dev), (unsigned long long)st.st_ino);
    return locks.count(key) > 0;
}

static std::string read_proc_cmdline(int pid)
{
    if (pid <= 0)
        return "";
    char path[64];
    snprintf(path, sizeof(path), "/proc/%d/cmdline", pid);
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0)
        return "";
    char buf[1024];
    ssize_t n = read(fd, buf, sizeof(buf) - 1);
    close(fd);
    if (n <= 0)
        return "";
    std::string out;
    out.reserve((size_t)n);
    for (ssize_t i = 0; i < n; ++i) {
        unsigned char c = (unsigned char)buf[i];
        if (c == 0) {
            if (!out.empty() && out.back() != ' ')
                out.push_back(' ');
        } else if (c >= 0x20 && c < 0x7f) {
            out.push_back((char)c);
        }
    }
    while (!out.empty() && out.back() == ' ')
        out.pop_back();
    return out;
}

static bool read_proc_cpu_rss(int pid, uint64_t &cpu_ticks, uint64_t &starttime, uint64_t &rss_bytes)
{
    if (pid <= 0)
        return false;
    char path[64];
    snprintf(path, sizeof(path), "/proc/%d/stat", pid);
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0)
        return false;
    char buf[2048];
    ssize_t n = read(fd, buf, sizeof(buf) - 1);
    close(fd);
    if (n <= 0)
        return false;
    buf[n] = '\0';
    // Skip past "(comm)" safely using the last ')'
    char *rp = strrchr(buf, ')');
    if (!rp || rp[1] == '\0')
        return false;
    // Fields after ')': 3=state, ..., 14=utime, 15=stime, ..., 22=starttime, ..., 24=rss
    const char *p = rp + 2;
    int field = 3;
    uint64_t utime = 0, stime = 0, st_val = 0;
    long rss_pages = 0;
    while (*p && field <= 24) {
        while (*p == ' ')
            ++p;
        if (!*p)
            break;
        char *end = nullptr;
        if (field == 14) {
            utime = strtoull(p, &end, 10);
        } else if (field == 15) {
            stime = strtoull(p, &end, 10);
        } else if (field == 22) {
            st_val = strtoull(p, &end, 10);
        } else if (field == 24) {
            rss_pages = strtol(p, &end, 10);
        }
        while (*p && *p != ' ')
            ++p;
        ++field;
    }
    if (field <= 24)
        return false;
    cpu_ticks = utime + stime;
    starttime = st_val;
    static const long page_sz = sysconf(_SC_PAGESIZE);
    rss_bytes = (rss_pages > 0 && page_sz > 0) ? (uint64_t)rss_pages * (uint64_t)page_sz : 0;
    return true;
}

static bool pread_exact(int fd, void *buf, size_t len, off_t off)
{
    uint8_t *p = (uint8_t *)buf;
    size_t done = 0;
    while (done < len) {
        ssize_t n = pread(fd, p + done, len - done, off + (off_t)done);
        if (n < 0) {
            if (errno == EINTR)
                continue;
            return false;
        }
        if (n == 0)
            return false;
        done += (size_t)n;
    }
    return true;
}

class MpsReader {
public:
    MpsReader() : run_dir_(resolve_run_dir()), clk_tck_(sysconf(_SC_CLK_TCK)) {
        if (clk_tck_ <= 0)
            clk_tck_ = 100;
    }

    MpsSnapshot sample() {
        MpsSnapshot snap;
        snap.run_dir = run_dir_;
        snap.zero_prefix = check_zero_prefix_enabled();
        uint64_t now_ns = monotonic_ns();
        double now = (double)now_ns * 1e-9;

        auto locks = scan_ofd_write_locks();

        // Parse hbm.state
        struct HbmRec {
            int pid = 0;
            uint64_t rel_off = 0;
            uint64_t size_bytes = 0;
            uint64_t cfg_gb = 0;
            int is_daemon = 0;
        } hbm[kMaxSlots];

        std::string hbm_path = run_dir_ + "/" + kHbmStateFile;
        if (FILE *fp = fopen(hbm_path.c_str(), "r")) {
            char line[256];
            while (fgets(line, sizeof(line), fp)) {
                int s = -1, pid = 0, is_d = 0;
                unsigned long long off = 0, sz = 0, cfg = 0;
                if (sscanf(line, "%d %d %llx %llx %llu %d", &s, &pid, &off, &sz, &cfg, &is_d) >= 5) {
                    if (s >= 0 && s < kMaxSlots) {
                        hbm[s].pid = pid;
                        hbm[s].rel_off = off;
                        hbm[s].size_bytes = sz;
                        hbm[s].cfg_gb = cfg;
                        hbm[s].is_daemon = is_d;
                    }
                }
            }
            fclose(fp);
        }

        // Read compute-sync.v4 via open + fstat + pread (immune to SIGBUS on truncation)
        int raw_active_slot = -1;
        int raw_active_pid = 0;
        uint32_t raw_cq_idx = 0xffffffffu;
        uint64_t raw_target_val = 0;
        uint64_t raw_cq_val = 0;
        uint64_t raw_last_enqueue_ns = 0;

        read_compute_shm(snap, raw_active_slot, raw_active_pid, raw_cq_idx,
                         raw_target_val, raw_cq_val, raw_last_enqueue_ns, now);

        bool slot0_locked = file_is_ofd_locked(run_dir_ + "/slot-0.lock", locks);
        int slot0_pid = slot0_locked ? read_pid_from_file(run_dir_ + "/slot-0.lock") : 0;
        int ready_pid = read_pid_from_file(run_dir_ + "/" + kReadyFile);
        if (slot0_locked && slot0_pid > 0 && slot0_pid == ready_pid) {
            snap.primary_ready = true;
            snap.primary_pid = slot0_pid;
        }
        bool slot0_is_daemon = slot0_locked && (hbm[0].pid == slot0_pid) && (hbm[0].is_daemon != 0);

        // Determine whether raw_active_slot is actively running or launched compute recently
        constexpr uint64_t kRecentComputeWindowNs = 500000000ULL; // 500 ms
        bool compute_in_flight = (raw_cq_idx < 512 && raw_target_val > 0 && raw_cq_val < raw_target_val);
        bool compute_recent = (raw_last_enqueue_ns > 0 && now_ns >= raw_last_enqueue_ns &&
                               (now_ns - raw_last_enqueue_ns) < kRecentComputeWindowNs);

        std::unordered_map<int, ProcPrev> next_proc;
        for (int s = 0; s < kMaxSlots; ++s) {
            SlotSnapshot &ss = snap.slots[s];
            ss.slot = s;
            ss.cfg_gb = hbm[s].cfg_gb;

            std::string lock_path = run_dir_ + "/slot-" + std::to_string(s) + ".lock";
            ss.active = file_is_ofd_locked(lock_path, locks);
            if (ss.active) {
                ss.pid = read_pid_from_file(lock_path);
                if (ss.pid > 0 && hbm[s].pid == ss.pid) {
                    ss.is_daemon = (hbm[s].is_daemon != 0);
                    ss.rel_offset = hbm[s].rel_off;
                    ss.quota_bytes = hbm[s].size_bytes;
                    ss.phys_bytes = align_up_dram_page(hbm[s].size_bytes);
                    snap.reserved_phys_bytes += ss.phys_bytes;
                }
                ss.cmdline = read_proc_cmdline(ss.pid);
                ss.compute_active = snap.has_shm &&
                                    raw_active_slot == s &&
                                    raw_active_pid > 0 &&
                                    raw_active_pid == ss.pid &&
                                    (compute_in_flight || compute_recent);
                ss.compute_waiting = snap.has_shm && ((snap.waiting_mask & (1u << s)) != 0);
                if (ss.compute_active) {
                    snap.active_compute_slot = s;
                    snap.active_compute_pid = ss.pid;
                }

                uint64_t ticks = 0, st_val = 0, rss = 0;
                if (read_proc_cpu_rss(ss.pid, ticks, st_val, rss)) {
                    ss.rss_bytes = rss;
                    auto it = prev_proc_.find(ss.pid);
                    if (it != prev_proc_.end() &&
                        it->second.starttime == st_val &&
                        now > it->second.time_sec &&
                        ticks >= it->second.cpu_ticks) {
                        double dt = now - it->second.time_sec;
                        ss.cpu_pct = ((double)(ticks - it->second.cpu_ticks) / (double)clk_tck_) / dt * 100.0;
                    }
                    next_proc[ss.pid] = {ticks, st_val, now};
                }
            } else {
                uint64_t def_gb = (hbm[s].cfg_gb >= kMinClientHbmGb && hbm[s].cfg_gb <= kMaxClientHbmGb)
                                      ? hbm[s].cfg_gb
                                      : ((slot0_is_daemon && s > 0) ? kDefaultDaemonClientGb : kDefaultActiveSlotGb);
                ss.quota_bytes = def_gb << 30;
            }
        }
        prev_proc_ = std::move(next_proc);
        prev_sample_time_ = now;
        return snap;
    }

private:
    struct ProcPrev {
        uint64_t cpu_ticks = 0;
        uint64_t starttime = 0;
        double time_sec = 0.0;
    };

    void read_compute_shm(
        MpsSnapshot &snap,
        int &raw_active_slot,
        int &raw_active_pid,
        uint32_t &raw_cq_idx,
        uint64_t &raw_target_val,
        uint64_t &raw_cq_val,
        uint64_t &raw_last_enqueue_ns,
        double now)
    {
        std::string path = run_dir_ + "/" + kComputeSyncFile;
        int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
        if (fd < 0) {
            has_prev_switches_ = false;
            return;
        }
        struct stat st;
        if (fstat(fd, &st) != 0 || !S_ISREG(st.st_mode) || st.st_size < (off_t)sizeof(ComputeLaunchSyncShm)) {
            close(fd);
            has_prev_switches_ = false;
            return;
        }

        // Read metadata region [magic .. slot_dram_handles) into a local copy
        static constexpr size_t kMetaOff = offsetof(ComputeLaunchSyncShm, magic);
        static constexpr size_t kMetaEnd = offsetof(ComputeLaunchSyncShm, slot_dram_handles);
        uint8_t meta_buf[kMetaEnd - kMetaOff];
        if (!pread_exact(fd, meta_buf, sizeof(meta_buf), (off_t)kMetaOff)) {
            close(fd);
            has_prev_switches_ = false;
            return;
        }

        auto load_field = [&](size_t member_off, auto &out) {
            memcpy(&out, meta_buf + (member_off - kMetaOff), sizeof(out));
        };

        uint64_t magic = 0;
        load_field(offsetof(ComputeLaunchSyncShm, magic), magic);
        if (magic != kComputeSyncMagic) {
            close(fd);
            has_prev_switches_ = false;
            return;
        }

        int32_t aslot = -1, apid = 0;
        uint32_t cq_idx = 0xffffffffu, wmask = 0, sw = 0;
        uint64_t target = 0, last_enq = 0;
        load_field(offsetof(ComputeLaunchSyncShm, active_slot), aslot);
        load_field(offsetof(ComputeLaunchSyncShm, active_slot_pid), apid);
        load_field(offsetof(ComputeLaunchSyncShm, active_cq_idx), cq_idx);
        load_field(offsetof(ComputeLaunchSyncShm, active_target_val), target);
        load_field(offsetof(ComputeLaunchSyncShm, waiting_mask), wmask);
        load_field(offsetof(ComputeLaunchSyncShm, slot_switches), sw);
        load_field(offsetof(ComputeLaunchSyncShm, last_enqueue_ns), last_enq);

        uint64_t cq_val = 0;
        if (cq_idx < 512) {
            off_t cq_off = (off_t)(offsetof(ComputeLaunchSyncShm, cq_page) + cq_idx * sizeof(uint64_t));
            if (!pread_exact(fd, &cq_val, sizeof(cq_val), cq_off))
                cq_val = target;
        }
        close(fd);

        bool same_file = has_prev_switches_ && (st.st_ino == prev_shm_ino_) && (st.st_dev == prev_shm_dev_);
        snap.has_shm = true;
        snap.waiting_mask = wmask;
        snap.slot_switches = sw;
        if (same_file && prev_sample_time_ > 0.0 && now > prev_sample_time_ && sw >= prev_switches_)
            snap.switches_per_sec = (double)(sw - prev_switches_) / (now - prev_sample_time_);

        prev_switches_ = sw;
        prev_shm_ino_ = st.st_ino;
        prev_shm_dev_ = st.st_dev;
        has_prev_switches_ = true;

        raw_active_slot = aslot;
        raw_active_pid = apid;
        raw_cq_idx = cq_idx;
        raw_target_val = target;
        raw_cq_val = cq_val;
        raw_last_enqueue_ns = last_enq;
    }

    std::string run_dir_;
    long clk_tck_ = 100;
    double prev_sample_time_ = 0.0;
    bool has_prev_switches_ = false;
    ino_t prev_shm_ino_ = 0;
    dev_t prev_shm_dev_ = 0;
    uint32_t prev_switches_ = 0;
    std::unordered_map<int, ProcPrev> prev_proc_;
};

// ============================================================================
// Formatting Helpers & Unicode Braille Time-Series Plot
// ============================================================================

static bool write_all(int fd, const char *buf, size_t len)
{
    size_t done = 0;
    while (done < len) {
        ssize_t n = write(fd, buf + done, len - done);
        if (n < 0) {
            if (errno == EINTR)
                continue;
            return false;
        }
        done += (size_t)n;
    }
    return true;
}

static std::string format_bytes_gib(uint64_t bytes, bool zero_as_dash = true)
{
    char buf[32];
    if (bytes == 0)
        return zero_as_dash ? "-" : "0 GiB";
    double gib = (double)bytes / (double)(1ULL << 30);
    if (fabs(gib - round(gib)) < 0.005)
        snprintf(buf, sizeof(buf), "%.0f GiB", gib);
    else if (fabs(gib * 10.0 - round(gib * 10.0)) < 0.05)
        snprintf(buf, sizeof(buf), "%.1f GiB", gib);
    else
        snprintf(buf, sizeof(buf), "%.2f GiB", gib);
    return buf;
}

static std::string format_rss(uint64_t bytes)
{
    char buf[32];
    if (bytes == 0)
        return "-";
    double mib = (double)bytes / (1024.0 * 1024.0);
    if (mib >= 1024.0)
        snprintf(buf, sizeof(buf), "%.1fG", mib / 1024.0);
    else
        snprintf(buf, sizeof(buf), "%.0fM", mib);
    return buf;
}

static std::string json_escape(const std::string &s)
{
    std::string out;
    out.reserve(s.size() + 8);
    for (unsigned char c : s) {
        if (c == '"')
            out += "\\\"";
        else if (c == '\\')
            out += "\\\\";
        else if (c >= 0x20 && c < 0x7f)
            out.push_back((char)c);
    }
    return out;
}

static void append_utf8_braille(std::string &out, uint8_t bits)
{
    uint32_t cp = 0x2800u + bits;
    out.push_back((char)(0xE0 | ((cp >> 12) & 0x0F)));
    out.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back((char)(0x80 | (cp & 0x3F)));
}

static void append_bar(std::string &out, double pct, int width, const char *color, bool use_color)
{
    if (width < 4)
        width = 4;
    pct = std::max(0.0, std::min(100.0, pct));
    int filled = (int)llround(pct * width / 100.0);
    out += use_color ? "\x1b[90m[\x1b[0m" : "[";
    if (use_color)
        out += color;
    for (int i = 0; i < filled; ++i)
        out += "█";
    if (use_color)
        out += "\x1b[90m";
    for (int i = filled; i < width; ++i)
        out += "░";
    out += use_color ? "]\x1b[0m" : "]";
}

struct PlotSample {
    double aip = 0.0;
    double hbm = 0.0;
    double pwr = 0.0;
    bool offline = false;
};

static void render_braille_plot(
    std::string &out,
    const std::deque<PlotSample> &hist,
    int plot_cols,
    int plot_rows,
    bool show_aip,
    bool show_hbm,
    bool show_pwr,
    double interval_sec)
{
    if (plot_cols < 10 || plot_rows < 4)
        return;

    // Braille dot bit positions for (dx in 0..1, dy in 0..3 from top)
    static const uint8_t kDot[2][4] = {
        {0x01, 0x02, 0x04, 0x40},
        {0x08, 0x10, 0x20, 0x80},
    };

    const int dot_w = plot_cols * 2;
    const int dot_h = plot_rows * 4;
    std::vector<uint8_t> bits_aip((size_t)plot_cols * (size_t)plot_rows, 0);
    std::vector<uint8_t> bits_hbm((size_t)plot_cols * (size_t)plot_rows, 0);
    std::vector<uint8_t> bits_pwr((size_t)plot_cols * (size_t)plot_rows, 0);

    auto plot_series = [&](std::vector<uint8_t> &grid, auto getter) {
        if (hist.empty())
            return;
        int n = (int)std::min<size_t>(hist.size(), (size_t)dot_w);
        int start_x = dot_w - n;
        int prev_y = -1;
        for (int i = 0; i < n; ++i) {
            const PlotSample &sample = hist[hist.size() - (size_t)n + (size_t)i];
            if (sample.offline) {
                prev_y = -1;
                continue;
            }
            double val = getter(sample);
            val = std::max(0.0, std::min(100.0, val));
            int y = dot_h - 1 - (int)llround(val * (dot_h - 1) / 100.0);
            int x = start_x + i;
            int y0 = (prev_y < 0) ? y : std::min(prev_y, y);
            int y1 = (prev_y < 0) ? y : std::max(prev_y, y);
            for (int yy = y0; yy <= y1; ++yy) {
                int cx = x / 2;
                int cy = yy / 4;
                grid[(size_t)cy * (size_t)plot_cols + (size_t)cx] |= kDot[x & 1][yy & 3];
            }
            prev_y = y;
        }
    };

    if (show_aip)
        plot_series(bits_aip, [](const PlotSample &s) { return s.aip; });
    if (show_hbm)
        plot_series(bits_hbm, [](const PlotSample &s) { return s.hbm; });
    if (show_pwr)
        plot_series(bits_pwr, [](const PlotSample &s) { return s.pwr; });

    for (int r = 0; r < plot_rows; ++r) {
        int pct_label = (r == 0) ? 100 : (r == plot_rows - 1 ? 0 : (int)llround(100.0 * (plot_rows - 1 - r) / (plot_rows - 1)));
        bool show_lbl = (r == 0 || r == plot_rows - 1 || r == plot_rows / 2);
        if (plot_rows >= 16 && (r == plot_rows / 4 || r == (3 * plot_rows) / 4))
            show_lbl = true;
        char lbl[16];
        if (show_lbl)
            snprintf(lbl, sizeof(lbl), "%3d%% ", pct_label);
        else
            snprintf(lbl, sizeof(lbl), "     ");
        out += "\x1b[90m";
        out += lbl;
        out += "│\x1b[0m";

        int cur_color = 0; // 0=none, 1=aip(cyan), 2=pwr(yellow), 3=hbm(magenta)
        for (int c = 0; c < plot_cols; ++c) {
            size_t idx = (size_t)r * (size_t)plot_cols + (size_t)c;
            uint8_t a = bits_aip[idx], p = bits_pwr[idx], h = bits_hbm[idx];
            uint8_t merged = a | p | h;
            if (merged == 0) {
                if (cur_color != 0) {
                    out += "\x1b[0m";
                    cur_color = 0;
                }
                out.push_back(' ');
            } else {
                int want = a ? 1 : (p ? 2 : 3);
                if (want != cur_color) {
                    out += (want == 1) ? "\x1b[96m" : (want == 2 ? "\x1b[93m" : "\x1b[95m");
                    cur_color = want;
                }
                append_utf8_braille(out, merged);
            }
        }
        if (cur_color != 0)
            out += "\x1b[0m";
        out += "\x1b[K\n";
    }

    char axis[64];
    int span_s = (int)llround(dot_w * interval_sec);
    snprintf(axis, sizeof(axis), " (%ds)", span_s);
    int axis_len = (int)strlen(axis);
    int dash_cnt = plot_cols - axis_len;
    out += "\x1b[90m     └";
    if (dash_cnt >= 0) {
        for (int i = 0; i < dash_cnt; ++i)
            out += "─";
        out += axis;
    } else {
        for (int i = 0; i < plot_cols; ++i)
            out += "─";
    }
    out += "\x1b[0m\x1b[K\n";
}

// ============================================================================
// Terminal Guard & Signal Handling
// ============================================================================

static volatile sig_atomic_t g_stop = 0;
static volatile sig_atomic_t g_resized = 0;
static volatile sig_atomic_t g_tstp = 0;
static volatile sig_atomic_t g_cont = 0;

static void on_stop_signal(int) { g_stop = 1; }
static void on_winch(int) { g_resized = 1; }
static void on_tstp(int) { g_tstp = 1; }
static void on_cont(int) { g_cont = 1; }

class TerminalGuard {
public:
    bool enter() {
        if (active_)
            return true;
        if (!isatty(STDIN_FILENO) || !isatty(STDOUT_FILENO))
            return false;
        if (tcgetattr(STDIN_FILENO, &orig_) != 0)
            return false;
        struct termios raw = orig_;
        raw.c_lflag &= ~(tcflag_t)(ICANON | ECHO);
        raw.c_cc[VMIN] = 0;
        raw.c_cc[VTIME] = 0;
        if (tcsetattr(STDIN_FILENO, TCSANOW, &raw) != 0)
            return false;
        active_ = true;
        const char *seq = "\x1b[?1049h\x1b[?25l\x1b[2J";
        return write_all(STDOUT_FILENO, seq, strlen(seq));
    }

    void leave() {
        if (!active_)
            return;
        active_ = false;
        const char *seq = "\x1b[0m\x1b[?25h\x1b[?1049l";
        write_all(STDOUT_FILENO, seq, strlen(seq));
        tcsetattr(STDIN_FILENO, TCSANOW, &orig_);
    }

    ~TerminalGuard() { leave(); }

private:
    bool active_ = false;
    struct termios orig_ = {};
};

static void get_term_size(int &cols, int &rows)
{
    struct winsize ws = {};
    if ((ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0 ||
         ioctl(STDIN_FILENO, TIOCGWINSZ, &ws) == 0) &&
        ws.ws_col > 0 && ws.ws_row > 0) {
        cols = ws.ws_col;
        rows = ws.ws_row;
    } else {
        cols = 100;
        rows = 28;
    }
}

static void drain_escape_sequence()
{
    char discard[16];
    struct pollfd pfd = {STDIN_FILENO, POLLIN, 0};
    while (poll(&pfd, 1, 15) > 0 && (pfd.revents & POLLIN)) {
        if (read(STDIN_FILENO, discard, sizeof(discard)) <= 0)
            break;
    }
}

// ============================================================================
// Renderers: JSON, Plain (--once), and Interactive TUI
// ============================================================================

static void print_json(const hlml::Client::Telemetry &hw, const MpsSnapshot &mps)
{
    printf("{\n");
    printf("  \"device\": {\n");
    printf("    \"valid\": %s,\n", hw.valid ? "true" : "false");
    printf("    \"operational\": %s,\n", hw.operational ? "true" : "false");
    printf("    \"device_status\": \"%s\",\n", json_escape(hw.device_status).c_str());
    printf("    \"hlml_error\": %d,\n", hw.last_err);
    printf("    \"name\": \"%s\",\n", json_escape(hw.name).c_str());
    printf("    \"bus_id\": \"%s\",\n", json_escape(hw.bus_id).c_str());
    printf("    \"driver\": \"%s\",\n", json_escape(hw.driver).c_str());
    printf("    \"aip_util_pct\": %u,\n", hw.aip_util);
    printf("    \"mem_used_bytes\": %llu,\n", (unsigned long long)hw.mem_used);
    printf("    \"mem_total_bytes\": %llu,\n", (unsigned long long)hw.mem_total);
    printf("    \"temp_aip_c\": %u,\n", hw.temp_aip);
    if (hw.has_temp_board)
        printf("    \"temp_board_c\": %u,\n", hw.temp_board);
    printf("    \"power_w\": %.1f,\n", hw.power_w);
    printf("    \"power_limit_w\": %.1f,\n", hw.power_limit_w);
    printf("    \"clock_soc_mhz\": %u,\n", hw.clock_soc);
    printf("    \"clock_soc_max_mhz\": %u,\n", hw.clock_soc_max);
    printf("    \"throttle_power\": %s,\n", (hw.throttle & hlml::kThrottlePower) ? "true" : "false");
    printf("    \"throttle_thermal\": %s,\n", (hw.throttle & hlml::kThrottleThermal) ? "true" : "false");
    printf("    \"pcie_gen\": %u,\n", hw.pcie_gen);
    printf("    \"pcie_width\": %u\n", hw.pcie_width);
    printf("  },\n");
    printf("  \"mps\": {\n");
    printf("    \"run_dir\": \"%s\",\n", json_escape(mps.run_dir).c_str());
    printf("    \"primary_ready\": %s,\n", mps.primary_ready ? "true" : "false");
    printf("    \"zero_prefix\": %s,\n", mps.zero_prefix ? "true" : "false");
    printf("    \"active_compute_slot\": %d,\n", mps.active_compute_slot);
    printf("    \"slot_switches\": %u,\n", mps.slot_switches);
    printf("    \"reserved_phys_bytes\": %llu,\n", (unsigned long long)mps.reserved_phys_bytes);
    printf("    \"slots\": [\n");
    for (int s = 0; s < kMaxSlots; ++s) {
        const auto &ss = mps.slots[s];
        printf("      {\"slot\": %d, \"active\": %s, \"is_daemon\": %s, \"pid\": %d, "
               "\"quota_bytes\": %llu, \"phys_bytes\": %llu, \"rss_bytes\": %llu, "
               "\"compute_active\": %s, \"compute_waiting\": %s, \"cmdline\": \"%s\"}%s\n",
               ss.slot,
               ss.active ? "true" : "false",
               ss.is_daemon ? "true" : "false",
               ss.pid,
               (unsigned long long)ss.quota_bytes,
               (unsigned long long)ss.phys_bytes,
               (unsigned long long)ss.rss_bytes,
               ss.compute_active ? "true" : "false",
               ss.compute_waiting ? "true" : "false",
               json_escape(ss.cmdline).c_str(),
               (s + 1 < kMaxSlots) ? "," : "");
    }
    printf("    ]\n");
    printf("  }\n");
    printf("}\n");
}

static void append_tui_line(std::string &out, const std::string &src, int cols, bool tui, bool newline = true)
{
    if (!tui) {
        out += src;
        if (newline)
            out.push_back('\n');
        return;
    }
    int limit = newline ? std::max(1, cols) : std::max(0, cols - 1);
    int vis = 0;
    bool truncated = false;
    size_t i = 0, n = src.size();
    while (i < n) {
        unsigned char c = (unsigned char)src[i];
        if (c == 0x1b && i + 1 < n && src[i + 1] == '[') {
            size_t j = i + 2;
            while (j < n && ((unsigned char)src[j] < 0x40 || (unsigned char)src[j] > 0x7e))
                ++j;
            if (j < n)
                ++j;
            out.append(src, i, j - i);
            i = j;
            continue;
        }
        if (c < 0x20 || c == 0x7f) {
            ++i;
            continue;
        }
        size_t len = 1;
        if ((c & 0xE0) == 0xC0)
            len = 2;
        else if ((c & 0xF0) == 0xE0)
            len = 3;
        else if ((c & 0xF8) == 0xF0)
            len = 4;
        if (i + len > n)
            len = n - i;
        if (vis + 1 > limit) {
            truncated = true;
            break;
        }
        out.append(src, i, len);
        ++vis;
        i += len;
    }
    if (truncated)
        out += "\x1b[0m";
    if (vis < cols)
        out += "\x1b[K";
    if (newline)
        out.push_back('\n');
    else
        out += "\x1b[J";
}

static std::string build_frame(
    const hlml::Client::Telemetry &hw,
    const MpsSnapshot &mps,
    const std::deque<PlotSample> &hist,
    int cols,
    int rows,
    bool tui,
    bool use_color,
    bool paused,
    double interval_sec,
    bool show_aip,
    bool show_hbm,
    bool show_pwr,
    bool full_clear = false)
{
    static const char *kRoles[kMaxSlots] = {"PRIMARY", "ATTACH-1", "ATTACH-2"};
    auto C = [&](const char *esc) -> const char * { return use_color ? esc : ""; };

    std::string out;
    out.reserve((size_t)std::max(8192, std::min(cols, 512) * std::min(rows, 200) * 4));
    if (tui)
        out += full_clear ? "\x1b[2J\x1b[H" : "\x1b[H";

    time_t now_t = time(nullptr);
    struct tm lt = {};
    localtime_r(&now_t, &lt);
    char timestr[16];
    strftime(timestr, sizeof(timestr), "%H:%M:%S", &lt);

    char line[512];
    std::string row;
    row.reserve(256);
    int rows_emitted = 0;
    auto emit_row = [&](const std::string &r, bool force_last = false) {
        if (!tui) {
            append_tui_line(out, r, cols, false, !force_last);
            return;
        }
        if (rows_emitted >= rows)
            return;
        bool is_last = force_last || (rows_emitted + 1 >= rows);
        append_tui_line(out, r, cols, true, !is_last);
        ++rows_emitted;
    };

    // Header
    snprintf(line, sizeof(line), " %s%s%s [%s]  Driver %s",
             C("\x1b[1;97m"), hw.name.c_str(), C("\x1b[0m"),
             hw.bus_id.c_str(), hw.driver.c_str());
    row = line;
    if ((hw.valid && !hw.operational) ||
        (!hw.valid && !hw.device_status.empty() && hw.device_status != "Operational")) {
        const char *st_label = hw.device_status.empty() ? "UNUSABLE" : hw.device_status.c_str();
        snprintf(line, sizeof(line), "  %sDEVICE: %s%s", C("\x1b[1;91m"), st_label, C("\x1b[0m"));
        row += line;
    }
    row += "  MPS: ";
    if (mps.primary_ready) {
        row += C("\x1b[92m");
        row += mps.zero_prefix ? "ACTIVE (Zero-Prefix)" : "ACTIVE";
        row += C("\x1b[0m");
    } else {
        row += C("\x1b[93m");
        row += "INACTIVE";
        row += C("\x1b[0m");
    }
    if (paused) {
        row += "  ";
        row += C("\x1b[1;93m");
        row += "[PAUSED]";
        row += C("\x1b[0m");
    }
    snprintf(line, sizeof(line), "  %s%s%s", C("\x1b[90m"), timestr, C("\x1b[0m"));
    row += line;
    emit_row(row);

    // Divider
    row = C("\x1b[90m");
    for (int i = 0; i < cols; ++i)
        row += "─";
    row += C("\x1b[0m");
    emit_row(row);

    double hbm_pct = (hw.operational && hw.has_mem && hw.mem_total > 0)
                         ? (100.0 * (double)hw.mem_used / (double)hw.mem_total)
                         : 0.0;
    double pwr_pct = (hw.operational && hw.has_power && hw.has_power_limit && hw.power_limit_w > 0.0)
                         ? (100.0 * hw.power_w / hw.power_limit_w)
                         : 0.0;
    int bar_w = std::max(8, std::min((cols - 48) / 2, cols - 81));

    if (hw.valid && hw.operational) {
        row = " ";
        row += C("\x1b[1;96m");
        row += "AIP";
        row += C("\x1b[0m");
        row += " ";
        append_bar(row, hw.has_util ? (double)hw.aip_util : 0.0, bar_w, "\x1b[96m", use_color);
        if (hw.has_util)
            snprintf(line, sizeof(line), " %3u%%   %sHBM%s ", hw.aip_util, C("\x1b[1;95m"), C("\x1b[0m"));
        else
            snprintf(line, sizeof(line), "  N/A   %sHBM%s ", C("\x1b[1;95m"), C("\x1b[0m"));
        row += line;
        append_bar(row, hbm_pct, bar_w, "\x1b[95m", use_color);
        if (hw.has_mem && hw.mem_total > 0) {
            snprintf(line, sizeof(line), " %.1f / %.1f GiB (%2.0f%%)",
                     (double)hw.mem_used / (double)(1ULL << 30),
                     (double)hw.mem_total / (double)(1ULL << 30),
                     hbm_pct);
        } else {
            snprintf(line, sizeof(line), " N/A");
        }
        row += line;
        emit_row(row);

        const char *pwr_col = (pwr_pct >= 90.0) ? "\x1b[91m" : "\x1b[93m";
        const char *tmp_col = (hw.temp_aip >= 80 ? "\x1b[91m" : (hw.temp_aip >= 70 ? "\x1b[93m" : "\x1b[92m"));
        row = " ";
        row += C("\x1b[1;93m");
        row += "PWR";
        row += C("\x1b[0m");
        row += " ";
        append_bar(row, pwr_pct, bar_w, pwr_col, use_color);
        if (hw.has_power && hw.has_power_limit && hw.power_limit_w > 0.0)
            snprintf(line, sizeof(line), " %3.0f / %.0f W", hw.power_w, hw.power_limit_w);
        else if (hw.has_power)
            snprintf(line, sizeof(line), " %3.0f W", hw.power_w);
        else
            snprintf(line, sizeof(line), " N/A");
        row += line;

        if (hw.has_temp_aip) {
            snprintf(line, sizeof(line), "   %sTEMP%s %s%uC%s",
                     C("\x1b[1m"), C("\x1b[0m"), C(tmp_col), hw.temp_aip, C("\x1b[0m"));
        } else {
            snprintf(line, sizeof(line), "   %sTEMP%s N/A", C("\x1b[1m"), C("\x1b[0m"));
        }
        row += line;
        if (hw.has_temp_board && hw.temp_board > 0) {
            snprintf(line, sizeof(line), " (Board %uC)", hw.temp_board);
            row += line;
        }
        if (hw.has_clock && hw.has_clock_max) {
            snprintf(line, sizeof(line), "   %sCLK%s %u/%u MHz",
                     C("\x1b[1m"), C("\x1b[0m"), hw.clock_soc, hw.clock_soc_max);
        } else if (hw.has_clock) {
            snprintf(line, sizeof(line), "   %sCLK%s %u MHz",
                     C("\x1b[1m"), C("\x1b[0m"), hw.clock_soc);
        } else {
            snprintf(line, sizeof(line), "   %sCLK%s N/A", C("\x1b[1m"), C("\x1b[0m"));
        }
        row += line;
        if (hw.pcie_gen > 0 && hw.pcie_width > 0) {
            snprintf(line, sizeof(line), "   %sPCIe%s Gen%u x%u",
                     C("\x1b[1m"), C("\x1b[0m"), hw.pcie_gen, hw.pcie_width);
            row += line;
        }
        emit_row(row);
    } else if (hw.valid || (!hw.device_status.empty() && hw.device_status != "Operational")) {
        const char *st_label = hw.device_status.empty() ? "NOT OPERATIONAL" : hw.device_status.c_str();
        if (hw.last_err > 0) {
            snprintf(line, sizeof(line),
                     " %sDevice %s (hlml err %d) — telemetry unavailable (check dmesg / hl-smi)%s",
                     C("\x1b[1;91m"), st_label, hw.last_err, C("\x1b[0m"));
        } else if (!hw.lib_loaded) {
            snprintf(line, sizeof(line),
                     " %sDevice %s — libhlml.so unavailable (check dmesg / hl-smi)%s",
                     C("\x1b[1;91m"), st_label, C("\x1b[0m"));
        } else {
            snprintf(line, sizeof(line),
                     " %sDevice %s — telemetry unavailable (check dmesg / hl-smi)%s",
                     C("\x1b[1;91m"), st_label, C("\x1b[0m"));
        }
        row = line;
        emit_row(row);
        snprintf(line, sizeof(line),
                 " %sAIP%s N/A   %sHBM%s N/A   %sPWR%s N/A   %sTEMP%s N/A   %sCLK%s N/A",
                 C("\x1b[1;96m"), C("\x1b[0m"),
                 C("\x1b[1;95m"), C("\x1b[0m"),
                 C("\x1b[1;93m"), C("\x1b[0m"),
                 C("\x1b[1m"), C("\x1b[0m"),
                 C("\x1b[1m"), C("\x1b[0m"));
        row = line;
        emit_row(row);
    } else {
        snprintf(line, sizeof(line), " %sHardware telemetry unavailable (libhlml.so / device not accessible)%s",
                 C("\x1b[93m"), C("\x1b[0m"));
        row = line;
        emit_row(row);
    }

    // Throttle, Physical HBM pool summary, MPS scheduler
    row = " Throttle: ";
    if (!hw.operational || !hw.has_throttle) {
        row += (hw.valid || !hw.device_status.empty()) ? "N/A" : "-";
    } else if ((hw.throttle & (hlml::kThrottlePower | hlml::kThrottleThermal)) ==
               (hlml::kThrottlePower | hlml::kThrottleThermal)) {
        row += C("\x1b[91m");
        row += "PWR+THERM";
        row += C("\x1b[0m");
    } else if (hw.throttle & hlml::kThrottlePower) {
        row += C("\x1b[93m");
        row += "POWER";
        row += C("\x1b[0m");
    } else if (hw.throttle & hlml::kThrottleThermal) {
        row += C("\x1b[91m");
        row += "THERMAL";
        row += C("\x1b[0m");
    } else {
        row += C("\x1b[92m");
        row += "NONE";
        row += C("\x1b[0m");
    }

    uint64_t usable_phys = (hw.operational && hw.has_mem && hw.mem_total > kDramPageBytes)
                               ? (hw.mem_total - kDramPageBytes)
                               : 0;
    uint64_t free_phys = (usable_phys > mps.reserved_phys_bytes) ? (usable_phys - mps.reserved_phys_bytes) : 0;
    if (usable_phys > 0) {
        snprintf(line, sizeof(line), "   Pool Phys: %s / %s (Free %s)",
                 format_bytes_gib(mps.reserved_phys_bytes, false).c_str(),
                 format_bytes_gib(usable_phys, false).c_str(),
                 format_bytes_gib(free_phys, false).c_str());
        row += line;
    }
    if (mps.has_shm) {
        if (mps.active_compute_slot >= 0 && mps.active_compute_slot < kMaxSlots) {
            snprintf(line, sizeof(line), "   Sched: Slot %d  Sw: %u (+%.0f/s)",
                     mps.active_compute_slot, mps.slot_switches, mps.switches_per_sec);
        } else {
            snprintf(line, sizeof(line), "   Sched: idle  Sw: %u (+%.0f/s)",
                     mps.slot_switches, mps.switches_per_sec);
        }
        row += line;
    }
    emit_row(row);
    emit_row("");

    // Plot area (only in interactive TUI when terminal has room and device is operational or has history)
    if (tui && (hw.operational || !hist.empty())) {
        int plot_rows = rows - rows_emitted - (4 + kMaxSlots);
        int plot_cols = cols - 7;
        if (plot_rows >= 4 && plot_cols >= 10 && cols >= 30) {
            if (hw.operational) {
                char aip_s[16], hbm_s[16], pwr_s[16];
                if (hw.has_util)
                    snprintf(aip_s, sizeof(aip_s), "%u%%", hw.aip_util);
                else
                    snprintf(aip_s, sizeof(aip_s), "N/A");
                if (hw.has_mem && hw.mem_total > 0)
                    snprintf(hbm_s, sizeof(hbm_s), "%.0f%%", hbm_pct);
                else
                    snprintf(hbm_s, sizeof(hbm_s), "N/A");
                if (hw.has_power && hw.has_power_limit && hw.power_limit_w > 0.0)
                    snprintf(pwr_s, sizeof(pwr_s), "%.0f%%", pwr_pct);
                else
                    snprintf(pwr_s, sizeof(pwr_s), "N/A");
                snprintf(line, sizeof(line),
                         " Legend: %s1:AIP %s\x1b[0m  %s2:HBM %s\x1b[0m  %s3:PWR %s\x1b[0m",
                         show_aip ? "\x1b[96m● " : "\x1b[90m○ ", aip_s,
                         show_hbm ? "\x1b[95m● " : "\x1b[90m○ ", hbm_s,
                         show_pwr ? "\x1b[93m● " : "\x1b[90m○ ", pwr_s);
            } else {
                snprintf(line, sizeof(line),
                         " Legend: %s1:AIP N/A\x1b[0m  %s2:HBM N/A\x1b[0m  %s3:PWR N/A\x1b[0m  %s(device offline)%s",
                         show_aip ? "\x1b[96m● " : "\x1b[90m○ ",
                         show_hbm ? "\x1b[95m● " : "\x1b[90m○ ",
                         show_pwr ? "\x1b[93m● " : "\x1b[90m○ ",
                         C("\x1b[91m"), C("\x1b[0m"));
            }
            row = line;
            emit_row(row);
            render_braille_plot(out, hist, plot_cols, plot_rows,
                                show_aip, show_hbm, show_pwr, interval_sec);
            rows_emitted += plot_rows + 1;
        }
    }

    // MPS Slot Process Table
    snprintf(line, sizeof(line),
             "%s %-4s %-8s %-7s %-7s %6s %6s %-10s %-8s %s%s",
             C("\x1b[1m"), "SLOT", "ROLE", "STATE", "PID", "CPU%", "RSS",
             "QUOTA", "PHYS", "COMMAND", C("\x1b[0m"));
    row = line;
    emit_row(row);

    int cmd_w = std::max(12, cols - 65);
    for (int s = 0; s < kMaxSlots; ++s) {
        const auto &ss = mps.slots[s];
        char state_str[16];
        const char *state_col = "\x1b[90m";
        if (ss.active) {
            state_col = ss.compute_active ? "\x1b[1;96m" : "\x1b[92m";
            if (ss.compute_active)
                snprintf(state_str, sizeof(state_str), "ACTIVE*");
            else if (ss.compute_waiting)
                snprintf(state_str, sizeof(state_str), "WAIT");
            else
                snprintf(state_str, sizeof(state_str), "ACTIVE");
        } else {
            snprintf(state_str, sizeof(state_str), "FREE");
        }

        char pid_str[16], cpu_str[16];
        if (ss.active && ss.pid > 0)
            snprintf(pid_str, sizeof(pid_str), "%d", ss.pid);
        else
            snprintf(pid_str, sizeof(pid_str), "-");

        if (ss.active && ss.cpu_pct >= 0.0)
            snprintf(cpu_str, sizeof(cpu_str), "%5.1f%%", ss.cpu_pct);
        else
            snprintf(cpu_str, sizeof(cpu_str), "-");

        std::string quota = format_bytes_gib(ss.quota_bytes);
        if (!ss.active) {
            bool is_cfg = (ss.cfg_gb >= kMinClientHbmGb && ss.cfg_gb <= kMaxClientHbmGb);
            quota += is_cfg ? " (c)" : " (d)";
        }
        std::string phys = ss.active ? format_bytes_gib(ss.phys_bytes) : "-";
        std::string cmd = ss.active ? (ss.cmdline.empty() ? "-" : ss.cmdline) : "-";
        if ((int)cmd.size() > cmd_w) {
            cmd.resize((size_t)std::max(0, cmd_w - 3));
            cmd += "...";
        }

        snprintf(line, sizeof(line), " %-4d %-8s %s%-7s%s %-7s %6s %6s %-10s %-8s ",
                 s, kRoles[s], C(state_col), state_str, C("\x1b[0m"),
                 pid_str, cpu_str, format_rss(ss.rss_bytes).c_str(),
                 quota.c_str(), phys.c_str());
        row = line;
        row += cmd;
        emit_row(row);
    }

    if (tui) {
        snprintf(line, sizeof(line),
                 "\x1b[90m q quit  p pause  +/- rate (%.2gs)  1/2/3 series  c clear  (* active compute)\x1b[0m",
                 interval_sec);
        row = line;
        emit_row(row, true);
    }

    return out;
}

static bool parse_uint_arg(const char *s, unsigned int &out)
{
    if (!s || !s[0] || s[0] == '-' || s[0] == '+' || isspace((unsigned char)s[0]))
        return false;
    errno = 0;
    char *end = nullptr;
    unsigned long v = strtoul(s, &end, 10);
    if (errno != 0 || end == s || *end != '\0' || v > 1024)
        return false;
    out = (unsigned int)v;
    return true;
}

static bool parse_interval_arg(const char *s, double &out)
{
    if (!s || !s[0] || isspace((unsigned char)s[0]))
        return false;
    errno = 0;
    char *end = nullptr;
    double v = strtod(s, &end);
    if (errno != 0 || end == s || *end != '\0' || !std::isfinite(v) || v < 0.2 || v > 30.0)
        return false;
    out = v;
    return true;
}

static void print_usage()
{
    printf("Usage: hpu-mps top [options]\n"
           "  -d, --device <idx>      HPU device index (default: 0)\n"
           "  -i, --interval <sec>    Refresh interval in seconds (0.2..30.0, default: 1.0)\n"
           "  -1, --once              Print one snapshot and exit\n"
           "      --json              Print one JSON snapshot and exit\n"
           "  -h, --help              Show this help\n");
}

int main(int argc, char **argv)
{
    unsigned int dev_index = 0;
    bool dev_explicit = false;
    double interval_sec = 1.0;
    bool once = false;
    bool json_mode = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-d") == 0 || strcmp(argv[i], "--device") == 0) {
            if (i + 1 >= argc || !parse_uint_arg(argv[i + 1], dev_index)) {
                fprintf(stderr, "hpu-mps-top: invalid device index for '%s'\n", argv[i]);
                return 1;
            }
            dev_explicit = true;
            ++i;
        } else if (strcmp(argv[i], "-i") == 0 || strcmp(argv[i], "--interval") == 0) {
            if (i + 1 >= argc || !parse_interval_arg(argv[i + 1], interval_sec)) {
                fprintf(stderr, "hpu-mps-top: invalid interval for '%s' (must be 0.2..30.0)\n", argv[i]);
                return 1;
            }
            ++i;
        } else if (strcmp(argv[i], "-1") == 0 || strcmp(argv[i], "--once") == 0) {
            once = true;
        } else if (strcmp(argv[i], "--json") == 0) {
            json_mode = true;
            once = true;
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage();
            return 0;
        } else {
            fprintf(stderr, "hpu-mps-top: unknown option '%s'\n", argv[i]);
            print_usage();
            return 1;
        }
    }

    signal(SIGPIPE, SIG_IGN);

    bool stdout_tty = isatty(STDOUT_FILENO) != 0;
    bool stdin_tty = isatty(STDIN_FILENO) != 0;
    if (!stdout_tty || !stdin_tty)
        once = true;

    hlml::Client hlml_client;
    if (!hlml_client.init(dev_index) && dev_explicit) {
        bool out_of_range = hlml_client.device_count() > 0 && dev_index >= hlml_client.device_count();
        if (once || !hlml_client.has_lib() || out_of_range) {
            fprintf(stderr, "hpu-mps-top: failed to open HPU device %u via libhlml.so\n", dev_index);
            return 1;
        }
    }
    MpsReader mps_reader;

    if (once) {
        auto hw = hlml_client.sample();
        auto mps = mps_reader.sample();
        if (json_mode) {
            print_json(hw, mps);
        } else {
            int cols = 100, rows = 28;
            get_term_size(cols, rows);
            bool use_color = stdout_tty && (getenv("NO_COLOR") == nullptr);
            std::deque<PlotSample> empty;
            std::string frame = build_frame(hw, mps, empty, cols, rows, false, use_color,
                                            false, interval_sec, true, true, true);
            write_all(STDOUT_FILENO, frame.data(), frame.size());
        }
        return 0;
    }

    struct sigaction sa = {};
    sa.sa_handler = on_stop_signal;
    sigaction(SIGINT, &sa, nullptr);
    sigaction(SIGTERM, &sa, nullptr);
    sigaction(SIGHUP, &sa, nullptr);
    sa.sa_handler = on_winch;
    sigaction(SIGWINCH, &sa, nullptr);
    sa.sa_handler = on_tstp;
    sigaction(SIGTSTP, &sa, nullptr);
    sa.sa_handler = on_cont;
    sigaction(SIGCONT, &sa, nullptr);

    TerminalGuard term;
    if (!term.enter())
        return 1;

    std::deque<PlotSample> hist;
    constexpr size_t kMaxHist = 4096;
    bool paused = false;
    bool show_aip = true, show_hbm = true, show_pwr = true;
    static const double kRates[] = {0.25, 0.5, 1.0, 2.0, 5.0};
    int rate_idx = 0;
    double best_diff = 1e9;
    for (int i = 0; i < 5; ++i) {
        double d = fabs(interval_sec - kRates[i]);
        if (d < best_diff) {
            best_diff = d;
            rate_idx = i;
        }
    }

    hlml::Client::Telemetry hw;
    MpsSnapshot mps;
    double next_sample = 0.0;
    double last_reinit = monotonic_sec();
    bool had_first_sample = false;
    int prev_cols = 0, prev_rows = 0;

    while (!g_stop) {
        bool full_clear = false;
        if (g_tstp) {
            g_tstp = 0;
            term.leave();
            signal(SIGTSTP, SIG_DFL);
            raise(SIGTSTP);
            // Resumes here after SIGCONT
            struct sigaction tstp_sa = {};
            tstp_sa.sa_handler = on_tstp;
            sigaction(SIGTSTP, &tstp_sa, nullptr);
            if (!term.enter())
                break;
            g_resized = 1;
            full_clear = true;
        }
        if (g_cont) {
            g_cont = 0;
            if (!term.enter())
                break;
            g_resized = 1;
            full_clear = true;
        }

        double now = monotonic_sec();
        bool need_redraw = false;

        if (!paused && now >= next_sample) {
            bool need_reinit = (!hlml_client.ready() || (had_first_sample && !hw.operational));
            if (need_reinit && (now - last_reinit >= 5.0)) {
                hlml_client.reinit(dev_index);
                last_reinit = monotonic_sec();
            }
            hw = hlml_client.sample();
            had_first_sample = true;
            mps = mps_reader.sample();
            if (hw.operational) {
                double hbm_pct = (hw.has_mem && hw.mem_total > 0)
                                     ? (100.0 * (double)hw.mem_used / (double)hw.mem_total)
                                     : 0.0;
                double pwr_pct = (hw.has_power && hw.has_power_limit && hw.power_limit_w > 0.0)
                                     ? (100.0 * hw.power_w / hw.power_limit_w)
                                     : 0.0;
                hist.push_back({hw.has_util ? (double)hw.aip_util : 0.0, hbm_pct, pwr_pct, false});
                if (hist.size() > kMaxHist)
                    hist.pop_front();
            } else if (!hist.empty()) {
                hist.push_back({0.0, 0.0, 0.0, true});
                if (hist.size() > kMaxHist)
                    hist.pop_front();
            }
            next_sample = monotonic_sec() + interval_sec;
            need_redraw = true;
        }

        if (g_resized) {
            g_resized = 0;
            need_redraw = true;
        }

        int cols = 100, rows = 28;
        get_term_size(cols, rows);
        if (cols != prev_cols || rows != prev_rows) {
            full_clear = true;
            need_redraw = true;
        }

        if (need_redraw) {
            prev_cols = cols;
            prev_rows = rows;
            std::string frame = build_frame(hw, mps, hist, cols, rows, true, true, paused,
                                            interval_sec, show_aip, show_hbm, show_pwr, full_clear);
            if (!write_all(STDOUT_FILENO, frame.data(), frame.size()))
                break;
        }

        int timeout_ms = 200;
        if (!paused) {
            double rem = (next_sample - monotonic_sec()) * 1000.0;
            rem = std::max(20.0, std::min(250.0, rem));
            timeout_ms = (int)rem;
        }

        struct pollfd pfd = {STDIN_FILENO, POLLIN, 0};
        int pr = poll(&pfd, 1, timeout_ms);
        if (pr > 0) {
            if (pfd.revents & (POLLHUP | POLLERR | POLLNVAL))
                break;
            if (pfd.revents & POLLIN) {
                char ch = 0;
                ssize_t nr = read(STDIN_FILENO, &ch, 1);
                if (nr == 0)
                    break;
                if (nr == 1) {
                    if (ch == '\x1b') {
                        drain_escape_sequence();
                    } else if (ch == 'q' || ch == 'Q' || ch == 3) {
                        break;
                    } else if (ch == 'p' || ch == 'P' || ch == ' ') {
                        paused = !paused;
                        if (!paused)
                            next_sample = 0.0;
                        g_resized = 1;
                    } else if (ch == 'c' || ch == 'C') {
                        hist.clear();
                        g_resized = 1;
                    } else if (ch == '1') {
                        show_aip = !show_aip;
                        g_resized = 1;
                    } else if (ch == '2') {
                        show_hbm = !show_hbm;
                        g_resized = 1;
                    } else if (ch == '3') {
                        show_pwr = !show_pwr;
                        g_resized = 1;
                    } else if (ch == '+' || ch == '=') {
                        if (rate_idx > 0) {
                            interval_sec = kRates[--rate_idx];
                            next_sample = 0.0;
                            g_resized = 1;
                        }
                    } else if (ch == '-' || ch == '_') {
                        if (rate_idx < 4) {
                            interval_sec = kRates[++rate_idx];
                            next_sample = 0.0;
                            g_resized = 1;
                        }
                    }
                }
            }
        }
    }

    term.leave();
    return 0;
}
