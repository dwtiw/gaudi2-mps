// SPDX-License-Identifier: Apache-2.0
/*
 * mps_shm.h -- Shared constants and IPC layout between libhpu_mps.so and hpu-mps-top.
 */
#pragma once

#include <limits.h>
#include <pthread.h>
#include <stdint.h>

static constexpr int kMaxSlots = 3;

static constexpr uint64_t kHostSharedVaBase      = 0xfff0f80090000000ULL;
static constexpr uint64_t kHostSharedSliceSize   = 0x04000000ULL; // 64 MiB per slot (3 * 64 MiB = 192 MiB <= 256 MiB)
static constexpr uint64_t kHbmPoolVaBase         = 0x1001600150000000ULL;
static constexpr uint64_t kHbmArcVaEnd           = 0x1001700000000000ULL; // RESERVED_VA_RANGE_FOR_ARC_ON_HBM_END + 1
static constexpr uint64_t kMaxHbmVaSpan          = kHbmArcVaEnd - kHbmPoolVaBase; // ~1018.75 GiB ARC-reachable VA span
static constexpr uint64_t kDramPageBytes         = 768ULL << 20;  // 768 MiB Gaudi2 DRAM physical allocation granularity
static constexpr uint64_t kHbmAlignUnit          = 0x300000000ULL; // 12 GiB VA alignment (LCM of 4 GiB ARC region & 768 MiB page)
static constexpr uint64_t kDaemonHbmMinBytes     = kDramPageBytes;       // 768 MiB minimum (1 DRAM page)
static constexpr uint64_t kDaemonHbmMaxBytes     = 6ULL * kDramPageBytes; // 4608 MiB maximum override (6 DRAM pages)
static constexpr uint64_t kDaemonHbmCapBytes     = 2ULL * kDramPageBytes; // 1536 MiB (1.5 GiB, 2 DRAM pages, zero rounding waste)
static constexpr uint64_t kDefaultDaemonClientGb = 36ULL;         // 36 GiB default for Slot 1 & 2 when daemon holds Slot 0
static constexpr uint64_t kDefaultActiveSlotGb   = 24ULL;         // 24 GiB default when 3 active workloads share card
static constexpr uint64_t kMinClientHbmGb        = 1ULL;
static constexpr uint64_t kMaxClientHbmGb        = 90ULL;

static constexpr uint32_t kCoeffTableUnknown     = 0u;
static constexpr uint32_t kCoeffTablePending     = 1u;
static constexpr uint32_t kCoeffTableReady       = 2u;
static constexpr uint32_t kCoeffTableFailed      = 3u;

static constexpr uint64_t kComputeSyncMagic      = 0x4850554d50534334ULL; // "HPUPSC4"
static constexpr const char *kComputeSyncFile    = "compute-sync.v4";
static constexpr const char *kHbmStateFile       = "hbm.state";
static constexpr const char *kReadyFile          = "slot-0.ready";

static inline uint64_t align_up_dram_page(uint64_t bytes)
{
    if (bytes == 0)
        return 0;
    if (bytes > UINT64_MAX - (kDramPageBytes - 1))
        return UINT64_MAX;
    return ((bytes + kDramPageBytes - 1) / kDramPageBytes) * kDramPageBytes;
}

struct ComputeLaunchSyncShm {
    alignas(4096) volatile uint64_t cq_page[512];
    uint64_t magic;
    pthread_mutex_t mu;
    volatile uint64_t cq_page_dev_va;
    volatile int32_t active_slot;
    volatile int32_t active_slot_pid;
    volatile uint32_t active_cq_idx;
    volatile uint32_t burst_count;
    volatile uint64_t active_target_val;
    volatile uint64_t last_slot_switch_ns;
    volatile uint32_t waiting_mask;
    volatile uint32_t cfg_burst_limit;
    volatile uint32_t cfg_max_inflight;
    volatile uint32_t cfg_pdma_overlap;
    volatile uint32_t cfg_step_yield;
    volatile uint32_t slot_switches;
    volatile uint64_t last_enqueue_ns;
    volatile uint64_t cfg_idle_steal_ns;
    volatile uint32_t slot_burst_limit[8];
    volatile uint64_t active_rx_base;
    volatile uint64_t cfg_quantum_ns;
    volatile uint32_t cfg_min_burst;
    volatile uint32_t coeff_table_state;
    volatile int32_t coeff_table_pid;
    volatile uint32_t reserved_cfg;
    volatile uint64_t slot_dram_handles[8][64];
    volatile uint64_t slot_mapped_vas[8][256];
    // Guards slot_dram_handles/slot_mapped_vas only; mu is held across recipe
    // launches and must not be taken from memory map/alloc paths.
    pthread_mutex_t bk_mu;
};
