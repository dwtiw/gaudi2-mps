// SPDX-License-Identifier: Apache-2.0
/* Round-trip a known pattern through the card and check it comes back intact.
 *
 * Does H2D, D2D and D2H copies of a fixed pattern and compares, which is the
 * cheapest functional test of the DMA path. `hpu-mps start` also runs it in
 * `serve` mode as the Slot 0 primary.
 *
 *   scal-xfer                    one process, one round trip
 *   scal-xfer hold <s>           same, then keep the device for <s> seconds
 *   scal-xfer park <s>           same, then sleep with the stream still alive
 *   scal-xfer serve              same, then hold the stream until SIGTERM/SIGINT
 *   scal-xfer loop <n>           repeat the round trip n times
 *   scal-xfer multistream <n>    loop with H2D, D2D and D2H on separate streams
 *   scal-xfer idle <s>           acquire and hold without submitting anything
 *
 * Exit status is 0 only when the pattern came back byte for byte.
 */
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>

#include <habanalabs/synapse_api.h>

#define BUF_SIZE	(4u << 20)
#define PATTERN_SEED	0xA5A50000u

static volatile sig_atomic_t g_stop;

static void on_stop(int sig)
{
	(void)sig;
	g_stop = 1;
}

static void wait_for_stop(void)
{
	struct sigaction sa;
	sigset_t block, old;

	sigemptyset(&block);
	sigaddset(&block, SIGTERM);
	sigaddset(&block, SIGINT);
	sigprocmask(SIG_BLOCK, &block, &old);
	memset(&sa, 0, sizeof(sa));
	sa.sa_handler = on_stop;
	sigaction(SIGTERM, &sa, NULL);
	sigaction(SIGINT, &sa, NULL);
	while (!g_stop)
		sigsuspend(&old);
	sigprocmask(SIG_SETMASK, &old, NULL);
}

static int verify(const uint32_t *a, const uint32_t *b, size_t n)
{
	size_t i;

	for (i = 0; i < n; i++)
		if (a[i] != b[i]) {
			printf("mismatch at word %zu: expected %#x got %#x\n",
			       i, a[i], b[i]);
			return -1;
		}

	return 0;
}

int main(int argc, char **argv)
{
	int hold_s = (argc > 2 && strcmp(argv[1], "hold") == 0) ? atoi(argv[2]) : 0;
	/* loop <n>: with two processes, the one still completing shows the
	 * device as a whole is not what blocks the other. */
	int iters = (argc > 2 && strcmp(argv[1], "loop") == 0) ? atoi(argv[2]) : 1;
	int multistream = (argc > 1 && strcmp(argv[1], "multistream") == 0) ? 1 : 0;
	if (multistream && argc > 2)
		iters = atoi(argv[2]);
	/* park <s> / serve keep the stream and its buffers alive, unlike hold. */
	int park_s = (argc > 2 && strcmp(argv[1], "park") == 0) ? atoi(argv[2]) : 0;
	int serve = (argc > 1 && strcmp(argv[1], "serve") == 0) ? 1 : 0;
	/*
	 * idle <s> acquires the device, runs whatever initialisation the runtime
	 * does, and then holds without submitting anything. It separates "the
	 * second process breaks because a first one exists" from "the second
	 * process breaks because the first one has been submitting".
	 */
	int idle_s = (argc > 2 && strcmp(argv[1], "idle") == 0) ? atoi(argv[2]) : 0;
	int iter = 0;
	synStreamHandle stream = NULL, stream2 = NULL, stream3 = NULL;
	synDeviceId dev;
	synStatus st;
	void *host_a = NULL, *host_b = NULL;
	uint64_t dev_a = 0, dev_b = 0;
	uint32_t i, nwords = BUF_SIZE / sizeof(uint32_t);
	int rc = 0;

	st = synInitialize();
	printf("synInitialize      -> %d\n", st);
	if (st != synSuccess && st != synAlreadyInitialized)
		return 1;

	st = synDeviceAcquireByDeviceType(&dev, synDeviceGaudi2);
	printf("synDeviceAcquire   -> %d\n", st);
	if (st != synSuccess)
		return 1;

	st = synDeviceMalloc(dev, BUF_SIZE, 0, 0, &dev_a);
	printf("synDeviceMalloc A  -> %d\n", st);
	if (st != synSuccess)
		return 1;

	st = synDeviceMalloc(dev, BUF_SIZE, 0, 0, &dev_b);
	printf("synDeviceMalloc B  -> %d\n", st);
	if (st != synSuccess)
		return 1;

	st = synHostMalloc(dev, BUF_SIZE, 0, &host_a);
	printf("synHostMalloc A    -> %d\n", st);
	if (st != synSuccess || !host_a) {
		rc = 1;
		goto release;
	}

	st = synHostMalloc(dev, BUF_SIZE, 0, &host_b);
	printf("synHostMalloc B    -> %d\n", st);
	if (st != synSuccess || !host_b) {
		rc = 1;
		goto release;
	}

	for (i = 0; i < nwords; i++)
		((uint32_t *) host_a)[i] = PATTERN_SEED ^ i;
	memset(host_b, 0, BUF_SIZE);

	st = synStreamCreateGeneric(&stream, dev, 0);
	printf("synStreamCreate    -> %d\n", st);
	if (st != synSuccess) {
		rc = 1;
		goto release;
	}
	if (multistream) {
		st = synStreamCreateGeneric(&stream2, dev, 0);
		printf("synStreamCreate 2  -> %d\n", st);
		if (st != synSuccess) {
			rc = 1;
			goto destroy_stream;
		}
		st = synStreamCreateGeneric(&stream3, dev, 0);
		printf("synStreamCreate 3  -> %d\n", st);
		if (st != synSuccess) {
			rc = 1;
			goto destroy_stream;
		}
	}

	if (idle_s > 0) {
		printf("idle for %d s (pid %d)\n", idle_s, getpid());
		fflush(stdout);
		sleep(idle_s);
		goto destroy_stream;
	}

	for (iter = 0 ; iter < iters ; iter++) {
		synStreamHandle s_h2d = stream;
		synStreamHandle s_d2d = multistream ? stream2 : stream;
		synStreamHandle s_d2h = multistream ? stream3 : stream;
		/* Host -> A, then A -> B on the device, then A -> host B. The
		 * device-side copy exercises a DMA that never touches the host at
		 * all, so a failure on the second leg but not the first points at
		 * the host path. */
		st = synMemCopyAsync(s_h2d, (uint64_t) (uintptr_t) host_a, BUF_SIZE,
				     dev_a, HOST_TO_DRAM);
		if (!iter)
			printf("synMemCopyAsync H2D-> %d\n", st);
		if (multistream && st == synSuccess)
			st = synStreamSynchronize(s_h2d);
		if (st == synSuccess)
			st = synMemCopyAsync(s_d2d, dev_a, BUF_SIZE, dev_b, DRAM_TO_DRAM);
		if (!iter)
			printf("synMemCopyAsync D2D-> %d\n", st);
		if (multistream && st == synSuccess)
			st = synStreamSynchronize(s_d2d);
		if (st == synSuccess)
			st = synMemCopyAsync(s_d2h, dev_b, BUF_SIZE,
					     (uint64_t) (uintptr_t) host_b, DRAM_TO_HOST);
		if (!iter)
			printf("synMemCopyAsync D2H-> %d\n", st);

		if (st == synSuccess) {
			st = synStreamSynchronize(s_d2h);
			if (!iter)
				printf("synStreamSynchronize-> %d\n", st);
		}

		if (st != synSuccess) {
			printf("iter %d -> failed (%d)\n", iter, st);
			rc = 1;
			goto destroy_stream;
		}

		if (verify((const uint32_t *) host_a, (const uint32_t *) host_b, nwords)) {
			printf("roundTrip           -> CORRUPTED\n");
			rc = 1;
			break;
		}

		printf("iter %d -> intact (%u bytes)\n", iter, BUF_SIZE);
	}

	/* Before the stream is destroyed: the control page belongs to it. */
	if (park_s > 0) {
		printf("parking for %d s with the stream alive (pid %d)\n",
		       park_s, getpid());
		fflush(stdout);
		sleep(park_s);
	}
	if (serve && !rc) {
		printf("serving until SIGTERM (pid %d)\n", getpid());
		fflush(stdout);
		wait_for_stop();
	}

destroy_stream:
	if (stream3)
		synStreamDestroy(stream3);
	if (stream2)
		synStreamDestroy(stream2);
	if (stream)
		synStreamDestroy(stream);

release:
	if (host_a)
		synHostFree(dev, host_a, 0);
	if (host_b)
		synHostFree(dev, host_b, 0);
	if (dev_a)
		synDeviceFree(dev, dev_a, 0);
	if (dev_b)
		synDeviceFree(dev, dev_b, 0);

	if (hold_s > 0) {
		printf("holding device for %d s (pid %d)\n", hold_s, getpid());
		fflush(stdout);
		sleep(hold_s);
	}

	st = synDeviceRelease(dev);
	printf("synDeviceRelease   -> %d\n", st);
	if (st != synSuccess)
		rc = 1;

	st = synDestroy();
	printf("synDestroy         -> %d\n", st);

	return rc;
}
