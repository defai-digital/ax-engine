#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static atomic_bool observed_earlier_payload;
static atomic_bool reported_fault;

static void mark_once(atomic_bool *mark, const char *message, size_t length) {
    if (!atomic_exchange(mark, true)) {
        (void)write(STDERR_FILENO, message, length);
    }
}

static ssize_t fail_selected_pread(int descriptor, void *buffer, size_t count, off_t offset) {
    const char *earlier = getenv("AX_TEST_EARLIER_SHARD");
    const char *target = getenv("AX_TEST_FAILED_SHARD");
    char path[PATH_MAX];
    if (!earlier || !target || count == 0 || offset < 8 ||
        fcntl(descriptor, F_GETPATH, path) < 0) {
        return pread(descriptor, buffer, count, offset);
    }
    if (strcmp(path, target) == 0) {
        const char message[] = "ax_fixture selected shard pread returned EIO\n";
        mark_once(&reported_fault, message, sizeof(message) - 1);
        errno = EIO;
        return -1;
    }
    ssize_t result = pread(descriptor, buffer, count, offset);
    if (result > 0 && strcmp(path, earlier) == 0) {
        const char message[] = "ax_fixture earlier shard payload read\n";
        mark_once(&observed_earlier_payload, message, sizeof(message) - 1);
    }
    return result;
}

__attribute__((used, section("__DATA,__interpose")))
static const struct {
    const void *replacement;
    const void *original;
} interpose_pread = { (const void *)fail_selected_pread, (const void *)pread };
