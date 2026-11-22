#include <libproc.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/proc_info.h>

int main(int argc, char **argv) {
    if (argc != 2) {
        return 2;
    }
    char *end;
    long pid = strtol(argv[1], &end, 10);
    if (*end != '\0' || pid <= 0 || pid > INT32_MAX) {
        return 2;
    }

    uint64_t threads[4096];
    int bytes = proc_pidinfo((int)pid, PROC_PIDLISTTHREADS, 0, threads, sizeof(threads));
    if (bytes <= 0 || (size_t)bytes >= sizeof(threads) || (size_t)bytes % sizeof(threads[0])) {
        return 1;
    }

    int inspected = 0;
    int matching = 0;
    const char prefix[] = "ax-ple-reader-";
    for (size_t i = 0; i < (size_t)bytes / sizeof(threads[0]); ++i) {
        struct proc_threadinfo info = {0};
        int copied = proc_pidinfo((int)pid, PROC_PIDTHREADINFO, threads[i], &info, sizeof(info));
        if (copied == sizeof(info)) {
            ++inspected;
            if (strncmp(info.pth_name, prefix, sizeof(prefix) - 1) == 0) {
                ++matching;
            }
        }
    }
    if (!inspected) {
        return 1;
    }
    printf("%d\n", matching);
    return 0;
}
