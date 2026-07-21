#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void skunk_panic_index_out_of_bounds(int64_t index, int64_t length) {
    fprintf(stderr,
            "panic: index %lld out of bounds for length %lld\n",
            (long long)index,
            (long long)length);
    fflush(stderr);
    abort();
}

void skunk_panic_slice_range_out_of_bounds(int64_t start,
                                            int64_t end,
                                            int64_t length) {
    fprintf(stderr,
            "panic: slice range [%lld:%lld] out of bounds for length %lld\n",
            (long long)start,
            (long long)end,
            (long long)length);
    fflush(stderr);
    abort();
}

typedef struct SkunkArenaNode {
    void *memory;
    struct SkunkArenaNode *next;
} SkunkArenaNode;

typedef struct SkunkAllocator SkunkAllocator;

typedef struct SkunkArena {
    SkunkArenaNode *head;
    SkunkAllocator *allocator;
} SkunkArena;

struct SkunkAllocator {
    int kind;
    void *state;
};

enum {
    SKUNK_ALLOC_SYSTEM = 0,
    SKUNK_ALLOC_ARENA = 1,
};

static SkunkAllocator skunk_global_allocator = {SKUNK_ALLOC_SYSTEM, NULL};

static void *skunk_zero_alloc(size_t size) {
    if (size == 0) {
        size = 1;
    }
    return calloc(1, size);
}

void *skunk_system_allocator(void) {
    return &skunk_global_allocator;
}

void *skunk_arena_init(void *backing_allocator) {
    (void)backing_allocator;
    SkunkArena *arena = (SkunkArena *)calloc(1, sizeof(SkunkArena));
    SkunkAllocator *allocator = (SkunkAllocator *)calloc(1, sizeof(SkunkAllocator));
    allocator->kind = SKUNK_ALLOC_ARENA;
    allocator->state = arena;
    arena->allocator = allocator;
    return arena;
}

void *skunk_arena_allocator(void *arena_ptr) {
    SkunkArena *arena = (SkunkArena *)arena_ptr;
    if (arena == NULL) {
        return NULL;
    }
    return arena->allocator;
}

void skunk_arena_reset(void *arena_ptr) {
    SkunkArena *arena = (SkunkArena *)arena_ptr;
    if (arena == NULL) {
        return;
    }
    SkunkArenaNode *node = arena->head;
    while (node != NULL) {
        SkunkArenaNode *next = node->next;
        free(node->memory);
        free(node);
        node = next;
    }
    arena->head = NULL;
}

void skunk_arena_deinit(void *arena_ptr) {
    SkunkArena *arena = (SkunkArena *)arena_ptr;
    if (arena == NULL) {
        return;
    }
    skunk_arena_reset(arena_ptr);
    free(arena->allocator);
    free(arena);
}

static void *skunk_alloc_impl(void *allocator_ptr, size_t size) {
    SkunkAllocator *allocator = (SkunkAllocator *)allocator_ptr;
    if (allocator == NULL || allocator->kind == SKUNK_ALLOC_SYSTEM) {
        return skunk_zero_alloc(size);
    }

    if (allocator->kind == SKUNK_ALLOC_ARENA) {
        SkunkArena *arena = (SkunkArena *)allocator->state;
        void *memory = skunk_zero_alloc(size);
        SkunkArenaNode *node = (SkunkArenaNode *)calloc(1, sizeof(SkunkArenaNode));
        node->memory = memory;
        node->next = arena->head;
        arena->head = node;
        return memory;
    }

    return skunk_zero_alloc(size);
}

static int skunk_arena_release_node(SkunkArena *arena, void *memory) {
    if (arena == NULL || memory == NULL) {
        return 0;
    }

    SkunkArenaNode *previous = NULL;
    SkunkArenaNode *node = arena->head;
    while (node != NULL) {
        if (node->memory == memory) {
            if (previous == NULL) {
                arena->head = node->next;
            } else {
                previous->next = node->next;
            }
            free(node);
            return 1;
        }
        previous = node;
        node = node->next;
    }

    return 0;
}

void *skunk_alloc_create(void *allocator_ptr, uint64_t size) {
    return skunk_alloc_impl(allocator_ptr, (size_t)size);
}

void *skunk_alloc_buffer(void *allocator_ptr, uint64_t elem_size, int32_t len) {
    size_t count = len < 0 ? 0 : (size_t)len;
    return skunk_alloc_impl(allocator_ptr, (size_t)elem_size * count);
}

void skunk_alloc_destroy(void *allocator_ptr, void *memory) {
    SkunkAllocator *allocator = (SkunkAllocator *)allocator_ptr;
    if (memory == NULL) {
        return;
    }
    if (allocator == NULL || allocator->kind == SKUNK_ALLOC_SYSTEM) {
        free(memory);
        return;
    }
    if (allocator->kind == SKUNK_ALLOC_ARENA) {
        SkunkArena *arena = (SkunkArena *)allocator->state;
        if (skunk_arena_release_node(arena, memory)) {
            free(memory);
        }
    }
}

void skunk_alloc_free(void *allocator_ptr, void *memory) {
    skunk_alloc_destroy(allocator_ptr, memory);
}

/* ---------------------------------------------------------------------------
 * Native test harness.
 *
 * `skunk test` generates a runner main that drives these hooks:
 *   skunk_test_begin(name) -> run test body -> skunk_test_end()
 * and finally returns skunk_test_summary() as the process exit code.
 * Assertion helpers (expect / expect_eq / fail) mark the current test failed
 * but keep executing so a single run reports every failing assertion.
 * ------------------------------------------------------------------------- */

#include <time.h>

static int skunk_test_total = 0;
static int skunk_test_failures = 0;
static int skunk_test_current_failed = 0;
static const char *skunk_test_current_name = "";
static double skunk_test_started_ms = 0.0;

static double skunk_test_now_ms(void) {
    struct timespec ts;
#if defined(CLOCK_MONOTONIC)
    clock_gettime(CLOCK_MONOTONIC, &ts);
#else
    clock_gettime(CLOCK_REALTIME, &ts);
#endif
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
}

void skunk_test_begin(const char *name) {
    skunk_test_total += 1;
    skunk_test_current_failed = 0;
    skunk_test_current_name = name == NULL ? "" : name;
    skunk_test_started_ms = skunk_test_now_ms();
}

void skunk_test_expect(_Bool condition) {
    if (!condition) {
        skunk_test_current_failed = 1;
        printf("    expect failed in test \"%s\"\n", skunk_test_current_name);
    }
}

void skunk_test_expect_eq_int(int32_t expected, int32_t actual) {
    if (expected != actual) {
        skunk_test_current_failed = 1;
        printf("    expect_eq failed in test \"%s\": expected %d, got %d\n",
               skunk_test_current_name, expected, actual);
    }
}

void skunk_test_expect_eq_long(int64_t expected, int64_t actual) {
    if (expected != actual) {
        skunk_test_current_failed = 1;
        printf("    expect_eq failed in test \"%s\": expected %lld, got %lld\n",
               skunk_test_current_name, (long long)expected, (long long)actual);
    }
}

void skunk_test_fail(void) {
    skunk_test_current_failed = 1;
    printf("    fail() called in test \"%s\"\n", skunk_test_current_name);
}

void skunk_test_end(void) {
    double elapsed = skunk_test_now_ms() - skunk_test_started_ms;
    if (skunk_test_current_failed) {
        skunk_test_failures += 1;
        printf("FAIL %s (%.2f ms)\n", skunk_test_current_name, elapsed);
    } else {
        printf("PASS %s (%.2f ms)\n", skunk_test_current_name, elapsed);
    }
}

int32_t skunk_test_summary(void) {
    printf("\n%d test%s, %d passed, %d failed\n",
           skunk_test_total,
           skunk_test_total == 1 ? "" : "s",
           skunk_test_total - skunk_test_failures,
           skunk_test_failures);
    return skunk_test_failures > 0 ? 1 : 0;
}
