#include <stdint.h>
#include <stdlib.h>

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
