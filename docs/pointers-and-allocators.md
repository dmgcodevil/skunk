# Skunk Pointer And Allocator Design

This document defines the intended v1 direction for pointers, allocators, arenas, and ownership in Skunk.

## Goals

- Keep fixed arrays and structs as value types
- Make heap allocation explicit
- Support allocator-backed single objects and buffers
- Add pointer semantics without exposing unsafe raw memory too early
- Leave room for a later Zig-like custom allocator story

## Value And Reference Model

Skunk should distinguish these forms clearly:

- `T`
  A value of type `T`
- `*T`
  A pointer/reference to one `T`
- `[]T`
  A non-owning slice/view over many `T`
- `[N]T`
  A fixed-size value array containing exactly `N` `T` values

Examples:

```skunk
point: Point;        // value
point_ptr: *Point;   // pointer/reference to one Point
bytes: []byte;       // slice/view over bytes
row: [16]int;        // fixed-size value array
```

## Core Rule

- Plain `T` uses value semantics
- Allocator-created single objects use `*T`
- Allocator-created buffers use `[]T`

Examples:

```skunk
function make_point_value(): Point {
    return Point { x: 1, y: 2 };
}

function make_point_ref(alloc: Allocator): *Point {
    p: *Point = Point::create(alloc);
    p.x = 1;
    p.y = 2;
    return p;
}

function make_bytes(alloc: Allocator, n: int): []byte {
    return []byte::alloc(alloc, n);
}
```

## Pointer Syntax

### V1 Pointer Type

The first pointer form should be:

```skunk
*T
```

Examples:

```skunk
*Point
*int
*[16]byte
```

### V1 Pointer Operations

Skunk v1 should support:

- pointer types: `*T`
- pointer-returning allocator APIs
- field access through pointers with automatic dereference
- method calls through pointers with automatic dereference
- assignment through pointer field access

Examples:

```skunk
p: *Point = Point::create(alloc);
p.x = 10;
print(p.x);
p.set_x(20);
```

### Explicit Dereference

Skunk should reserve explicit dereference syntax for later, but the preferred direction is:

```skunk
p.*
```

Reason:

- it avoids overloading prefix `*`
- it aligns well with a future `*T` syntax
- it keeps field access ergonomic because `p.x` can auto-deref

### Address-Of

Skunk v1 should **not** expose a general address-of operator yet.

That means:

- pointers are produced by allocator/runtime APIs
- pointers are not yet produced freely from arbitrary locals using `&x`

Reason:

- allowing `&local` immediately creates lifetime/escape problems
- Skunk does not yet have borrow checking or escape analysis
- allocator-created pointers solve the main must-have use case first

Address-of can be added later in an `unsafe` phase.

## Auto-Deref Rules

Skunk should auto-deref `*T` for:

- field access
- method calls

Examples:

```skunk
p: *Point = Point::create(alloc);
p.x = 1;
print(p.x);
p.set_x(3);
```

This should behave as if the pointer is dereferenced to access the underlying value.

Skunk v1 should **not** auto-deref everywhere. In particular:

- `Point` and `*Point` are still distinct types
- passing `*Point` where `Point` is expected should not silently copy
- passing `Point` where `*Point` is expected should not silently take an address

## Methods And Receivers

Method dispatch should work on both values and pointers:

- calling on `T` lvalues uses the address of the existing storage
- calling on `*T` uses the pointer directly
- calling on temporaries uses temporary storage

Examples:

```skunk
point: Point = Point { x: 0, y: 0 };
point.set_x(1);   // mutates point storage

ptr: *Point = Point::create(alloc);
ptr.set_x(2);     // mutates pointed-to storage
```

This keeps existing struct method syntax useful after pointer types are added.

## Allocator Is Built-In In V1

`Allocator` should be a core runtime type, not a user-implemented Skunk type in v1.

Reason:

- it gives Skunk explicit allocation immediately
- it avoids requiring raw memory primitives too early
- it leaves custom allocators for a later `unsafe` phase

Skunk v1 should also include:

- `Arena`
- a process/system allocator entry point

## V1 Allocator API

Recommended v1 surface:

```skunk
heap: Allocator = System::allocator();

p: *Point = Point::create(heap);
heap.destroy(p);

buf: []byte = []byte::alloc(heap, 128);
heap.free(buf);
```

Recommended built-ins:

- `System::allocator(): Allocator`
- `T::create(alloc: Allocator): *T`
- `alloc.destroy(ptr: *T): void`
- `[]T::alloc(alloc: Allocator, len: int): []T`
- `alloc.free(buf: []T): void`

Optional later additions:

- `[]T::alloc_fill(alloc, len, value)`
- `alloc.clone(buf)`
- `alloc.resize(buf, new_len)`

## V1 Arena API

Recommended surface:

```skunk
heap: Allocator = System::allocator();
arena: Arena = Arena::init(heap);
alloc: Allocator = arena.allocator();

p: *Point = Point::create(alloc);
buf: []int = []int::alloc(alloc, 64);

arena.reset();
arena.deinit();
```

Recommended built-ins:

- `Arena::init(backing: Allocator): Arena`
- `arena.allocator(): Allocator`
- `arena.reset(): void`
- `arena.deinit(): void`

Notes:

- `reset` invalidates memory allocated from that arena
- `deinit` releases arena-owned memory
- individual `destroy`/`free` through an arena allocator may be supported but should not be the preferred style

## Ownership And Return Rules

### Values

Returning `T` returns a value:

```skunk
function f(): Point {
    return Point { x: 1, y: 2 };
}
```

Semantically this is a value return. The compiler may optimize copies, but the language meaning is value-based.

### Pointers

Returning `*T` returns a reference:

```skunk
function f(alloc: Allocator): *Point {
    p: *Point = Point::create(alloc);
    return p;
}
```

No value copy is implied. The result refers to the same allocated object.

### Slices

Returning `[]T` returns a slice header:

```skunk
function f(alloc: Allocator): []byte {
    return []byte::alloc(alloc, 32);
}
```

The slice header is copied, but the underlying storage is not.

## Lifetime Rules

Skunk v1 should use these lifetime rules:

- `[N]T` is owned by the enclosing value/storage
- `[]T` does not own storage
- `*T` does not own storage
- the allocator or owning value determines lifetime

Programmer responsibility in v1:

- do not use a `*T` after its allocator has destroyed it
- do not use a `[]T` after its backing storage is freed or reset
- do not return or store references to memory whose lifetime has ended

Skunk v1 will likely rely on programmer discipline here rather than a borrow checker.

## Why No User-Implemented Allocators Yet

To implement allocators in Skunk itself, user code would need lower-level memory primitives:

- raw byte pointers
- allocation/reallocation/free at byte granularity
- pointer casts
- pointer offset/arithmetic
- memory copy/set
- `size_of(T)`
- `align_of(T)`
- probably `unsafe { ... }`

That is too much surface area for the current stage of the language.

So the recommended phases are:

### Phase 1

- built-in `Allocator`
- built-in `Arena`
- `*T`
- allocator-created single objects and slices

### Phase 2

- explicit dereference `p.*`
- nullable pointers like `?*T`
- maybe address-of `&x` in restricted or unsafe contexts

### Phase 3

- raw memory primitives
- `unsafe`
- user-defined allocators implemented in Skunk

## Future Raw Memory Layer

When Skunk is ready for user-written allocators, the likely additional surface is:

- `*byte`
- many-item/raw pointer forms
- `&expr`
- `p.*`
- `size_of(T)`
- `align_of(T)`
- `mem.copy(dst, src, len)`
- `mem.set(dst, byte, len)`
- raw allocate/reallocate/free hooks
- pointer casts
- `unsafe { ... }`

At that point Skunk could support an allocator interface implemented in Skunk itself.

## Recommended Next Implementation Order

1. Add `*T` as a first-class type
2. Add built-in `Allocator` and `Arena`
3. Add `create/destroy` and `[]T::alloc/free`
4. Auto-deref field/method access for `*T`
5. Add ownership/lifetime docs and compiler diagnostics
6. Delay raw pointers and user-defined allocators until later
