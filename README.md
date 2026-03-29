![Alt text](logo.jpg)

# Skunk Programming Language

**Skunk** is a statically typed programming language with both an interpreter and an LLVM-based native compiler. It provides a clean syntax for working with structured data, control flow, and functions while supporting extensible features like user-defined types and type inference.

## Features

- **Basic Types**: `byte`, `short`, `int`, `long`, `float`, `double`, `boolean`, `char`, `string`
- **User-Defined Structs**: Define custom types with fields and methods
- **Control Flow**: `if`, `for` loops, and blocks for scoped variable overrides
- **Arrays**: Fixed-size arrays with zero initialization, explicit fill initialization, and slice types
- **Pointers and Allocation**: `*T`, `T::create(alloc)`, `[]T::alloc(alloc, len)`, `alloc.destroy(...)`, `alloc.free(...)`, and `Arena`
- **Functions**: First-class functions with support for closures, lambdas, and higher-order programming
- **Type Checking**: Ensures type correctness at parse-time with detailed error messages
- **Type Inference**: Planned for a cleaner developer experience
- **String Interpolation and Concatenation**: Upcoming for intuitive string operations
- **Generics**: Upcoming for flexible and reusable data structures

Current design notes:

- [Pointer and Allocator Design](/Users/dmgcodevil/dev/skunk-llvm/skunk/docs/pointers-and-allocators.md)

## Example Programs

### Primitive Types
```skunk
function main(): void {
    b: byte = 10;
    s: short = 20;
    i: int = 30;
    l: long = 40L;
    f: float = 1.5f;
    d: double = l + f;
    c: char = 'A';
    ok: boolean = c == 'A';

    print(c);
    print(d);
    print(ok);
}
```

### Hello, World
```skunk
function main(): void {
    print("Hello, World!");
}
```

### Variables and Control Flow
```skunk
function main(): void {
    x: int = 10;
    if (x > 5) {
        print("x is greater than 5");
    } else {
        print("x is not greater than 5");
    }
}
```

### Structs and Methods
```skunk
struct Point {
    x: int;
    y: int;

    function set_x(self, x: int): void {
        self.x = x;
    }

    function get_x(self): int {
        return self.x;
    }
}

function main(): void {
    p: Point = Point { x: 0, y: 0 };
    p.set_x(10);
    print(p.get_x());
}
```

### Arrays
```skunk
function main(): void {
    a: [3]int;
    b: [3]int = [3]int::fill(7);
    c: [3]int = [1, 2, 3];

    print(a[0]); // 0
    print(b[1]); // 7
    print(c[2]); // 3
}
```

### Pointers and Allocators
```skunk
struct Point {
    x: int;
    y: int;
}

function main(): void {
    heap: Allocator = System::allocator();
    arena: Arena = Arena::init(heap);
    alloc: Allocator = arena.allocator();

    p: *Point = Point::create(alloc);
    p.x = 4;
    p.y = 5;
    print(p.x + p.y);
    alloc.destroy(p);

    values: []int = []int::alloc(alloc, 3);
    values[1] = 7;
    print(values[1]);
    alloc.free(values);

    arena.deinit();
}
```

### Function Calls
```skunk
function add(a: int, b: int): int {
    return a + b;
}

function main(): void {
    result: int = add(5, 7);
    print(result);
}
```

### Nested Blocks
```skunk
function main(): void {
    x: int = 1;
    {
        x: int = 2;
        print(x); // Prints 2
    }
    print(x); // Prints 1
}
```

### Lambdas, Anonymous Functions, and Closures in Skunk

Skunk supports **lambdas**, **anonymous functions**, and **closures**, providing flexible and powerful tools for functional-style programming. These features allow you to define and manipulate functions dynamically, capturing variables from their enclosing scope.

---

#### **Lambdas and Anonymous Functions**

Lambdas in Skunk are inline, unnamed functions that can be assigned to variables or passed as arguments to other functions. They are defined using the `function` keyword.

#### Syntax:

```skunk
function(parameters): return_type {
    // function body
}
```

#### Example:

##### Simple Lambda:

```skunk
greet: (string) -> void = function(name: string): void {
    print("Hello, " + name);
}
greet("Alice");
```

##### Lambda as Argument:

```skunk
function execute(task: () -> void): void {
    task();
}

execute(function(): void {
    print("Task executed!");
});
```

---

#### **Closures**

Closures are functions that capture variables from their enclosing scope. This allows closures to "remember" the environment in which they were created.

#### Syntax:

Closures are defined like regular functions but can access variables from the scope in which they were created.

#### Example:

##### Counter Example:

```skunk
function createCounter(): () -> int {
    counter: int = 0;
    return function(): int {
        counter = counter + 1;
        return counter;
    }
}

increment: () -> int = createCounter();
print(increment()); // Output: 1
print(increment()); // Output: 2
```

#### Nested Closures:

Closures can also nest, allowing inner closures to capture variables from their parent closures.

##### Example:

```skunk
i: int = 0;

a: () -> () -> int = function(): () -> int {
    i = i + 1;
    j: int = 0;

    b: () -> () -> int = function(): () -> int {
        i = i + 1;
        j = j + 1;
        k: int = 0;

        c: () -> int = function(): int {
            i = i + 1;
            j = j + 1;
            k = k + 1;
            return i + j + k;
        }

        return c;
    }

    return b();
}

f: () -> int = a();
res: int = f();
print(res); // Output: 6 (3 + 2 + 1)
```

---

#### **Use Cases**

1. **Encapsulation**: Closures allow you to encapsulate state and behavior.
2. **Callbacks**: Lambdas can be used as callbacks for asynchronous operations.
3. **Functional Programming**: Enable higher-order functions like `map`, `filter`, and `reduce`.
4. **Recursion**: Support recursive functions with captured variables.

---

#### **Tips and Best Practices**

1. **Avoid Over-Capturing:** Be mindful of what variables a closure captures to avoid unintended dependencies.
2. **Mutability:** Remember that captured variables are shared, and modifying them in one closure affects others.
3. **Keep Closures Simple:** For readability and maintainability, try to keep closures concise.

---

Closures and lambdas are integral features of Skunk, empowering you to write expressive and modular code. By understanding how they work, you can leverage their full potential in your programs.




### Error Handling with Type Checker
```skunk
function main(): void {
    i: int = "wrong"; // Type error: Cannot assign string to int
}
```

## Installation and Usage

Skunk is still under development. For now, you can build and run Skunk programs locally:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/skunk
   cd skunk
   ```

2. Build Skunk:
   ```bash
   cargo build
   ```

3. Interpret a Skunk program:
   ```bash
   cargo run -- examples/hello.skunk
   ```

4. Compile a Skunk program to a native executable with LLVM/Clang:
   ```bash
   cargo run -- compile examples/hello.skunk ./hello
   ./hello
   ```

## LLVM Compilation Status

Skunk now includes an LLVM-based compiler path alongside the interpreter.

Currently supported in `cargo run -- compile ...`:

- Top-level function declarations
- `byte`, `short`, `int`, `long`, `float`, `double`, `boolean`, `char`, and string literals
- Local variables and assignments
- Arithmetic and comparisons on numeric primitives
- Boolean logic
- `if`, `for`, `return`
- Function calls, including chained call forms like `f()(1)`
- `print`
- Fixed arrays: zero initialization, `::fill(...)`, inline array literals, indexing, assignment, `.len`, and array pass/return by value
- Structs, field access, nested structs, methods, and method chaining that returns callable values
- Slices: `[]T`, slice literals, `a[lo:hi]`, omitted bounds, `.len`, indexing, and slice parameters
- Closures and lambdas, including captured locals, recursive lambdas, returned functions, and methods returning closures
- Pointers and allocation: `*T`, `System::allocator()`, `Arena::init(...)`, `arena.allocator()`, `T::create(alloc)`, `[]T::alloc(alloc, len)`, `alloc.destroy(ptr)`, and `alloc.free(slice)`

Current compiler/runtime trade-off:

- The native compiler currently uses a small C runtime for allocator-backed storage and for values that must outlive a stack frame.
- `Arena` now supports `reset()` and `deinit()`, and allocator-backed pointers/slices can be released with `alloc.destroy(...)` and `alloc.free(...)`.
- User-defined allocators, raw pointer operations, and an `unsafe` memory layer are still future work.

## Fibonacci Benchmark

To compare the current LLVM compiler path against Python, this repository includes:

- `examples/fibonacci_recursive.skunk`
- `examples/fibonacci_recursive.py`

Both programs recursively compute and sum the first `N = 35` Fibonacci values.

Build and run the Skunk version:

```bash
cargo build
target/debug/skunk compile examples/fibonacci_recursive.skunk /tmp/skunk_fibonacci_recursive
/tmp/skunk_fibonacci_recursive
```

Run the Python version:

```bash
python3 examples/fibonacci_recursive.py
```

Measured on this machine on March 28, 2026:

- Skunk compile time: `0.53347s`
- Skunk compiled binary median runtime: `0.066419s`
- Python median runtime: `1.703019s`
- Skunk speedup vs Python: `25.64x`

Both programs produced the same result:

```text
14930351
```

These numbers are only an example benchmark and will vary by machine, optimization settings, and workload.

## Language Development

Implementation direction and the current language-development contract live in [docs/language-development.md](/Users/dmgcodevil/dev/skunk-llvm/skunk/docs/language-development.md).

## Contributing

We welcome contributions to Skunk! If you have ideas, suggestions, or bug fixes, please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature-name`)
3. Commit your changes (`git commit -m 'Add feature'`)
4. Push to the branch (`git push origin feature-name`)
5. Create a pull request

## License

Skunk is open-source and distributed under the MIT License. See `LICENSE` for details.

---

Happy coding with Skunk! 🦨
