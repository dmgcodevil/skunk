# Skunk Programming Language

**Skunk** is a statically typed, interpreted programming language designed for simplicity, learning, and extensibility. It provides a clean syntax for working with structured data, control flow, and functions while supporting extensible features like user-defined types and type inference.

## Features

- **Basic Types**: `int`, `string`, `boolean`
- **User-Defined Structs**: Define custom types with fields and methods
- **Control Flow**: `if`, `for` loops, and blocks for scoped variable overrides
- **Arrays**: Support for array initialization, slicing (upcoming), and dynamic resizing
- **Functions**: First-class functions with support for closures and higher-order programming (upcoming)
- **Type Checking**: Ensures type correctness at parse-time with detailed error messages
- **Type Inference**: Planned for a cleaner developer experience
- **String Interpolation and Concatenation**: Upcoming for intuitive string operations
- **Generics**: Upcoming for flexible and reusable data structures

## Example Programs

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

    function set_x(x: int): void {
        self.x = x;
    }

    function get_x(): int {
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
    arr: int[3] = [1, 2, 3];
    for (i: int = 0; i < 3; i = i + 1) {
        print(arr[i]);
    }
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

2. Build the interpreter:
   ```bash
   cargo build
   ```

3. Run a Skunk program:
   ```bash
   cargo run -- examples/hello.skunk
   ```

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

