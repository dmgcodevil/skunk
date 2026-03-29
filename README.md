# Skunk

Skunk is a human-designed, AI-implemented experimental programming language targeting LLVM. It is a language-design project and compiler playground, not a production-ready platform.

Do not use Skunk to build critical, safety-sensitive, security-sensitive, or high-reliability software.

## Status

- Native compilation through LLVM/Clang is the primary execution path.
- The repository still contains legacy interpreter code while the compiler/runtime continues to absorb older coverage.
- The language reference lives in [Skunk](https://dmgcodevil.github.io/skunk/)
- Syntax and implemented behavior are defined by [`src/grammar.pest`](src/grammar.pest) and the test suite.

## Build

Skunk currently requires Rust and `clang`.

```bash
cargo build
```

## Use

Compile a program to a native executable:

```bash
cargo run -- compile path/to/main.skunk ./out
./out
```

The legacy interpreter path still exists:

```bash
cargo run -- path/to/main.skunk
```

## In This Repository

- [docs/index.html](docs/index.html): Skunk Language Reference
- [docs/pointers-and-allocators.md](docs/pointers-and-allocators.md): pointer and allocator design note
- [docs/language-development.md](docs/language-development.md): development contract
- [`examples/`](examples): runnable sample programs

## License

Skunk is open-source and distributed under the MIT License. See `LICENSE` for details.
