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

A new macOS-first window/input runtime is also available for simple 2D programs. The repository includes a playable Pong example:

```bash
cargo run -- compile examples/pong.skunk ./pong
./pong
```

The legacy interpreter path still exists:

```bash
cargo run -- path/to/main.skunk
```

## VS Code

A lightweight VS Code extension now lives in [`editors/vscode/skunk`](editors/vscode/skunk). It includes:

- syntax highlighting for `.skunk`
- basic autocompletion for keywords, snippets, and top-level symbols
- a simple formatter for indentation and brace layout

Quick local install on macOS or Linux:

```bash
sh editors/vscode/skunk/build-vsix.sh
```

Then install `editors/vscode/skunk/dist/dmgcodevil.skunk-0.0.3.vsix` from VS Code via `Extensions > ... > Install from VSIX...`. More details are in [`editors/vscode/skunk/README.md`](editors/vscode/skunk/README.md).

## In This Repository

- [docs/index.html](docs/index.html): Skunk Language Reference
- [docs/pointers-and-allocators.md](docs/pointers-and-allocators.md): pointer and allocator design note
- [docs/language-development.md](docs/language-development.md): development contract
- [`examples/`](examples): runnable sample programs
- [`editors/vscode/skunk`](editors/vscode/skunk): VS Code syntax highlighting, completion, and formatter extension

## License

Skunk is open-source and distributed under the MIT License. See `LICENSE` for details.
