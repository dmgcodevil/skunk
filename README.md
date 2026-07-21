# Skunk

Skunk is a human-designed, AI-implemented experimental programming language targeting LLVM. It is a language-design project and compiler playground, not a production-ready platform.

Do not use Skunk to build critical, safety-sensitive, security-sensitive, or high-reliability software.

## Status

- Native compilation through LLVM/Clang is the primary execution path.
- Native compilation is the only execution path, keeping runtime behavior aligned with generated binaries.
- The language reference lives in [Skunk](https://dmgcodevil.github.io/skunk/)
- Syntax and implemented behavior are defined by [`src/grammar.pest`](src/grammar.pest) and the test suite.

## Install

Skunk needs `clang` at runtime to link native executables (macOS: `xcode-select --install`, Linux: install `clang` via your package manager).

Install the latest release (macOS and Linux):

```bash
curl -fsSL https://dmgcodevil.github.io/skunk/install.sh | sh
```

This downloads the prebuilt binary for your platform from [GitHub Releases](https://github.com/dmgcodevil/skunk/releases), verifies its checksum, and installs it to `~/.skunk/bin`. Pin a version with `SKUNK_VERSION=v0.1.0`. Installs are versioned side by side: `skunk --version` shows the running version, `skunk versions` lists installed ones, and `skunk use <version>` switches. See [RELEASE.md](RELEASE.md) for building and installing a release locally.

Alternatively, download a tarball from the [releases page](https://github.com/dmgcodevil/skunk/releases) manually, or build from source with Rust:

```bash
cargo install --git https://github.com/dmgcodevil/skunk
```

The release process is documented in [RELEASE.md](RELEASE.md).

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

Compile and run a program natively using a temporary executable:

```bash
cargo run -- path/to/main.skunk
# Equivalent explicit form:
cargo run -- run path/to/main.skunk
```

## Projects and Tests

Scaffold a project with a `skunk.toml` manifest, build it, and run its native tests:

```bash
skunk new demo
cd demo
skunk build        # compiles src/main.skunk into target/demo
skunk test         # runs `test "name" { ... }` blocks natively
skunk test --filter shorthand
```

`skunk.toml` configures linking for `extern "C"` interop (`libraries = ["sqlite3"]`, `frameworks = ["Cocoa"]`). The standard library ships embedded in the compiler; `import std.math;` resolves to the SDK (the `std.` prefix is reserved) and provides libm bindings plus integer helpers.

See the [language reference](https://dmgcodevil.github.io/skunk/) for the C interop, standard library, and native test sections, and `examples/c_interop.skunk`, `examples/math_test.skunk`, and `examples/calculator/` for runnable samples.

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
- [docs/compiler-booklet.html](docs/compiler-booklet.html): print-friendly compiler booklet with diagrams and a worked example
- [docs/compiler-notebook.md](docs/compiler-notebook.md): compiler notebook, Part 1
- [docs/compiler-notebook-part2.md](docs/compiler-notebook-part2.md): compiler notebook, Part 2
- [docs/compiler-notebook-part3.md](docs/compiler-notebook-part3.md): compiler notebook, Part 3, focused on extending the language/compiler
- [docs/pointers-and-allocators.md](docs/pointers-and-allocators.md): pointer and allocator design note
- [docs/language-development.md](docs/language-development.md): development contract
- [`examples/`](examples): runnable sample programs
- [`editors/vscode/skunk`](editors/vscode/skunk): VS Code syntax highlighting, completion, and formatter extension

## License

Skunk is open-source and distributed under the MIT License. See `LICENSE` for details.
