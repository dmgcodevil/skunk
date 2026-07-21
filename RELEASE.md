# Release Process

This document describes how Skunk is packaged, released, and installed.

## Overview

Skunk ships as a single self-contained compiler binary per platform. Each release
produces one tarball per target, published on GitHub Releases:

```
skunk-aarch64-apple-darwin.tar.gz        # macOS Apple Silicon
skunk-x86_64-apple-darwin.tar.gz         # macOS Intel
skunk-x86_64-unknown-linux-musl.tar.gz   # Linux x86_64 (static)
skunk-aarch64-unknown-linux-musl.tar.gz  # Linux ARM64 (static)
SHA256SUMS
```

Each tarball contains the `skunk` binary and `README.md`. Linux builds use musl
so the binary is fully static and runs on any distro regardless of glibc version.
Asset names are deterministic (no version in the filename) so that
`releases/latest/download/<asset>` URLs work; the release itself is versioned by
its git tag.

Users need `clang` installed to compile Skunk programs (Xcode Command Line Tools
on macOS, `clang` package on Linux). This is a documented runtime prerequisite
and is not bundled.

## Self-contained binaries

The runtime C sources (`runtime/skunk_runtime.c`,
`runtime/skunk_window_runtime.m`) are embedded into the compiler binary with
`include_str!` and materialized on demand into
`$SKUNK_HOME/runtime/<version>/` (default `~/.skunk`). A released binary
therefore works on any machine with `clang` installed — no source checkout
required. The embedded standard library (`lib/std/`, e.g. `std.math`) uses the
same mechanism and materializes into `$SKUNK_HOME/lib/<version>/`.

## Building and installing a release locally

To build and install a release from a source checkout, without CI or GitHub
Releases:

```bash
cargo test
cargo build --release
```

Install the binary versioned, with a `skunk` symlink selecting the active
version:

```bash
VERSION=$(./target/release/skunk --version | awk '{print $2}')
mkdir -p ~/.skunk/bin
install -m 755 target/release/skunk ~/.skunk/bin/skunk-$VERSION
ln -sf ~/.skunk/bin/skunk-$VERSION ~/.skunk/bin/skunk
```

Add `~/.skunk/bin` to `PATH` (once, in your shell profile):

```bash
export PATH="$HOME/.skunk/bin:$PATH"
```

Verify:

```bash
skunk --version
skunk run examples/c_interop.skunk
```

Alternatively `cargo install --path .` installs a single unversioned binary
into `~/.cargo/bin`; that is fine for day-to-day development but bypasses the
version management below.

## Managing installed versions

Everything version-specific lives under `$SKUNK_HOME` (default `~/.skunk`) in
version-keyed directories, so multiple versions coexist without interfering:

```
~/.skunk/
  bin/
    skunk -> skunk-0.2.0     # symlink selecting the active version
    skunk-0.1.0
    skunk-0.2.0
  runtime/<version>/          # materialized C runtime, keyed per version
  lib/<version>/std/          # materialized standard library, keyed per version
```

- `skunk --version` (also `-V`, `version`) prints the version of the binary
  being run.
- `skunk versions` lists the versioned binaries installed in
  `~/.skunk/bin` and marks the one the `skunk` symlink points at.
- `skunk use <version>` repoints the symlink, e.g. `skunk use 0.1.0`.
- Uninstall a version by deleting its binary (and optionally its
  `runtime/<version>` and `lib/<version>` directories).

The install script installs releases in this same versioned layout, so local
builds and downloaded releases can be switched between freely. Because the
runtime and standard library are embedded in each binary and materialized
under version-keyed paths, switching versions never mixes SDK files.

## Cutting a release

1. Make sure `main` is green (the `Rust` CI workflow passes).
2. Bump `version` in `Cargo.toml`. Commit.
3. Tag and push:

   ```bash
   git tag v0.1.0
   git push origin main v0.1.0
   ```

4. The `Release` workflow (`.github/workflows/release.yml`) triggers on the tag.
   It runs the test suite, builds all four targets, packages tarballs, generates
   `SHA256SUMS`, and creates a GitHub Release with the artifacts attached.
5. Verify the release: download a tarball for your platform, extract, and run
   `./skunk run examples/fibonacci_recursive.skunk`.

Nothing is built or packaged on a developer machine; all release artifacts come
from CI.

### Build matrix

| Runner           | Target                        |
| ---------------- | ----------------------------- |
| `macos-latest`   | `aarch64-apple-darwin`        |
| `macos-latest`   | `x86_64-apple-darwin`         |
| `ubuntu-latest`  | `x86_64-unknown-linux-musl`   |
| `ubuntu-24.04-arm` | `aarch64-unknown-linux-musl` |

macOS Intel is cross-compiled from the Apple Silicon runner via
`rustup target add x86_64-apple-darwin` (same OS, different arch — no extra
toolchain needed). Linux ARM64 builds natively on GitHub's ARM runners, so no
cross toolchain is required anywhere.

## Installation channels

- **install script** (primary):
  `curl -fsSL https://dmgcodevil.github.io/skunk/install.sh | sh`
  The script lives at `docs/install.sh` and is served by GitHub Pages. It
  detects OS/arch, downloads the matching tarball from the latest GitHub
  Release, verifies the SHA-256 checksum, and installs to `~/.skunk/bin`.
  A specific version can be requested with `SKUNK_VERSION=v0.1.0`.
- **manual**: download a tarball from
  [GitHub Releases](https://github.com/dmgcodevil/skunk/releases), extract, and
  put `skunk` on `PATH`.
- **from source**: `cargo install --git https://github.com/dmgcodevil/skunk`
  (requires Rust).
- **Homebrew** (planned): a tap repo with a formula pointing at the release
  tarballs, bumped automatically by the release workflow.

## Code signing (macOS)

Release binaries are not signed or notarized. Gatekeeper quarantine applies to
browser-downloaded files, not to files fetched with `curl` or installed via
Homebrew, so the install script path works without signing. Signing and
notarization (Apple Developer ID) become necessary only if Skunk is ever
distributed as a browser-download `.dmg`/`.pkg`.

## Versioning

Releases follow `v<MAJOR>.<MINOR>.<PATCH>`. The compiler and the embedded
SDK/standard library are one artifact and share one version (Go-style): a
compiler/stdlib mismatch cannot happen. Pre-1.0, minor versions may break
language compatibility; this is documented in the release notes.
