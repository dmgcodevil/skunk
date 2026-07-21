#!/bin/sh
# Skunk installer.
#
# Usage:
#   curl -fsSL https://dmgcodevil.github.io/skunk/install.sh | sh
#
# Installs the latest release to ~/.skunk/bin/skunk.
# Set SKUNK_VERSION=v0.1.0 to install a specific version.

set -eu

REPO="dmgcodevil/skunk"
INSTALL_DIR="${SKUNK_HOME:-$HOME/.skunk}/bin"

say() { printf '%s\n' "$*"; }
fail() { printf 'error: %s\n' "$*" >&2; exit 1; }

# --- Detect platform -------------------------------------------------------

os=$(uname -s)
arch=$(uname -m)

case "$os" in
  Darwin) os_part="apple-darwin" ;;
  Linux)  os_part="unknown-linux-musl" ;;
  *) fail "unsupported OS: $os (Skunk supports macOS and Linux)" ;;
esac

case "$arch" in
  arm64|aarch64) arch_part="aarch64" ;;
  x86_64|amd64)  arch_part="x86_64" ;;
  *) fail "unsupported architecture: $arch" ;;
esac

target="${arch_part}-${os_part}"
asset="skunk-${target}.tar.gz"

# --- Resolve download URL --------------------------------------------------

if [ -n "${SKUNK_VERSION:-}" ]; then
  base="https://github.com/$REPO/releases/download/$SKUNK_VERSION"
else
  base="https://github.com/$REPO/releases/latest/download"
fi

# --- Download and verify ---------------------------------------------------

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

say "Downloading $asset ..."
curl -fsSL "$base/$asset" -o "$tmp/$asset" \
  || fail "download failed: $base/$asset"

if command -v shasum >/dev/null 2>&1 || command -v sha256sum >/dev/null 2>&1; then
  say "Verifying checksum ..."
  curl -fsSL "$base/SHA256SUMS" -o "$tmp/SHA256SUMS" \
    || fail "download failed: $base/SHA256SUMS"
  expected=$(grep " $asset\$" "$tmp/SHA256SUMS" | awk '{print $1}')
  [ -n "$expected" ] || fail "no checksum entry for $asset"
  if command -v sha256sum >/dev/null 2>&1; then
    actual=$(sha256sum "$tmp/$asset" | awk '{print $1}')
  else
    actual=$(shasum -a 256 "$tmp/$asset" | awk '{print $1}')
  fi
  [ "$expected" = "$actual" ] || fail "checksum mismatch for $asset"
else
  say "warning: no sha256 tool found, skipping checksum verification"
fi

# --- Install ---------------------------------------------------------------

tar -xzf "$tmp/$asset" -C "$tmp"
mkdir -p "$INSTALL_DIR"

# Install as a versioned binary plus a `skunk` symlink so multiple versions
# can coexist; switch with `skunk use <version>`.
version=$("$tmp/skunk-$target/skunk" --version 2>/dev/null | awk '{print $2}')
if [ -n "$version" ]; then
  install -m 755 "$tmp/skunk-$target/skunk" "$INSTALL_DIR/skunk-$version"
  ln -sf "$INSTALL_DIR/skunk-$version" "$INSTALL_DIR/skunk"
  say "Installed skunk $version to $INSTALL_DIR/skunk-$version"
  say "Active version: $INSTALL_DIR/skunk -> skunk-$version"
else
  install -m 755 "$tmp/skunk-$target/skunk" "$INSTALL_DIR/skunk"
  say "Installed skunk to $INSTALL_DIR/skunk"
fi

# --- Prerequisite + PATH hints --------------------------------------------

if ! command -v clang >/dev/null 2>&1; then
  say ""
  say "note: clang was not found. Skunk needs clang to compile programs."
  if [ "$os" = "Darwin" ]; then
    say "      Install it with: xcode-select --install"
  else
    say "      Install it with your package manager, e.g.: sudo apt install clang"
  fi
fi

case ":$PATH:" in
  *":$INSTALL_DIR:"*) ;;
  *)
    say ""
    say "Add skunk to your PATH by appending this to your shell profile:"
    say ""
    say "  export PATH=\"$INSTALL_DIR:\$PATH\""
    ;;
esac

say ""
say "Done. Try: skunk --help"
