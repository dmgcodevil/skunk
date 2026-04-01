#!/bin/sh
set -eu

ROOT="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
OUT_DIR="$ROOT/dist"
VERSION="$(python3 - <<'PY'
import json, pathlib
pkg = json.loads((pathlib.Path("editors/vscode/skunk/package.json")).read_text())
print(pkg["version"])
PY
)"
NAME="dmgcodevil.skunk-${VERSION}.vsix"

mkdir -p "$OUT_DIR"
rm -f "$OUT_DIR/$NAME"
rm -rf "$OUT_DIR/extension"
mkdir -p "$OUT_DIR/extension"

cp "$ROOT/package.json" "$OUT_DIR/extension/package.json"
cp "$ROOT/extension.js" "$OUT_DIR/extension/extension.js"
cp "$ROOT/language-configuration.json" "$OUT_DIR/extension/language-configuration.json"
cp "$ROOT/README.md" "$OUT_DIR/extension/README.md"
cp -R "$ROOT/syntaxes" "$OUT_DIR/extension/syntaxes"
cp -R "$ROOT/snippets" "$OUT_DIR/extension/snippets"
cp "$ROOT/[Content_Types].xml" "$OUT_DIR/[Content_Types].xml"
cp "$ROOT/extension.vsixmanifest" "$OUT_DIR/extension.vsixmanifest"

(
    cd "$OUT_DIR"
    zip -qr "$NAME" "[Content_Types].xml" extension.vsixmanifest extension
)

rm -rf "$OUT_DIR/extension"
printf '%s\n' "$OUT_DIR/$NAME"
