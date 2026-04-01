# Skunk VS Code Extension

This extension adds lightweight editor support for Skunk:

- Syntax highlighting for `.skunk` files
- Basic autocompletion for keywords, builtins, snippets, and top-level symbols in the current file
- A simple formatter for brace-based indentation

## What It Supports Today

- `struct`, `trait`, `shape`, `attach`, and `conform` highlighting
- Builtin types such as `int`, `float`, `Allocator`, and `Arena`
- Snippets for common Skunk declarations
- Format Document support with a simple line-based formatter

## Limitations

- Completions are intentionally basic today. They are not semantic and do not yet use the compiler for type-aware suggestions.
- Formatting is heuristic. It handles indentation well, but it is not yet a full AST-aware formatter.

## Install In VS Code

### Option 1: Install From VSIX

Build the VSIX:

```bash
sh editors/vscode/skunk/build-vsix.sh
```

Then in VS Code:

1. Open Extensions
2. Click the `...` menu
3. Choose `Install from VSIX...`
4. Select `editors/vscode/skunk/dist/dmgcodevil.skunk-0.0.3.vsix`

This is the most reliable installation method.

### Option 2: Install As An Unpacked Local Extension

On macOS or Linux:

```bash
mkdir -p ~/.vscode/extensions/dmgcodevil.skunk-0.0.3
cp -R editors/vscode/skunk/. ~/.vscode/extensions/dmgcodevil.skunk-0.0.3/
```

Then fully restart VS Code.

### Option 3: Run It In An Extension Development Host

1. Open `/Users/dmgcodevil/dev/skunk-llvm/skunk/editors/vscode/skunk` in VS Code.
2. Press `F5`.
3. A new Extension Development Host window will open with the Skunk extension loaded.

## Recommended VS Code Settings

```json
{
  "[skunk]": {
    "editor.defaultFormatter": "dmgcodevil.skunk",
    "editor.formatOnSave": true
  }
}
```

## Future Direction

The next natural step would be an LSP-backed extension that reuses the Skunk parser/compiler for:

- semantic completions
- go to definition
- hover
- inline diagnostics
