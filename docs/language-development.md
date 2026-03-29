# Skunk Language Development Contract

This document captures the current working agreement for Skunk development.

## Source Of Truth

Skunk language behavior should be derived from these sources:

1. [`src/grammar.pest`](/Users/dmgcodevil/dev/skunk-llvm/skunk/src/grammar.pest) for syntax
2. Parser, interpreter, and type-checker tests in [`src/ast.rs`](/Users/dmgcodevil/dev/skunk-llvm/skunk/src/ast.rs), [`src/interpreter.rs`](/Users/dmgcodevil/dev/skunk-llvm/skunk/src/interpreter.rs), and [`src/type_checker.rs`](/Users/dmgcodevil/dev/skunk-llvm/skunk/src/type_checker.rs) for executable behavior
3. [`README.md`](/Users/dmgcodevil/dev/skunk-llvm/skunk/README.md) for language-facing examples and intent
4. Focused design notes in [`docs/`](/Users/dmgcodevil/dev/skunk-llvm/skunk/docs) for agreed future language directions such as pointers and allocators

If these disagree:

- Grammar and tests win over prose
- README should be updated to match implemented behavior
- New language decisions should add or update tests

## Roles

- The user is the language designer
- The implementation agent owns development, refactoring, testing, and architecture decisions

The preferred input format for new features is:

- syntax
- semantics
- 1-2 valid example programs
- expected invalid cases or errors

## Repository Direction

We are keeping this repository and evolving it in place rather than restarting from scratch.

Why:

- The repo already has a parser, AST, type checker, interpreter, and initial LLVM compiler path
- The language is still evolving, so preserving iteration speed is more valuable than a clean-slate rewrite
- Existing tests provide a growing language contract

Aggressive refactoring is allowed when it improves the long-term language implementation.

## Implementation Strategy

Skunk should move toward a clearer compiler pipeline over time:

1. Parse source into syntax-level AST
2. Perform semantic analysis and type checking
3. Lower into a simpler typed IR
4. Execute either through an interpreter/runtime path or LLVM code generation
5. Keep tests covering syntax, typing, and runtime behavior at each layer

The current interpreter remains valuable as a reference implementation while the compiler backend grows.

## Near-Term Engineering Priorities

- Separate language front-end concerns from runtime concerns
- Reduce coupling between parsing, type checking, evaluation, and codegen
- Continue expanding LLVM support feature by feature
- Prefer adding targeted tests before or alongside new language features
- Keep unsupported compiler features failing clearly rather than silently degrading behavior
