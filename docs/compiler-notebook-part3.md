# Skunk Compiler Notebook

## Part 3: Extending Skunk

Part 1 was about the map.

Part 2 was about following one tiny program through the compiler.

Part 3 is about making changes.

This guide is for the moment when you stop asking "How does this compiler work?" and start asking "How do I add a feature without getting lost?"

The most important idea in this part is simple:

Do not think of a feature as one code change.

Think of it as a path through the pipeline.

Some features touch only syntax and type checking. Some also touch monomorphization. Some need runtime support. The trick is learning how to recognize which stages need to know about the feature.

## Chapter 1: Start Small On Purpose

If you are new to compiler work, the best first features are not the most exciting ones. They are the ones that teach the pipeline.

Good beginner features:

- syntax sugar
- a small new built-in function
- a new type rule
- a new static attached function form
- a small extension to an existing expression form

Bad beginner features:

- full multithreading
- a brand-new ownership model
- a major IR rewrite
- a deep runtime model change that affects everything

Why?

Because beginner features should help you learn the compiler's architecture, not force you to redesign it before you understand it.

### Read next

- [`src/main.rs`](../src/main.rs)
- [`src/grammar.pest`](../src/grammar.pest)
- [`src/ast.rs`](../src/ast.rs)

## Chapter 2: Think In Terms Of A Feature Path

When adding a feature, ask this question first:

"Where does this feature first appear, and which later stages need to know about it?"

That one question is better than diving into random files.

For example:

- If you add syntax sugar for existing behavior, you may only need grammar, AST, and tests.
- If you add a new language rule, you probably need grammar, AST, type checking, and tests.
- If you add a new runtime capability, you may also need compiler lowering and runtime support files.
- If you add a generic feature, monomorphization may need to understand it too.

This is the core workflow of extending a compiler:

```text
feature idea
  -> where does it appear in source?
  -> how is it represented in AST?
  -> what semantic rules validate it?
  -> how is it lowered or executed?
  -> what tests prove it works?
```

### Read next

- [`src/ast.rs`](../src/ast.rs): `Node`, `Type`
- [`src/type_checker.rs`](../src/type_checker.rs): `check`, `resolve_type`
- [`src/compiler.rs`](../src/compiler.rs): `compile_to_llvm_ir`

## Chapter 3: Stage 1, Syntax

If the feature changes what users can write, the first stop is usually [`src/grammar.pest`](../src/grammar.pest).

This is where you decide:

- what the syntax looks like
- how ambiguous it might be
- whether it fits existing patterns

Then the parser in [`src/ast.rs`](../src/ast.rs) needs to turn that syntax into compiler nodes.

Important parser functions to know:

- `PestImpl::parse`
- `create_ast`
- `create_primary`
- `create_access`
- `create_struct_init`

The parser's job is not to decide whether something is semantically valid. Its job is to build a useful tree.

That means you should avoid putting deep semantic decisions in the parser when they belong later in the type checker.

### Practical advice

If the new feature can be desugared into an existing AST form early, do that whenever it keeps later stages simpler.

### Read next

- [`src/grammar.pest`](../src/grammar.pest)
- [`src/ast.rs`](../src/ast.rs): `create_ast`, `create_primary`, `create_access`, `create_struct_init`

## Chapter 4: Stage 2, AST Design

Sometimes the parser can reuse existing AST forms.

Sometimes a feature needs a new node or a richer field in an existing node.

When deciding whether to add a new AST variant, ask:

- Is this actually a new semantic kind of thing?
- Or is it just alternate syntax for something the compiler already understands?

Good reasons to add a new AST variant:

- the feature has its own meaning
- later stages need to distinguish it clearly
- forcing it into an unrelated node would make code harder to read

Good reasons not to add a new AST variant:

- it is only syntax sugar
- it can be translated cleanly into something already supported

This is one of the most important judgment calls in compiler engineering. Too many AST variants can make the tree noisy. Too few can make later stages cryptic.

### Read next

- [`src/ast.rs`](../src/ast.rs): `Node`
- [`src/ast.rs`](../src/ast.rs): `Type`

## Chapter 5: Stage 3, Source Loading And Normalization

Many features do not need changes here.

But if the feature affects modules, imports, exports, name visibility, or top-level declaration shapes, you may need to touch [`src/source.rs`](../src/source.rs).

Common cases:

- new top-level declaration forms
- features that interact with exports
- features that create names which may need private-name normalization

If the feature is local expression syntax, you probably do not need this stage.

That is a useful reminder:

Not every feature touches every phase.

### Read next

- [`src/source.rs`](../src/source.rs): `load_program`
- [`src/source.rs`](../src/source.rs): `ModuleNormalizer::normalize`

## Chapter 6: Stage 4, Monomorphization

If the feature interacts with generics, you should expect to read [`src/monomorphize.rs`](../src/monomorphize.rs).

Questions to ask:

- Does this feature behave differently for generic types?
- Does it introduce new generic constraints?
- Does it require specialized concrete declarations?
- Does it change how names are specialized?

Examples of features that probably touch monomorphization:

- generic syntax changes
- new bound forms
- generic attached functions
- generic trait or shape relationships

Examples of features that probably do not:

- a new local statement form with no generic impact
- a tiny syntax sugar feature for struct literals

### Read next

- [`src/monomorphize.rs`](../src/monomorphize.rs): `prepare_program`
- [`src/monomorphize.rs`](../src/monomorphize.rs): `Monomorphizer::prepare`
- [`src/monomorphize.rs`](../src/monomorphize.rs): `apply_substitutions`

## Chapter 7: Stage 5, Type Checking

This is where most features become real.

The type checker decides whether the new form is legal and what type it produces.

That usually means touching one or more of:

- `check`
- `resolve_type`
- `resolve_access`
- `is_assignable`
- global scope construction

This stage answers the important semantic questions:

- Is the feature allowed here?
- What type does it produce?
- What operations are permitted on it?
- How does it interact with const, references, traits, shapes, or unsafe code?

### Practical advice

When adding a new feature, write down the valid cases and invalid cases before editing `resolve_type`. That file gets much easier to work with when you know exactly what legal and illegal behavior you want.

### Read next

- [`src/type_checker.rs`](../src/type_checker.rs): `check`
- [`src/type_checker.rs`](../src/type_checker.rs): `resolve_type`, `resolve_access`, `is_assignable`

## Chapter 8: Stage 6, Code Generation

Once the type checker understands the feature, the backend may need to lower it into LLVM IR.

For backend work, the most important questions are:

- Does the feature introduce a new runtime representation?
- Does it only need different instruction selection?
- Does it require a new layout?
- Does it need a coercion rule?

Important backend areas:

- `LlvmType`
- layout collection
- `compile_statement`
- `compile_expr_with_expected`
- `compile_struct_literal`
- `coerce_expr`

This is where language meaning becomes storage and instructions.

### Read next

- [`src/compiler.rs`](../src/compiler.rs): `LlvmType`, `llvm_type`
- [`src/compiler.rs`](../src/compiler.rs): `collect_struct_layouts`, `collect_enum_layouts`, `collect_trait_layouts`
- [`src/compiler.rs`](../src/compiler.rs): `compile_statement`, `compile_expr_with_expected`, `compile_struct_literal`, `coerce_expr`

## Chapter 9: Stage 7, Runtime Support

Some features cannot live purely inside the compiler.

If the compiled program needs help from native support code, you may also need to touch the runtime files:

- [`runtime/skunk_runtime.c`](../runtime/skunk_runtime.c)
- [`runtime/skunk_window_runtime.m`](../runtime/skunk_window_runtime.m)

Examples:

- memory helpers
- console or file IO
- windowing
- keyboard input
- platform APIs

When this happens, remember that the feature path is no longer just "parser to backend." It now includes the runtime boundary too.

### Read next

- [`src/compiler.rs`](../src/compiler.rs): `compile_to_executable`
- [`runtime/skunk_runtime.c`](../runtime/skunk_runtime.c)
- [`runtime/skunk_window_runtime.m`](../runtime/skunk_window_runtime.m)

## Chapter 10: Tests Are The Feature Contract

One of the best things about this repository is that tests already act as part of the language contract.

When you add a feature, think in terms of at least three kinds of tests:

1. Parser tests
2. Type-checking tests
3. Runtime or compiler tests

Where those usually live:

- parser-oriented tests in [`src/ast.rs`](../src/ast.rs)
- type-checking tests in [`src/type_checker.rs`](../src/type_checker.rs)
- codegen/runtime tests in [`src/compiler.rs`](../src/compiler.rs)
- source-loading tests in [`src/source.rs`](../src/source.rs)
- monomorphization tests in [`src/monomorphize.rs`](../src/monomorphize.rs)

This is one of the best habits you can build:

Before you trust a feature, ask what proves it.

### Read next

- [`src/ast.rs`](../src/ast.rs)
- [`src/type_checker.rs`](../src/type_checker.rs)
- [`src/compiler.rs`](../src/compiler.rs)

## Chapter 11: Docs And Examples Are Part Of The Feature

In a language project, docs are not decorative.

They are part of how the language becomes usable.

When you add a feature, you should usually update:

- the language reference in [`docs/index.html`](./index.html)
- examples in [`examples/`](../examples)
- any design note if the feature has deeper implications

This matters because it forces you to explain the feature, not just implement it.

If a feature is hard to explain simply, that is often useful feedback about the design itself.

### Read next

- [`docs/index.html`](./index.html)
- [`examples/`](../examples)

## Chapter 12: A Practical Implementation Checklist

When you start a new feature, use this checklist.

1. Write the smallest example program that demonstrates the feature.
2. Decide whether the feature is syntax sugar or needs a new semantic node.
3. Update grammar and parser if the source syntax changes.
4. Update AST structures only if needed.
5. Update source loading only if the feature affects module-level behavior.
6. Update monomorphization if generics are involved.
7. Update type checking so valid and invalid cases are explicit.
8. Update backend lowering if the feature affects runtime behavior.
9. Update runtime code if native support is required.
10. Add parser, type-checker, and compiler/runtime tests.
11. Update docs and examples.

That is the real "end-to-end" path for a feature in this codebase.

## Chapter 13: Worked Example, Adding A Small Feature

Let’s say you want to add a tiny syntax sugar feature like struct literal field shorthand:

```skunk
Point { x, y }
```

meaning:

```skunk
Point { x: x, y: y }
```

That feature path would look like this:

1. Grammar
   - allow struct literal fields to be either `name: expr` or `name`
2. AST construction
   - either add a shorthand form or desugar immediately into `name: Identifier(name)`
3. Type checking
   - usually no major new rule if desugared early
4. Backend
   - likely no change if the AST already becomes a normal struct literal
5. Tests
   - parser test
   - maybe a compile/runtime test
6. Docs
   - add one short example

This is a great example because it teaches a deep lesson:

The earlier you can reduce a feature into existing forms, the less work later stages need.

## Chapter 14: Worked Example, Adding A Runtime Feature

Now compare that with a runtime feature like:

```skunk
Keyboard::is_down(window, 'w')
```

That path is broader:

1. Type checker must recognize the static call and its types.
2. Backend must lower the call to a runtime function.
3. Runtime native code must implement the behavior.
4. Tests must validate both typing and compiled execution.
5. Docs and examples must show how to use it.

This comparison is helpful because it teaches you to classify features.

Not every feature is equally "language-level" versus "runtime-level."

## Chapter 15: How To Avoid Common Beginner Mistakes

Here are the mistakes that most often waste time:

- changing too many stages before proving the syntax or semantics
- adding a new AST node when desugaring would have been enough
- changing the backend before the type checker knows what the feature means
- forgetting tests for invalid cases
- treating docs as optional
- trying to solve a huge language problem as a first contribution

A good rule is:

Make the feature visible in the earliest stage that needs to know about it, and no earlier.

## Chapter 16: A Final Mental Model

If you only remember one thing from this part, remember this:

Adding a feature is not one edit.

It is a small conversation between compiler stages.

The parser says, "I can recognize this."

The AST says, "I can represent this."

The type checker says, "I know what this means."

The backend says, "I know how to lower this."

The runtime says, "I can support this at execution time."

The tests say, "We can prove it works."

That is how Skunk grows.

## Next Step

If you want the printable version of these notes, open:

- [`docs/compiler-booklet.html`](./compiler-booklet.html)

And if you want the broader context again, revisit:

- [`docs/compiler-notebook.md`](./compiler-notebook.md)
- [`docs/compiler-notebook-part2.md`](./compiler-notebook-part2.md)
