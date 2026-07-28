# Skunk Compiler Notebook

## Part 1: A Gentle Tour Of The Pipeline

This notebook is for someone who is new to compilers, new to LLVM, and wants to understand this specific codebase without getting buried under jargon.

It is not a formal spec. It is not trying to prove compiler theory. It is a reading guide for the Skunk repository.

The central idea to keep in mind is this:

Skunk is a pipeline. Each stage takes the program in one form and turns it into a slightly more useful form for the next stage.

If you remember where you are in the pipeline, the code becomes much easier to follow.

## Chapter 1: The Story Of The Whole Compiler

At a high level, the compiler answers one question:

"How do we turn a `.skunk` program into something the machine can run?"

Skunk reads the program, checks it, lowers it into LLVM IR, and asks `clang` to produce a native executable. The same native pipeline powers both `skunk run` and `skunk compile`.

That means Skunk is not just a parser and not just a code generator. It is a whole pipeline made of loading, parsing, normalization, monomorphization, type checking, lowering, and runtime linkage.

Here is the shortest useful mental model:

```text
source files
  -> one source-oriented syntax module
  -> normalized / specialized syntax
  -> resolved and typed HIR
  -> LLVM IR
  -> native binary
```

### Read next

- [`src/main.rs`](../src/main.rs): `parse_cli`, `default_output_path`, `main`
- [`src/syntax/loader.rs`](../src/syntax/loader.rs): `load_program`
- [`src/pipeline.rs`](../src/pipeline.rs): `check_source`, `check_loaded`
- [`src/hir/lower.rs`](../src/hir/lower.rs): `lower`
- [`src/backend/mod.rs`](../src/backend/mod.rs): `compile_to_executable`, `compile_to_llvm_ir`

## Chapter 2: `main.rs` Is The Pipeline Coordinator

The easiest place to understand the project shape is [`src/main.rs`](../src/main.rs).

That file does not contain the language logic itself. Instead, it coordinates the major stages.

The important thing to notice is the order:

1. Parse the CLI
2. Load the program
3. Ask `pipeline` to normalize, specialize, resolve, type, and validate it
4. Give the resulting `CheckedProgram` to the backend
5. Either run the executable or keep it

That order tells you what the rest of the repository expects. For example, the compiler backend assumes it receives a program that has already been loaded, normalized, and checked.

This is a useful compiler lesson:

Big functions at the top of a compiler often tell you more about architecture than any design document does.

### Read next

- [`src/main.rs`](../src/main.rs): `main`
- [`src/syntax/loader.rs`](../src/syntax/loader.rs): `load_program`
- [`src/pipeline.rs`](../src/pipeline.rs): `check_loaded`
- [`src/backend/mod.rs`](../src/backend/mod.rs): `compile_to_executable`

## Chapter 3: Parsing Means "Turn Text Into Structure"

Before the compiler can reason about a program, it needs to stop seeing the program as raw text.

That happens in two layers.

The first layer is the grammar in [`src/syntax/grammar.pest`](../src/syntax/grammar.pest). This file describes what valid Skunk source looks like.

The second layer is direct AST construction in [`src/syntax/parser.rs`](../src/syntax/parser.rs). Pest's generated `Rule` and `Pair` values stay inside this module; successful parsing produces the source-level structures declared in [`src/syntax/ast.rs`](../src/syntax/ast.rs).

The main idea is simple:

- the grammar recognizes source forms
- the syntax AST gives those forms names, stable node identities, and source spans

For example, the compiler does not want to keep asking, "is this sequence of characters a struct initialization?" It wants an `ExprKind::StructInit` expression.

That is why ASTs matter. They replace text with structure.

### Read next

- [`src/syntax/grammar.pest`](../src/syntax/grammar.pest)
- [`src/syntax/parser.rs`](../src/syntax/parser.rs): `parse_module`, `DirectParser::expression`, `DirectParser::primary`, `DirectParser::access`, `DirectParser::struct_init`
- [`src/syntax/ast.rs`](../src/syntax/ast.rs): `Module`, `TopLevelKind`, `StmtKind`, `ExprKind`, `TypeKind`

## Chapter 4: AST And HIR Have Different Jobs

The syntax AST records what the programmer wrote. It contains names and spans, but no inferred semantic types or resolved declaration identities.

Typed HIR in [`src/hir/mod.rs`](../src/hir/mod.rs) records what the program means after resolution. It uses `DefId`, `LocalId`, `FieldId`, `VariantId`, and interned `TypeId` values instead of unresolved source names.

Keeping the representations separate prevents backend details from leaking into syntax and prevents parser assumptions from becoming semantic invariants. The stable handoff is `pipeline::CheckedProgram`; the LLVM backend does not accept raw syntax nodes.

This makes a feature slightly more explicit to implement: syntax, resolution/type analysis, HIR lowering, and code generation each handle only the information appropriate to that phase.

That separation is intentional compiler architecture, not duplicated models of the same stage.

### Read next

- [`src/syntax/ast.rs`](../src/syntax/ast.rs): source-level declarations, statements, expressions, and types
- [`src/analysis/resolver.rs`](../src/analysis/resolver.rs): resolved identities
- [`src/analysis/types.rs`](../src/analysis/types.rs): `TypeStore`, `TypeKind`
- [`src/hir/mod.rs`](../src/hir/mod.rs): typed high-level IR

## Chapter 5: Loading Modules Is A Real Compiler Pass

New compiler readers sometimes think the "real" compiler starts only after parsing. In practice, loading source files is already compiler work.

That logic lives in [`src/syntax/loader.rs`](../src/syntax/loader.rs).

`load_program` does more than read a single file. It:

- resolves the entry path
- loads imports recursively
- detects cyclic imports
- checks module names
- normalizes visibility and private names
- returns one merged syntax `Module` plus its `SourceMap`

This is a very important simplification for later stages. Instead of every later pass needing to think about a graph of files, most of the pipeline gets to think about one program tree.

The other important piece in this file is `ModuleRenamer`. It rewrites private names from imported modules so they do not collide later.

That means `source.rs` is where "many source files" becomes "one safe program to analyze."

### Read next

- [`src/syntax/loader.rs`](../src/syntax/loader.rs): `load_program`
- [`src/syntax/loader.rs`](../src/syntax/loader.rs): `ProgramLoader::load_file`, `ProgramLoader::module_path`
- [`src/syntax/loader.rs`](../src/syntax/loader.rs): `ModuleRenamer::new`, `ModuleRenamer::rename`

## Chapter 6: Monomorphization Makes Generics Concrete

Generics are nice for programmers, but low-level code generation usually wants concrete types.

That is why Skunk has a monomorphization pass in [`src/specialization/expand/mod.rs`](../src/specialization/expand/mod.rs).

The job of the monomorphizer is to take a program that still contains generic templates and produce the concrete versions that the rest of the pipeline needs.

The important intuition is this:

A generic declaration is like a recipe.

A monomorphized declaration is like the actual finished dish for one concrete set of type arguments.

Inside `specialization/expand` you will see template-like internal structures for functions, structs, enums, traits, shapes, and impls. The pass first collects abstract definitions, then decides which concrete instances need to exist.

This is one of the biggest "pipeline cleanup" stages in the compiler. It reduces later complexity by making the program more concrete before checking and code generation.

### Read next

- [`src/specialization/expand/mod.rs`](../src/specialization/expand/mod.rs): `prepare_program`
- [`src/specialization/expand/mod.rs`](../src/specialization/expand/mod.rs): `Monomorphizer::new`, `Monomorphizer::prepare`
- [`src/specialization/expand/mod.rs`](../src/specialization/expand/mod.rs): `apply_substitutions`, `specialized_struct_name`, `specialized_function_name`

## Chapter 7: The Type Checker Is The Semantic Referee

The type checker lives in [`src/analysis/check.rs`](../src/analysis/check.rs).

If the parser answers "What is written?", the type checker answers:

- Is this legal?
- What type does this expression have?
- Is this assignment allowed?
- Are these trait or shape bounds satisfied?
- Is this unsafe operation being used in a valid place?

The main public entry point is `check`.

The most important recursive engine underneath it is `resolve_type`.

That function walks the program and tries to determine what type each expression produces. While doing that, it also validates language rules.

This is a key compiler lesson:

Type checking is not only about labels on variables. It is about proving that operations make sense.

### Read next

- [`src/analysis/check.rs`](../src/analysis/check.rs): `check`
- [`src/analysis/check.rs`](../src/analysis/check.rs): `resolve_type`, `resolve_access`, `is_assignable`
- [`src/analysis/check.rs`](../src/analysis/check.rs): `GlobalScope`, `SymbolTables`

## Chapter 8: Global Scope And Local Scope Are Different Problems

One reason `type_checker.rs` can feel large is that it has to manage both global knowledge and local knowledge.

Global knowledge includes things like:

- which structs exist
- which enums exist
- which traits exist
- which functions exist
- which trait implementations exist

Local knowledge includes things like:

- which variables are currently in scope
- whether we are inside an unsafe block
- what `self` means here
- which names shadow earlier names

In Skunk, these worlds are represented by structures like `GlobalScope` and `SymbolTables`.

This is worth understanding early because many compiler bugs happen when a system confuses "defined somewhere in the program" with "visible right here."

### Read next

- [`src/analysis/check.rs`](../src/analysis/check.rs): `GlobalScope::new`, `GlobalScope::add`
- [`src/analysis/check.rs`](../src/analysis/check.rs): `SymbolTable`, `SymbolTables`

## Chapter 9: Access Resolution Is Where Many Language Rules Meet

A lot of language behavior is hidden inside access chains.

For example:

- `point.x`
- `window.draw_rect(...)`
- `slice[0]`
- `ptr.*`
- `thing.method().field`

Skunk handles much of this through access-resolution logic in the type checker.

That code has to understand fields, methods, arrays, slices, references, pointers, dereference rules, and a few built-in pseudo-members.

So if you want to understand why the language feels the way it does to the user, access resolution is one of the best places to study.

### Read next

- [`src/analysis/check.rs`](../src/analysis/check.rs): `resolve_access`
- [`src/syntax/parser.rs`](../src/syntax/parser.rs): `access`, `access_step`

## Chapter 10: The LLVM Backend Has Its Own Vocabulary

The backend lives in [`src/backend/mod.rs`](../src/backend/mod.rs).

This file is where the compiler shifts from language-level concepts to lower-level representation.

The central type here is `LlvmType`.

That enum is the backend's vocabulary for the code generation world. It includes primitives like `I32` and `F64`, but also higher-level backend concepts like:

- `Struct(String)`
- `Enum(String)`
- `TraitObject(String)`
- `Reference { ... }`
- `Pointer { ... }`
- `Slice { ... }`

This is a big compiler milestone:

At the front end, the program is still mostly "language meaning."

At the backend, the program is increasingly about memory layout, calling conventions, storage, and emitted instructions.

### Read next

- [`src/backend/mod.rs`](../src/backend/mod.rs): `LlvmType`, `llvm_type`
- [`src/backend/mod.rs`](../src/backend/mod.rs): `compile_to_llvm_ir`

## Chapter 11: Layouts Turn Language Features Into Memory Shapes

Layouts are one of the most important concepts in the backend.

A layout answers questions like:

- How many fields does this struct have?
- In what order are those fields stored?
- How is an enum represented?
- What method slots are present in a trait object's vtable?

Skunk uses internal layout structures such as:

- `StructLayout`
- `EnumLayout`
- `TraitLayout`
- `TraitMethodLayout`

These are not source-level concepts. They are backend data structures that make code generation possible.

Without layouts, the compiler might know that a struct exists, but it would not know how to load field `x` from it.

### Read next

- [`src/backend/mod.rs`](../src/backend/mod.rs): `StructLayout`, `EnumLayout`, `TraitLayout`, `TraitMethodLayout`
- [`src/backend/mod.rs`](../src/backend/mod.rs): `collect_typed_layouts`

## Chapter 12: Trait Objects Need Both Static Proof And Runtime Data

Traits are a good example of how one language feature can affect several stages of the compiler.

At type-check time, the compiler must prove that a type satisfies a trait.

At runtime, if dynamic dispatch is used, the compiled program needs enough information to call the correct concrete method implementation.

That means trait support is split across the pipeline:

- parsing recognizes trait and conformance syntax
- monomorphization expands generic uses
- type checking validates trait satisfaction
- the backend builds trait layouts and vtables

This is a very useful lesson for compiler work:

Some features are local. Some features are whole-pipeline features.

Traits are whole-pipeline features.

### Read next

- [`src/analysis/check.rs`](../src/analysis/check.rs): trait-related parts of `GlobalScope::add`, `is_assignable`
- [`src/backend/mod.rs`](../src/backend/mod.rs): `collect_typed_layouts`
- [`src/backend/coercion.rs`](../src/backend/coercion.rs): places that construct or coerce trait objects, especially `coerce_expr`

## Chapter 13: Emitting LLVM IR Is Still Just Another Transformation

Skunk emits textual LLVM IR rather than building a giant LLVM object model through the C++ API.

That is good news for beginners, because you can inspect the generated `.ll` file and compare it to the source program.

`compile_to_llvm_ir` collects layouts and signatures, prepares function plans, and emits the final IR text.

Then `compile_to_executable` writes that IR to disk and invokes `clang`, along with the runtime support files.

This means the final binary is a collaboration between:

- generated LLVM IR
- runtime support code
- the system compiler and linker

### Read next

- [`src/backend/mod.rs`](../src/backend/mod.rs): `compile_to_llvm_ir`, `compile_to_executable`
- [`runtime/skunk_runtime.c`](../runtime/skunk_runtime.c)
- [`runtime/skunk_window_runtime.m`](../runtime/skunk_window_runtime.m)

## Chapter 14: One Native Execution Model

Skunk has one authoritative execution path. Runtime behavior tests lower programs to LLVM IR, link them with the runtime support files, and execute the resulting native binaries.

Keeping one path means language semantics, command-line execution, and compiler tests all exercise the same implementation.

### Read next

- [`src/backend/mod.rs`](../src/backend/mod.rs): native runtime tests
- [`runtime/skunk_runtime.c`](../runtime/skunk_runtime.c)

## Chapter 15: How To Read The Codebase Without Drowning

If you try to understand every file in full detail before touching anything, you will probably stall.

A better approach is:

1. Read the pipeline in order
2. Pick one tiny feature
3. Trace that feature through the stages that care about it

If the feature is syntax-heavy, start at the grammar and AST.

If the feature is semantic, spend time in the type checker.

If the feature affects runtime representation, spend time in the backend and runtime files.

The key question is:

"Where does this feature first appear, and which later stages need to know about it?"

That question is a much better guide than trying to "understand compilers in general" all at once.

### Read next

- [`src/main.rs`](../src/main.rs)
- [`src/syntax/ast.rs`](../src/syntax/ast.rs)
- [`src/analysis/check.rs`](../src/analysis/check.rs)
- [`src/backend/mod.rs`](../src/backend/mod.rs)

## Chapter 16: A Good Reading Order

Here is the reading order I recommend for this repository.

First pass:

1. [`src/main.rs`](../src/main.rs)
2. [`src/syntax/loader.rs`](../src/syntax/loader.rs)
3. [`src/syntax/ast.rs`](../src/syntax/ast.rs)
4. [`src/analysis/check.rs`](../src/analysis/check.rs)
5. [`src/backend/mod.rs`](../src/backend/mod.rs)

Second pass:

1. [`src/syntax/grammar.pest`](../src/syntax/grammar.pest)
2. [`src/specialization/expand/mod.rs`](../src/specialization/expand/mod.rs)
3. [`runtime/skunk_runtime.c`](../runtime/skunk_runtime.c)
4. [`runtime/skunk_window_runtime.m`](../runtime/skunk_window_runtime.m)

That order works well because it gives you the story first and the details second.

## Chapter 17: Native C Interop

> Status: implemented. `extern "C"` declarations, ABI type validation, and
> linker configuration through `skunk.toml` (`libraries`, `frameworks`) are in
> the compiler; see `examples/c_interop.skunk`. Header parsing, by-value
> structs, varargs, and callbacks remain out of scope, as described below.

Skunk already talks to C in one direction: every compiled program is linked
against [`runtime/skunk_runtime.c`](../runtime/skunk_runtime.c), and builtins
like `print` lower to calls into that runtime. C interop generalizes this so
Skunk code can declare and call any C function directly.

### How a call reaches C today

It helps to see that the mechanism is already in place. When the backend lowers
a builtin, it emits an LLVM `declare` for the runtime symbol and a plain `call`.
`clang` then compiles the generated `.ll` file together with the runtime C
sources (see `compile_to_executable` in
[`src/backend/mod.rs`](../src/backend/mod.rs)), and the system linker resolves the
symbol. There is no FFI layer, no marshalling, no hidden cost: a call into C is
an ordinary native call.

`extern "C"` simply exposes this existing path to user code.

### The declaration form

```text
extern "C" function cos(value: double): double;
extern "C" function sqlite3_open(path: *const byte, db: **byte): int;
```

An `extern "C"` declaration has no body. The compiler:

1. Parses it as a new statement form (a `func_decl` variant with an ABI string
   and no block).
2. Type-checks call sites against the declared signature, exactly like a normal
   function.
3. Skips lowering a body and instead emits an LLVM `declare` with the exact,
   unmangled symbol name. Extern names bypass module-private name mangling in
   the source loader — `cos` must stay `cos`.

### The type contract

Only ABI-safe types are allowed in extern signatures. The first version
permits: primitive numeric types, `bool`, raw pointers, and `void`. Structs are
allowed only behind a pointer.

The mapping to C must be pinned and documented, because the standard library
will be built on top of it and it cannot change afterwards:

| Skunk    | C          | LLVM     |
| -------- | ---------- | -------- |
| `byte`   | `uint8_t`  | `i8`     |
| `int`    | `int32_t`  | `i32`    |
| `long`   | `int64_t`  | `i64`    |
| `float`  | `float`    | `float`  |
| `double` | `double`   | `double` |
| `bool`   | `_Bool`    | `i1`     |
| `*T`     | `T*`       | `ptr`    |
| `void`   | `void`     | `void`   |

These follow the existing lowerings in `llvm_type` in
[`src/backend/mod.rs`](../src/backend/mod.rs) (`Type::Int` is already `i32`,
`Type::Byte` is `i8`, and so on), so extern calls reuse the exact
representations the backend produces everywhere else.

Passing structs *by value* is deliberately excluded: platform ABI rules for
by-value aggregates (sret, register splitting) are complex to reproduce in
textual LLVM IR, and nothing in the planned standard library needs them.
The type checker rejects any extern signature outside this set with a clear
error rather than emitting IR that miscompiles.

### Linking

Symbols from these sources resolve with no extra configuration:

- the Skunk runtime (`skunk_runtime.c`), always compiled into the binary
- libc and libm (`-lm` is added on Linux; macOS links libm by default)
- Cocoa, on macOS, where the window runtime already links it

Other libraries need linker flags, configured in `skunk.toml` once the build
tool lands:

```toml
[build]
libraries = ["sqlite3"]      # -> -lsqlite3
frameworks = ["Metal"]       # -> -framework Metal (macOS only)
```

These map directly onto arguments appended to the existing `clang` invocation.

### Limitations

- No C header parsing. Bindings are written by hand (or generated by an
  external tool later). Parsing headers correctly requires bundling a C
  frontend, which is out of scope.
- No by-value struct passing, C strings are `*const byte` (null-terminated,
  caller's responsibility), and no callbacks (passing Skunk functions as C
  function pointers) in the first version.
- No varargs: functions like `printf` cannot be declared. Wrap them in a fixed
  signature C shim in the runtime instead.
- Calling C is inherently unsafe: the compiler trusts the declared signature.
  If the declaration does not match the real C symbol, the program
  miscompiles or crashes at runtime. Declarations are only permitted in
  modules, and the standard library wraps them in safe Skunk APIs.

### Read next

- [`src/backend/mod.rs`](../src/backend/mod.rs): `compile_to_executable`, where the
  clang invocation lives
- [`runtime/skunk_runtime.c`](../runtime/skunk_runtime.c): the C side of the
  existing runtime linkage
- [`docs/pointers-and-allocators.md`](./pointers-and-allocators.md): the
  pointer model extern declarations rely on

## Chapter 18: Next Step

If this first notebook gave you the broad map, Part 2 is where we slow down and trace one small Skunk program through the compiler stage by stage.

Read it next:

- [`docs/compiler-notebook-part2.md`](./compiler-notebook-part2.md)
- [`docs/compiler-booklet.html`](./compiler-booklet.html)
