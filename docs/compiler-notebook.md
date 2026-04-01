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

In this repository, there are currently two ways to answer that question.

The older path is the interpreter. It reads the program and executes it directly inside Rust. The newer and more important path is the compiler. It reads the program, checks it, lowers it into LLVM IR, and asks `clang` to produce a native executable.

That means Skunk is not just a parser and not just a code generator. It is a whole pipeline made of loading, parsing, normalization, monomorphization, type checking, lowering, and runtime linkage.

Here is the shortest useful mental model:

```text
source files
  -> one merged program
  -> parsed AST
  -> prepared / monomorphized program
  -> type-checked program
  -> LLVM IR
  -> native binary
```

### Read next

- [`src/main.rs`](../src/main.rs): `parse_cli`, `default_output_path`, `main`
- [`src/source.rs`](../src/source.rs): `load_program`
- [`src/monomorphize.rs`](../src/monomorphize.rs): `prepare_program`
- [`src/type_checker.rs`](../src/type_checker.rs): `check`
- [`src/compiler.rs`](../src/compiler.rs): `compile_to_executable`, `compile_to_llvm_ir`

## Chapter 2: `main.rs` Is The Pipeline Coordinator

The easiest place to understand the project shape is [`src/main.rs`](../src/main.rs).

That file does not contain the language logic itself. Instead, it coordinates the major stages.

The important thing to notice is the order:

1. Parse the CLI
2. Load the program
3. Prepare the program
4. Type-check the program
5. Interpret it or compile it

That order tells you what the rest of the repository expects. For example, the compiler backend assumes it receives a program that has already been loaded, normalized, and checked.

This is a useful compiler lesson:

Big functions at the top of a compiler often tell you more about architecture than any design document does.

### Read next

- [`src/main.rs`](../src/main.rs): `main`
- [`src/source.rs`](../src/source.rs): `load_program`
- [`src/compiler.rs`](../src/compiler.rs): `compile_to_executable`

## Chapter 3: Parsing Means "Turn Text Into Structure"

Before the compiler can reason about a program, it needs to stop seeing the program as raw text.

That happens in two layers.

The first layer is the grammar in [`src/grammar.pest`](../src/grammar.pest). This file describes what valid Skunk source looks like.

The second layer is AST construction in [`src/ast.rs`](../src/ast.rs). This file turns parsed grammar pieces into the compiler's own tree representation.

The main idea is simple:

- the grammar recognizes source forms
- the AST gives those forms names the compiler can work with

For example, the compiler does not want to keep asking, "is this sequence of characters a struct initialization?" It wants a node like `Node::StructInitialization`.

That is why ASTs matter. They replace text with structure.

### Read next

- [`src/grammar.pest`](../src/grammar.pest)
- [`src/ast.rs`](../src/ast.rs): `PestImpl::parse`, `create_ast`, `create_primary`, `create_access`, `create_struct_init`, `parse`

## Chapter 4: The AST Is The Shared Language Of The Compiler

The `Node` enum in [`src/ast.rs`](../src/ast.rs) is the compiler's common vocabulary.

Many parts of the compiler talk in terms of `Node` and `Type`, including:

- the parser
- the source loader
- the monomorphizer
- the type checker
- the interpreter
- the LLVM backend

This is powerful because it keeps the project easy to extend. A new language feature can often be added by introducing new AST cases and teaching a few later stages how to handle them.

It also means the AST stays important for a long time. Skunk does not yet have a large stack of middle IR layers between the front end and LLVM.

That is why understanding `Node` and `Type` pays off quickly. They are everywhere.

### Read next

- [`src/ast.rs`](../src/ast.rs): `Node`, `Type`, `TraitMethodSignature`, `MatchPattern`, `type_to_string`

## Chapter 5: Loading Modules Is A Real Compiler Pass

New compiler readers sometimes think the "real" compiler starts only after parsing. In practice, loading source files is already compiler work.

That logic lives in [`src/source.rs`](../src/source.rs).

`load_program` does more than read a single file. It:

- resolves the entry path
- loads imports recursively
- detects cyclic imports
- checks module names
- normalizes visibility and private names
- returns one merged `Node::Program`

This is a very important simplification for later stages. Instead of every later pass needing to think about a graph of files, most of the pipeline gets to think about one program tree.

The other important piece in this file is `ModuleNormalizer`. It rewrites private names from imported modules so they do not collide later.

That means `source.rs` is where "many source files" becomes "one safe program to analyze."

### Read next

- [`src/source.rs`](../src/source.rs): `load_program`
- [`src/source.rs`](../src/source.rs): `ProgramLoader::load_file`, `ProgramLoader::module_path`
- [`src/source.rs`](../src/source.rs): `ModuleNormalizer::new`, `ModuleNormalizer::normalize`, `rename_top_level`

## Chapter 6: Monomorphization Makes Generics Concrete

Generics are nice for programmers, but low-level code generation usually wants concrete types.

That is why Skunk has a monomorphization pass in [`src/monomorphize.rs`](../src/monomorphize.rs).

The job of the monomorphizer is to take a program that still contains generic templates and produce the concrete versions that the rest of the pipeline needs.

The important intuition is this:

A generic declaration is like a recipe.

A monomorphized declaration is like the actual finished dish for one concrete set of type arguments.

Inside `monomorphize.rs` you will see template-like internal structures for functions, structs, enums, traits, shapes, and impls. That is because the pass first collects abstract definitions, then decides which concrete instances need to exist.

This is one of the biggest "pipeline cleanup" stages in the compiler. It reduces later complexity by making the program more concrete before checking and code generation.

### Read next

- [`src/monomorphize.rs`](../src/monomorphize.rs): `prepare_program`
- [`src/monomorphize.rs`](../src/monomorphize.rs): `Monomorphizer::new`, `Monomorphizer::prepare`
- [`src/monomorphize.rs`](../src/monomorphize.rs): `apply_substitutions`, `specialized_struct_name`, `specialized_function_name`

## Chapter 7: The Type Checker Is The Semantic Referee

The type checker lives in [`src/type_checker.rs`](../src/type_checker.rs).

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

- [`src/type_checker.rs`](../src/type_checker.rs): `check`
- [`src/type_checker.rs`](../src/type_checker.rs): `resolve_type`, `resolve_access`, `is_assignable`
- [`src/type_checker.rs`](../src/type_checker.rs): `GlobalScope`, `SymbolTables`

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

- [`src/type_checker.rs`](../src/type_checker.rs): `GlobalScope::new`, `GlobalScope::add`
- [`src/type_checker.rs`](../src/type_checker.rs): `SymbolTable`, `SymbolTables`

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

- [`src/type_checker.rs`](../src/type_checker.rs): `resolve_access`
- [`src/ast.rs`](../src/ast.rs): `create_access`

## Chapter 10: The LLVM Backend Has Its Own Vocabulary

The backend lives in [`src/compiler.rs`](../src/compiler.rs).

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

- [`src/compiler.rs`](../src/compiler.rs): `LlvmType`, `llvm_type`
- [`src/compiler.rs`](../src/compiler.rs): `compile_to_llvm_ir`

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

- [`src/compiler.rs`](../src/compiler.rs): `StructLayout`, `EnumLayout`, `TraitLayout`, `TraitMethodLayout`
- [`src/compiler.rs`](../src/compiler.rs): `collect_struct_layouts`, `collect_enum_layouts`, `collect_trait_layouts`

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

- [`src/type_checker.rs`](../src/type_checker.rs): trait-related parts of `GlobalScope::add`, `is_assignable`
- [`src/compiler.rs`](../src/compiler.rs): `collect_trait_layouts`
- [`src/compiler.rs`](../src/compiler.rs): places that construct or coerce trait objects, especially `coerce_expr`

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

- [`src/compiler.rs`](../src/compiler.rs): `compile_to_llvm_ir`, `compile_to_executable`
- [`runtime/skunk_runtime.c`](../runtime/skunk_runtime.c)
- [`runtime/skunk_window_runtime.m`](../runtime/skunk_window_runtime.m)

## Chapter 14: The Interpreter Still Has Educational Value

Skunk still contains an interpreter in [`src/interpreter.rs`](../src/interpreter.rs).

Even though native compilation is the main path, the interpreter remains useful as:

- a semantic reference
- a fallback execution model for some features
- a source of tests and examples
- a reminder of the language's intended behavior independent of LLVM details

In young language projects, it is normal for interpreter and compiler paths to coexist for a while.

### Read next

- [`src/interpreter.rs`](../src/interpreter.rs): `evaluate`
- [`src/interpreter.rs`](../src/interpreter.rs): `evaluate_node`

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
- [`src/ast.rs`](../src/ast.rs)
- [`src/type_checker.rs`](../src/type_checker.rs)
- [`src/compiler.rs`](../src/compiler.rs)

## Chapter 16: A Good Reading Order

Here is the reading order I recommend for this repository.

First pass:

1. [`src/main.rs`](../src/main.rs)
2. [`src/source.rs`](../src/source.rs)
3. [`src/ast.rs`](../src/ast.rs)
4. [`src/type_checker.rs`](../src/type_checker.rs)
5. [`src/compiler.rs`](../src/compiler.rs)

Second pass:

1. [`src/grammar.pest`](../src/grammar.pest)
2. [`src/monomorphize.rs`](../src/monomorphize.rs)
3. [`src/interpreter.rs`](../src/interpreter.rs)
4. [`runtime/skunk_runtime.c`](../runtime/skunk_runtime.c)
5. [`runtime/skunk_window_runtime.m`](../runtime/skunk_window_runtime.m)

That order works well because it gives you the story first and the details second.

## Chapter 17: Next Step

If this first notebook gave you the broad map, Part 2 is where we slow down and trace one small Skunk program through the compiler stage by stage.

Read it next:

- [`docs/compiler-notebook-part2.md`](./compiler-notebook-part2.md)
- [`docs/compiler-booklet.html`](./compiler-booklet.html)
