# Skunk Compiler Notebook

## Part 2: One Small Program, Traced Through The Compiler

Part 1 gave you the map.

Part 2 gives you a guided walk.

Instead of talking about the compiler in the abstract, we will use one very small Skunk program and follow it through the important stages.

The goal is not to memorize every detail. The goal is to build intuition.

## Chapter 1: The Example Program

Here is the program we will follow:

```skunk
struct Point {
    x: int;
    y: int;
}

attach Point {
    function sum(self): int {
        return self.x + self.y;
    }
}

function main(): int {
    p: Point = Point { x: 20, y: 22 };
    return p.sum();
}
```

This program is useful because it includes several common language ideas without becoming complicated:

- a struct declaration
- an attach block
- a method
- a struct literal
- a local variable
- a method call
- a return value

That is enough to exercise most of the major compiler stages in a readable way.

### Read next

- [`src/ast.rs`](../src/ast.rs): `create_struct_init`, `create_access`
- [`src/type_checker.rs`](../src/type_checker.rs): `resolve_access`, `resolve_type`
- [`src/compiler.rs`](../src/compiler.rs): `collect_struct_layouts`, `compile_struct_literal`, `compile_expr_with_expected`

## Chapter 2: The Pipeline Diagram

Before we trace the example, here is the shape of the journey:

```text
source text
  |
  v
grammar + parser
  |
  v
AST (Node::Program)
  |
  v
source loading / normalization
  |
  v
monomorphization / preparation
  |
  v
type checking
  |
  v
LLVM layout collection
  |
  v
LLVM IR emission
  |
  v
clang + runtime files
  |
  v
native executable
```

For this tiny program, some stages do only a little work. That is fine. One of the best lessons in compiler reading is realizing that not every pass transforms every program in a dramatic way.

### Read next

- [`src/main.rs`](../src/main.rs): `main`
- [`src/source.rs`](../src/source.rs): `load_program`
- [`src/monomorphize.rs`](../src/monomorphize.rs): `prepare_program`
- [`src/compiler.rs`](../src/compiler.rs): `compile_to_llvm_ir`

## Chapter 3: Parsing The Program

The parser sees a string of characters and turns it into a tree.

For our example, the rough AST shape looks like this:

```text
Program
  StructDeclaration("Point")
    fields:
      x: int
      y: int
  AttachDeclaration(target = Point)
    FunctionDeclaration("sum")
      body:
        Return(
          BinaryOp(
            MemberAccess(self.x),
            Add,
            MemberAccess(self.y)
          )
        )
  FunctionDeclaration("main")
    body:
      VariableDeclaration("p", Point, StructInitialization(Point))
      Return(
        Access(
          Identifier("p")
          MemberAccess(FunctionCall("sum"))
        )
      )
```

This tree is not a perfect printout of the real `Node` values, but it is a good mental approximation.

The two parser ideas to focus on are:

1. the parser identifies what syntactic thing each piece of source is
2. expressions become nested trees, not flat text

That is why `self.x + self.y` becomes a binary operation over two access expressions instead of remaining a raw string.

### Read next

- [`src/ast.rs`](../src/ast.rs): `PestImpl::parse`
- [`src/ast.rs`](../src/ast.rs): `create_ast`
- [`src/ast.rs`](../src/ast.rs): `create_primary`, `create_access`, `create_struct_init`

## Chapter 4: Source Loading And Normalization

This example has no imports, so `source::load_program` has a quiet job here.

It still matters, though.

Even in this simple case, the source loader gives later stages one consistent `Node::Program` root. In bigger programs it would also resolve imports, detect cycles, and normalize private names.

One useful lesson here is that some passes are more visible on larger programs. That does not make them unimportant.

The source loader is still a foundational stage because every later pass depends on getting one coherent program tree.

### Read next

- [`src/source.rs`](../src/source.rs): `load_program`
- [`src/source.rs`](../src/source.rs): `ProgramLoader::load_file`
- [`src/source.rs`](../src/source.rs): `ModuleNormalizer::normalize`

## Chapter 5: Monomorphization On A Non-Generic Program

Our example is not generic.

That means the monomorphizer does not need to produce specialized copies of functions or structs here. The program mostly passes through unchanged.

This is a good thing to notice.

Monomorphization is not "magic that always changes everything." Its job is to make generic programs concrete when needed. If a program is already concrete, the pass may mostly collect information and hand the program along.

That is one reason learning with a simple example helps. You can see which stages are active, but you are not overwhelmed by every feature at once.

### Read next

- [`src/monomorphize.rs`](../src/monomorphize.rs): `prepare_program`
- [`src/monomorphize.rs`](../src/monomorphize.rs): `Monomorphizer::new`, `Monomorphizer::prepare`

## Chapter 6: Type Checking The Struct And The Method

Now the compiler starts asking semantic questions.

For the struct declaration:

- does `Point` define a legal set of fields?
- do the field types make sense?

For the attach block:

- is `Point` a known target type?
- is `sum` a valid method shape?
- does the body return an `int` as promised?

For the main function:

- is `Point { x: 20, y: 22 }` a valid struct literal?
- do both fields exist?
- are the field value types assignable to the declared field types?
- is `p.sum()` a valid method call?
- does `main` really return `int`?

The important thing to feel here is that the type checker is not simply reading declared types. It is proving that the operations line up correctly.

### Read next

- [`src/type_checker.rs`](../src/type_checker.rs): `check`
- [`src/type_checker.rs`](../src/type_checker.rs): `resolve_type`
- [`src/type_checker.rs`](../src/type_checker.rs): `resolve_access`
- [`src/type_checker.rs`](../src/type_checker.rs): `is_assignable`

## Chapter 7: What The Type Checker Knows At This Point

By the time type checking succeeds, the compiler has learned several important facts:

- `Point` is a real nominal type
- `Point` has two `int` fields
- `Point.sum` is a method that returns `int`
- `p` is a local variable of type `Point`
- `p.sum()` is an expression of type `int`
- `main` returns `int`

This is an important compiler milestone.

The program is no longer just "syntactically valid text." It has become a semantically validated program with known types and allowed operations.

That validated meaning is what the backend will rely on.

### Read next

- [`src/type_checker.rs`](../src/type_checker.rs): `GlobalScope::add`
- [`src/type_checker.rs`](../src/type_checker.rs): `SymbolTables`

## Chapter 8: Building Layouts For The Backend

Now we move into LLVM-oriented work.

The backend starts by collecting layouts.

For our example, the most important layout is the struct layout for `Point`.

Conceptually, the backend wants to know something like this:

```text
StructLayout("Point")
  field 0: x -> i32
  field 1: y -> i32
```

This is where the compiler stops thinking only in terms of source names and starts thinking in terms of storage positions and LLVM types.

For this small program there are no enums and no trait objects, so `EnumLayout` and `TraitLayout` are not the stars of the show. But the same general idea would apply if they were present.

### Read next

- [`src/compiler.rs`](../src/compiler.rs): `LlvmType`
- [`src/compiler.rs`](../src/compiler.rs): `llvm_type`
- [`src/compiler.rs`](../src/compiler.rs): `collect_struct_layouts`
- [`src/compiler.rs`](../src/compiler.rs): `collect_enum_layouts`, `collect_trait_layouts`

## Chapter 9: Lowering Expressions Into LLVM IR

Once layouts and function signatures are known, the backend can start lowering function bodies.

For our example, there are two especially interesting pieces:

1. the struct literal `Point { x: 20, y: 22 }`
2. the method call `p.sum()`

The struct literal lowering needs to:

- allocate or build a value of the correct LLVM struct type
- place `20` into field `x`
- place `22` into field `y`

The method call lowering needs to:

- find the right function symbol for `Point.sum`
- pass the receiver correctly
- produce an `int` result

This is where the backend starts to feel operational. It is no longer asking what the program means. It is asking how to emit instructions that make the machine do the right thing.

### Read next

- [`src/compiler.rs`](../src/compiler.rs): `compile_statement`
- [`src/compiler.rs`](../src/compiler.rs): `compile_expr_with_expected`
- [`src/compiler.rs`](../src/compiler.rs): `compile_struct_literal`
- [`src/compiler.rs`](../src/compiler.rs): `coerce_expr`

## Chapter 10: A Tiny Mental Model Of The Emitted IR

You do not need to know LLVM deeply to build a useful intuition.

For this example, the important mental picture is:

```text
define i32 @skunk_main() {
  ; build Point
  ; store 20 into field 0
  ; store 22 into field 1
  ; call Point.sum
  ; return the resulting i32
}
```

The actual IR will be more detailed than this. There will be temporaries, loads, stores, and concrete LLVM syntax.

But the high-level structure is not mysterious. It is still just the program being translated into a lower-level language.

A very good beginner exercise is to compile this kind of tiny example, open the `.ll` file, and try to match the source constructs to the emitted blocks and instructions.

### Read next

- [`src/compiler.rs`](../src/compiler.rs): `compile_to_llvm_ir`

## Chapter 11: Linking With Runtime Support

After IR generation, `compile_to_executable` writes a `.ll` file and invokes `clang`.

That means the final executable is not produced by Skunk alone.

It is produced by:

- the Skunk frontend and backend
- the runtime support files
- the system toolchain

For this tiny `Point` example, the runtime is not doing much visible work. But the same linkage path matters for richer programs that need memory helpers or window/input support.

This is another good beginner lesson:

Compilers often rely on runtimes and system toolchains. That is normal. It is not cheating.

### Read next

- [`src/compiler.rs`](../src/compiler.rs): `compile_to_executable`
- [`runtime/skunk_runtime.c`](../runtime/skunk_runtime.c)
- [`runtime/skunk_window_runtime.m`](../runtime/skunk_window_runtime.m)

## Chapter 12: A Stage-By-Stage Memory Aid

If you want one page to keep in your head while reading code, use this:

```text
main.rs
  Coordinates the pipeline

source.rs
  Turns many files into one program

ast.rs
  Turns syntax into compiler nodes

monomorphize.rs
  Makes generic definitions concrete when needed

type_checker.rs
  Proves the program is semantically legal

compiler.rs
  Turns the checked program into LLVM IR and a native binary

runtime/*.c / *.m
  Provide support code the compiled program can call
```

That summary is not complete, but it is enough to stay oriented.

## Chapter 13: What You Should Try Next

Once you are comfortable with this tiny example, the next great exercises are:

1. Repeat the trace with a generic function so you can see monomorphization matter more.
2. Repeat it with a trait and `conform` so you can see type-checking and backend trait support interact.
3. Repeat it with a windowed example so you can see where runtime support becomes visible.

That progression is nice because it adds only one major idea at a time.

### Read next

- [`examples/pong.skunk`](../examples/pong.skunk)
- [`docs/compiler-booklet.html`](./compiler-booklet.html)
