use super::*;
use crate::source;
use std::env;
use std::path::Path;
use std::process::Command;
use uuid::Uuid;

fn loaded_source(source: &str) -> crate::syntax::loader::LoadedProgram {
    let mut sources = crate::source_map::SourceMap::default();
    let file = sources.add_file("compiler-test.skunk", source).unwrap();
    let module = crate::syntax::parser::parse_module(&sources, file).unwrap();
    let module = crate::syntax::normalize::normalize(module).unwrap();
    crate::syntax::loader::LoadedProgram { module, sources }
}

fn compile_and_run(source: &str) -> Result<String, String> {
    let program = crate::pipeline::check_source("compiler-test.skunk", source)?;

    let id = Uuid::new_v4().to_string();
    let source_path = env::temp_dir().join(format!("skunk_compiler_test_{}.skunk", id));
    let output_path = env::temp_dir().join(format!("skunk_compiler_test_{}", id));
    fs::write(&source_path, source)
        .map_err(|err| format!("failed to write test source: {}", err))?;

    let artifact = compile_to_executable(&program, &source_path, &output_path)?;
    let output = Command::new(&artifact.binary_path)
        .output()
        .map_err(|err| format!("failed to run compiled test binary: {}", err))?;

    let _ = fs::remove_file(&source_path);
    let _ = fs::remove_file(&artifact.llvm_ir_path);
    let _ = fs::remove_file(&artifact.binary_path);

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!(
            "compiled program exited with status {}: {}",
            output.status,
            stderr.trim()
        ));
    }

    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

#[test]
fn nested_block_return_is_recognized_and_runs() {
    let output = compile_and_run(
        r#"
            function value(): int {
                {
                    return 42;
                }
            }

            function main(): void {
                print(value());
            }
        "#,
    )
    .unwrap();

    assert_eq!(output, "42\n");
}

#[test]
fn nested_block_local_does_not_escape_during_compilation() {
    let error = compile_and_run(
        r#"
            function main(): void {
                { hidden: int = 1; }
                print(hidden);
            }
        "#,
    )
    .unwrap_err();

    assert!(error.contains("unknown value `hidden`"));
}

fn compile_and_run_with_env(source: &str, env_vars: &[(&str, &str)]) -> Result<String, String> {
    let program = crate::pipeline::check_source("compiler-test.skunk", source)?;

    let id = Uuid::new_v4().to_string();
    let source_path = env::temp_dir().join(format!("skunk_compiler_test_{}.skunk", id));
    let output_path = env::temp_dir().join(format!("skunk_compiler_test_{}", id));
    fs::write(&source_path, source)
        .map_err(|err| format!("failed to write test source: {}", err))?;

    let artifact = compile_to_executable(&program, &source_path, &output_path)?;
    let mut command = Command::new(&artifact.binary_path);
    for (key, value) in env_vars {
        command.env(key, value);
    }
    let output = command
        .output()
        .map_err(|err| format!("failed to run compiled test binary: {}", err))?;

    let _ = fs::remove_file(&source_path);
    let _ = fs::remove_file(&artifact.llvm_ir_path);
    let _ = fs::remove_file(&artifact.binary_path);

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!(
            "compiled program exited with status {}: {}",
            output.status,
            stderr.trim()
        ));
    }

    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

fn compile_project_and_run(files: &[(&str, &str)], entry: &str) -> Result<String, String> {
    let root = env::temp_dir().join(format!("skunk_compiler_project_{}", Uuid::new_v4()));
    fs::create_dir_all(&root)
        .map_err(|err| format!("failed to create test project root: {}", err))?;

    for (relative_path, contents) in files {
        let path = root.join(relative_path);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .map_err(|err| format!("failed to create `{}`: {}", parent.display(), err))?;
        }
        fs::write(&path, contents)
            .map_err(|err| format!("failed to write `{}`: {}", path.display(), err))?;
    }

    let entry_path = root.join(entry);
    let program = crate::pipeline::check_loaded(source::load_program(&entry_path)?)?;

    let output_path = root.join("app_out");
    let artifact = compile_to_executable(&program, Path::new(&entry_path), &output_path)?;
    let output = Command::new(&artifact.binary_path)
        .output()
        .map_err(|err| format!("failed to run compiled test binary: {}", err))?;

    let _ = fs::remove_dir_all(&root);

    if !output.status.success() {
        return Err(format!(
            "compiled program exited with status {}",
            output.status
        ));
    }

    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

#[test]
fn materialized_file_writes_are_safe_when_concurrent() {
    use std::sync::{Arc, Barrier};
    use std::thread;

    let root = env::temp_dir().join(format!("skunk_materialization_test_{}", Uuid::new_v4()));
    fs::create_dir_all(&root).unwrap();
    let path = Arc::new(root.join("skunk_runtime.c"));
    let contents = Arc::new("/* embedded runtime */\n".repeat(16_384));
    let barrier = Arc::new(Barrier::new(16));

    let writers = (0..16)
        .map(|_| {
            let path = Arc::clone(&path);
            let contents = Arc::clone(&contents);
            let barrier = Arc::clone(&barrier);
            thread::spawn(move || {
                barrier.wait();
                write_if_changed(&path, &contents)
            })
        })
        .collect::<Vec<_>>();

    for writer in writers {
        writer.join().unwrap().unwrap();
    }
    assert_eq!(fs::read_to_string(path.as_ref()).unwrap(), *contents);
    assert_eq!(fs::read_dir(&root).unwrap().count(), 1);
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn compiles_basic_program_to_ir() {
    let program = crate::pipeline::check_source(
        "compiler-test.skunk",
        r#"
        function add(a: int, b: int): int {
            return a + b;
        }

        function main(): void {
            total: int = add(2, 3);
            print(total);
        }
        "#,
    )
    .unwrap();

    let ir = compile_to_llvm_ir(&program).unwrap();
    assert!(ir.contains("define i32 @skunk_add(i32 %arg0, i32 %arg1)"));
    assert!(ir.contains("define void @skunk_main()"));
    assert!(ir.contains("call i32 (ptr, ...) @printf"));
}

#[test]
fn compiles_structs_to_ir() {
    let program = crate::pipeline::check_source(
        "compiler-test.skunk",
        r#"
        struct Point {
            x: int;
            y: int;
        }

        attach Point {
            function sum(self): int {
                return self.x + self.y;
            }
        }

        function main(): void {
            p: Point = Point { x: 2, y: 3 };
            print(p.sum());
        }
        "#,
    )
    .unwrap();

    let ir = compile_to_llvm_ir(&program).unwrap();
    assert!(ir.contains("%struct.Point = type { i32, i32 }"));
    assert!(ir.contains("define i32 @skunk_Point_sum(ptr %arg0)"));
    assert!(ir.contains("insertvalue %struct.Point"));
    assert!(ir.contains("getelementptr inbounds %struct.Point"));
}

#[test]
fn compiles_new_primitives_to_ir() {
    let program = crate::pipeline::check_source(
        "compiler-test.skunk",
        r#"
        function main(): double {
            b: byte = 10;
            s: short = 20;
            l: long = 30L;
            f: float = 1.5f;
            c: char = 'A';
            print(c);
            return l + f;
        }
        "#,
    )
    .unwrap();

    let ir = compile_to_llvm_ir(&program).unwrap();
    assert!(ir.contains("define double @skunk_main()"));
    assert!(ir.contains("sitofp i64"));
    assert!(ir.contains("fadd float"));
}

#[test]
fn compiles_fixed_arrays_to_ir() {
    let program = crate::pipeline::check_source(
        "compiler-test.skunk",
        r#"
        function main(): void {
            values: [3]int = [1, 2, 3];
            values[1] = values[0] + 9;
            print(values[1]);
            print(values.len);
        }
        "#,
    )
    .unwrap();

    let ir = compile_to_llvm_ir(&program).unwrap();
    assert!(ir.contains("[3 x i32]"));
    assert!(ir.contains("getelementptr inbounds [3 x i32]"));
    assert!(ir.contains("insertvalue [3 x i32]"));
    assert!(ir.contains("call void @skunk_panic_index_out_of_bounds"));
}

#[test]
fn runs_compiled_fixed_array_program() {
    let stdout = compile_and_run(
        r#"
        function main(): void {
            zeros: [3]int;
            filled: [3]int = [3]int::fill(7);
            filled[1] = filled[1] + 1;
            print(zeros[0]);
            print(filled[1]);
            print(filled.len);
        }
        "#,
    )
    .unwrap();

    assert_eq!(stdout, "0\n8\n3\n");
}

#[test]
fn fixed_array_index_out_of_bounds_panics() {
    let error = compile_and_run(
        r#"
        function main(): void {
            values: [3]int = [10, 20, 30];
            index: int = 3;
            print(values[index]);
        }
        "#,
    )
    .unwrap_err();

    assert!(error.contains("panic: index 3 out of bounds for length 3"));
}

#[test]
fn negative_array_index_panics() {
    let error = compile_and_run(
        r#"
        function main(): void {
            values: [2]int = [10, 20];
            index: int = -1;
            values[index] = 99;
        }
        "#,
    )
    .unwrap_err();

    assert!(error.contains("panic: index -1 out of bounds for length 2"));
}

#[test]
fn runs_compiled_nested_array_program() {
    let stdout = compile_and_run(
        r#"
        function main(): void {
            matrix: [2][2]int = [
                [1, 2],
                [3, 4]
            ];
            matrix[1][0] = matrix[0][1] + 5;
            print(matrix[1][0]);
            print(matrix[0].len);
        }
        "#,
    )
    .unwrap();

    assert_eq!(stdout, "7\n2\n");
}

#[test]
fn runs_compiled_array_return_and_argument_program() {
    let stdout = compile_and_run(
        r#"
        function make(): [3]int {
            return [3]int::fill(2);
        }

        function sum(values: [3]int): int {
            return values[0] + values[1] + values[2];
        }

        function main(): void {
            values: [3]int = make();
            values[1] = 5;
            print(sum(values));
        }
        "#,
    )
    .unwrap();

    assert_eq!(stdout, "9\n");
}

#[test]
fn runs_compiled_struct_field_program() {
    let stdout = compile_and_run(
        r#"
        struct Point {
            x: int;
            y: int;
        }

        function main(): void {
            p: Point = Point { x: 1, y: 2 };
            p.x = p.x + 9;
            print(p.x);
            print(p.y);
        }
        "#,
    )
    .unwrap();

    assert_eq!(stdout, "10\n2\n");
}

#[test]
fn runs_compiled_struct_field_shorthand_program() {
    let stdout = compile_and_run(
        r#"
        struct Point {
            x: int;
            y: int;
        }

        function main(): void {
            x: int = 3;
            y: int = 4;
            shorthand: Point = Point { x, y };
            explicit: Point = Point { x: x, y: y };
            print(shorthand.x);
            print(shorthand.y);
            print(explicit.x);
            print(explicit.y);
        }
        "#,
    )
    .unwrap();

    assert_eq!(stdout, "3\n4\n3\n4\n");
}

#[test]
fn runs_compiled_defer_on_scope_exit_return_and_loop_iterations() {
    let stdout = compile_and_run(
        r#"
        function log(value: int): void {
            print(value);
        }

        function finish(early: boolean): int {
            defer log(1);
            if (early) {
                defer log(2);
                defer log(3);
                return 7;
            }
            defer log(4);
            return 8;
        }

        function shadowed_return(): void {
            value: int = 1;
            defer log(value);
            {
                value: int = 2;
                return;
            }
        }

        function main(): void {
            print(finish(true));
            print(finish(false));
            shadowed_return();

            {
                defer log(5);
                defer log(6);
                log(0);
            }

            for (i: int = 0; i < 2; i = i + 1) {
                defer log(i);
                log(i + 10);
            }

            marker: int = 9;
            defer log(marker);
            marker = 10;

            bytes: [1]byte;
            unsafe {
                defer Memory::set(&bytes[0], 12, 1);
            }
            print(bytes[0]);
        }
        "#,
    )
    .unwrap();

    assert_eq!(
        stdout,
        "3\n2\n1\n7\n4\n1\n8\n1\n0\n6\n5\n10\n0\n11\n1\n12\n10\n"
    );
}

#[test]
fn rejects_safe_defer_unwound_from_an_unsafe_return() {
    let result = compile_and_run(
        r#"
        function main(): void {
            value: int = 0;
            defer Memory::set(&value, 7, int::size_of());
            unsafe {
                return;
            }
        }
        "#,
    );

    assert!(result.is_err());
    assert!(result.unwrap_err().contains("unsafe block"));
}

#[test]
fn runs_compiled_nested_struct_program() {
    let stdout = compile_and_run(
        r#"
        struct Point {
            x: int;
            y: int;
        }

        struct Line {
            start: Point;
            end: Point;
        }

        function main(): void {
            line: Line = Line {
                start: Point { x: 3, y: 4 },
                end: Point { x: 5, y: 6 }
            };
            line.start.x = line.start.x + line.end.y;
            print(line.start.x);
            print(line.end.x);
        }
        "#,
    )
    .unwrap();

    assert_eq!(stdout, "9\n5\n");
}

#[test]
fn runs_compiled_struct_method_program() {
    let stdout = compile_and_run(
        r#"
        struct Point {
            x: int;
            y: int;
        }

        attach Point {
            function set_x(mut self, x: int): void {
                self.x = x;
            }

            function sum(self): int {
                return self.x + self.y;
            }
        }

        function main(): void {
            p: Point = Point { x: 1, y: 2 };
            p.set_x(10);
            print(p.sum());
        }
        "#,
    )
    .unwrap();

    assert_eq!(stdout, "12\n");
}

#[test]
fn runs_compiled_static_attached_function_program() {
    let stdout = compile_and_run(
        r#"
        struct Point {
            x: int;
            y: int;
        }

        attach Point {
            function new(x: int, y: int): Point {
                return Point { x: x, y: y };
            }

            function origin(): Point {
                return Point { x: 0, y: 0 };
            }

            function sum(self): int {
                return self.x + self.y;
            }
        }

        function main(): void {
            point: Point = Point::new(4, 9);
            origin: Point = Point::origin();
            print(point.sum());
            print(origin.sum());
        }
        "#,
    )
    .unwrap();

    assert_eq!(stdout, "13\n0\n");
}

#[test]
fn runs_compiled_mut_self_method_program() {
    let stdout = compile_and_run(
        r#"
        struct Counter {
            value: int;
        }

        attach Counter {
            function bump(mut self): void {
                self.value = self.value + 1;
            }

            function get(self): int {
                return self.value;
            }
        }

        function main(): void {
            counter: Counter = Counter { value: 9 };
            counter.bump();
            print(counter.get());
        }
        "#,
    )
    .unwrap();

    assert_eq!(stdout, "10\n");
}

#[test]
fn runs_compiled_struct_destructure_program() {
    let stdout = compile_and_run(
        r#"
        struct Point {
            x: int;
            y: int;
        }

        function main(): void {
            point: Point = Point { x: 5, y: 8 };
            Point { x, y: py } = point;
            print(x);
            print(py);
            print(x + py);
        }
        "#,
    )
    .unwrap();

    assert_eq!(stdout, "5\n8\n13\n");
}

#[test]
fn runs_compiled_struct_match_pattern_program() {
    let stdout = compile_and_run(
        r#"
        struct Point {
            x: int;
            y: int;
        }

        function sum(point: Point): int {
            match (point) {
                case Point { x, y }: {
                    return x + y;
                }
            }
        }

        function main(): void {
            print(sum(Point { x: 2, y: 9 }));
        }
        "#,
    )
    .unwrap();

    assert_eq!(stdout, "11\n");
}

#[test]
fn rejects_calling_mut_self_method_on_const_binding() {
    let result = compile_and_run(
        r#"
        struct Counter {
            value: int;
        }

        attach Counter {
            function bump(mut self): void {
                self.value = self.value + 1;
            }
        }

        function main(): void {
            const counter: Counter = Counter { value: 0 };
            counter.bump();
        }
        "#,
    );

    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .contains("cannot call mutating method through const or immutable receiver"));
}

#[test]
fn runs_compiled_const_pointer_readonly_method_program() {
    let stdout = compile_and_run(
        r#"
        struct Point {
            x: int;
        }

        attach Point {
            function get(self): int {
                return self.x;
            }
        }

        function print_point(point: *const Point): void {
            print(point.get());
        }

        function main(): void {
            heap: Allocator = System::allocator();
            point: *Point = Point::create(heap);
            point.x = 7;
            print_point(point);
            heap.destroy(point);
        }
        "#,
    )
    .unwrap();

    assert_eq!(stdout, "7\n");
}

#[test]
fn runs_compiled_slice_program() {
    let stdout = compile_and_run(
        r#"
        function sum_pair(values: []int): int {
            pair: []int = values[1:3];
            return pair[0] + pair[1];
        }

        function main(): void {
            values: [5]int = [10, 20, 30, 40, 50];
            middle: []int = values[1:4];
            head: []int = values[:2];
            tail: []int = middle[1:];
            print(middle.len);
            print(head[1]);
            print(tail[0]);
            print(sum_pair(values[:4]));
        }
        "#,
    )
    .unwrap();

    assert_eq!(stdout, "3\n20\n30\n50\n");
}

#[test]
fn slice_index_out_of_bounds_panics() {
    let error = compile_and_run(
        r#"
        function main(): void {
            values: []int = [10, 20];
            index: int = values.len;
            print(values[index]);
        }
        "#,
    )
    .unwrap_err();

    assert!(error.contains("panic: index 2 out of bounds for length 2"));
}

#[test]
fn invalid_slice_range_panics() {
    let error = compile_and_run(
        r#"
        function main(): void {
            values: [4]int = [10, 20, 30, 40];
            start: int = 3;
            end: int = 2;
            invalid: []int = values[start:end];
            print(invalid.len);
        }
        "#,
    )
    .unwrap_err();

    assert!(error.contains("panic: slice range [3:2] out of bounds for length 4"));
}

#[test]
fn explicit_bounds_check_uses_logical_length() {
    let error = compile_and_run(
        r#"
        function main(): void {
            backing_capacity: int = 8;
            logical_length: int = 2;
            index: int = 3;
            Bounds::check(index, logical_length);
            print(backing_capacity);
        }
        "#,
    )
    .unwrap_err();

    assert!(error.contains("panic: index 3 out of bounds for length 2"));
}

#[test]
fn runs_compiled_slice_literal_program() {
    let stdout = compile_and_run(
        r#"
        function main(): void {
            values: []int = [1, 2, 3];
            print(values.len);
            print(values[2]);
        }
        "#,
    )
    .unwrap();

    assert_eq!(stdout, "3\n3\n");
}

#[test]
fn runs_compiled_lambda_program() {
    let stdout = compile_and_run(
        r#"
        function main(): void {
            id: (int) -> int = function(a: int): int {
                return a;
            };
            print(id(7));
        }
        "#,
    )
    .unwrap();

    assert_eq!(stdout, "7\n");
}

#[test]
fn runs_compiled_closure_program() {
    let stdout = compile_and_run(
        r#"
        function counter(): () -> int {
            c: int = 0;
            return function(): int {
                c = c + 1;
                return c;
            };
        }

        function main(): void {
            count: () -> int = counter();
            print(count());
            print(count());
        }
        "#,
    )
    .unwrap();

    assert_eq!(stdout, "1\n2\n");
}

#[test]
fn runs_compiled_recursive_lambda_program() {
    let stdout = compile_and_run(
        r#"
        function main(): void {
            factorial: (int) -> int = function(n: int): int {
                if (n == 0) {
                    return 1;
                } else {
                    return n * factorial(n - 1);
                }
            };
            print(factorial(5));
        }
        "#,
    )
    .unwrap();

    assert_eq!(stdout, "120\n");
}

#[test]
fn runs_compiled_method_returning_lambda_program() {
    let stdout = compile_and_run(
        r#"
        struct Foo {
            factor: int;
        }

        attach Foo {
            function make(self): (int) -> int {
                return function(i: int): int {
                    return self.factor + i;
                };
            }
        }

        function main(): void {
            foo: Foo = Foo { factor: 4 };
            print(foo.make()(3));
        }
        "#,
    )
    .unwrap();

    assert_eq!(stdout, "7\n");
}

#[test]
fn runs_compiled_pointer_allocator_program() {
    let stdout = compile_and_run(
        r#"
        struct Point {
            x: int;
            y: int;
        }

        attach Point {
            function sum(self): int {
                return self.x + self.y;
            }
        }

        function main(): void {
            heap: Allocator = System::allocator();
            arena: Arena = Arena::init(heap);
            alloc: Allocator = arena.allocator();
            p: *Point = Point::create(alloc);
            p.x = 4;
            p.y = 5;
            print(p.sum());
            arena.deinit();
        }
        "#,
    )
    .unwrap();

    assert_eq!(stdout, "9\n");
}

#[test]
fn runs_compiled_function_returning_pointer_program() {
    let stdout = compile_and_run(
        r#"
        struct Point {
            x: int;
            y: int;
        }

        function create_point(alloc: Allocator): *Point {
            point: *Point = Point::create(alloc);
            point.x = 11;
            point.y = 31;
            return point;
        }

        function main(): void {
            heap: Allocator = System::allocator();
            point: *Point = create_point(heap);
            print(point.x);
            print(point.y);
            heap.destroy(point);
        }
        "#,
    )
    .unwrap();

    assert_eq!(stdout, "11\n31\n");
}

#[test]
fn runs_compiled_generic_struct_and_function_program() {
    let stdout = compile_and_run(
        r#"
        struct Box[T] {
            value: T;
        }

        attach[T] Box[T] {
            function get(self): T {
                return self.value;
            }
        }

        function wrap[T](value: T): Box[T] {
            return Box[T] { value: value };
        }

        function unwrap[T](box: Box[T]): T {
            return box.get();
        }

        function main(): void {
            int_box: Box[int] = wrap(41);
            string_box: Box[string] = wrap("done");
            print(unwrap(int_box) + 1);
            print(unwrap(string_box));
        }
        "#,
    )
    .unwrap();

    assert_eq!(stdout, "42\ndone\n");
}

#[test]
fn runs_compiled_generic_static_attached_function_program() {
    let stdout = compile_and_run(
        r#"
        struct Box[T] {
            value: T;
        }

        attach[T] Box[T] {
            function wrap(value: T): Box[T] {
                return Box[T] { value: value };
            }

            function get(self): T {
                return self.value;
            }
        }

        function main(): void {
            int_box: Box[int] = Box[int]::wrap(41);
            string_box: Box[string] = Box[string]::wrap("ok");
            print(int_box.get() + 1);
            print(string_box.get());
        }
        "#,
    )
    .unwrap();

    assert_eq!(stdout, "42\nok\n");
}

#[test]
fn runs_compiled_concrete_enum_attached_functions() {
    let stdout = compile_and_run(
        r#"
        enum Direction {
            North;
            South;
        }

        attach Direction {
            function is_north(self): bool {
                match (self) {
                    case North: {
                        return true;
                    }
                    case South: {
                        return false;
                    }
                }
            }

            function default_direction(): Direction {
                return Direction::North();
            }
        }

        function main(): void {
            north: Direction = Direction::default_direction();
            south: Direction = Direction::South();
            print(north.is_north());
            print(south.is_north());
        }
        "#,
    )
    .unwrap();

    assert_eq!(stdout, "true\nfalse\n");
}

#[test]
fn runs_compiled_generic_enum_attached_functions() {
    let stdout = compile_and_run(
        r#"
        enum Option[T] {
            Some(T);
            None;
        }

        attach[T] Option[T] {
            function is_some(self): bool {
                match (self) {
                    case Some(value): {
                        return true;
                    }
                    case None: {
                        return false;
                    }
                }
            }

            function unwrap_or(self, fallback: T): T {
                match (self) {
                    case Some(value): {
                        return value;
                    }
                    case None: {
                        return fallback;
                    }
                }
            }

            function empty(): Option[T] {
                return Option::None();
            }

            function replace(mut self, value: T): void {
                self = Option::Some(value);
            }
        }

        function main(): void {
            some: Option[int] = Option::Some(7);
            none: Option[int] = Option[int]::empty();
            text: Option[string] = Option::Some("skunk");
            print(some.is_some());
            print(none.is_some());
            print(some.unwrap_or(99));
            print(none.unwrap_or(99));
            print(text.unwrap_or("fallback"));
            none.replace(11);
            print(none.unwrap_or(99));
        }
        "#,
    )
    .unwrap();

    assert_eq!(stdout, "true\nfalse\n7\n99\nskunk\n11\n");
}

#[test]
fn rejects_mutating_enum_method_on_const_binding() {
    let error = compile_and_run(
        r#"
        enum Option[T] {
            Some(T);
            None;
        }

        attach[T] Option[T] {
            function replace(mut self, value: T): void {
                self = Option::Some(value);
            }
        }

        function main(): void {
            const value: Option[int] = Option::None();
            value.replace(7);
        }
        "#,
    )
    .unwrap_err();

    assert!(error.contains("cannot call mutating method through const or immutable receiver"));
}

#[test]
fn runs_compiled_nested_generic_program() {
    let stdout = compile_and_run(
        r#"
        struct Box[T] {
            value: T;
        }

        function wrap[T](value: T): Box[T] {
            return Box[T] { value: value };
        }

        function main(): void {
            inner: Box[int] = wrap(7);
            outer: Box[Box[int]] = wrap(inner);
            print(outer.value.value);
        }
        "#,
    )
    .unwrap();

    assert_eq!(stdout, "7\n");
}

#[test]
fn runs_compiled_slice_allocator_program() {
    let stdout = compile_and_run(
        r#"
        function main(): void {
            heap: Allocator = System::allocator();
            arena: Arena = Arena::init(heap);
            alloc: Allocator = arena.allocator();
            values: []int = []int::alloc(alloc, 3);
            values[1] = 7;
            print(values.len);
            print(values[1]);
            arena.deinit();
        }
        "#,
    )
    .unwrap();

    assert_eq!(stdout, "3\n7\n");
}

#[test]
fn runs_compiled_allocator_destroy_and_free_program() {
    let stdout = compile_and_run(
        r#"
        struct Point {
            x: int;
        }

        function main(): void {
            heap: Allocator = System::allocator();
            p: *Point = Point::create(heap);
            p.x = 12;
            print(p.x);
            heap.destroy(p);

            values: []int = []int::alloc(heap, 2);
            values[0] = 3;
            print(values[0]);
            heap.free(values);
        }
        "#,
    )
    .unwrap();

    assert_eq!(stdout, "12\n3\n");
}

#[test]
fn runs_compiled_const_slice_copy_program() {
    let stdout = compile_and_run(
        r#"
        function copy_into(const dst: []int, src: []const int): void {
            for (i: int = 0; i < src.len; i = i + 1) {
                dst[i] = src[i];
            }
        }

        function main(): void {
            heap: Allocator = System::allocator();
            dst: []int = []int::alloc(heap, 2);
            src: []int = []int::alloc(heap, 2);
            src[0] = 7;
            src[1] = 11;
            copy_into(dst, src);
            print(dst[0]);
            print(dst[1]);
            heap.free(dst);
            heap.free(src);
        }
        "#,
    )
    .unwrap();

    assert_eq!(stdout, "7\n11\n");
}

#[test]
fn rejects_reassigning_const_variable() {
    let result = compile_and_run(
        r#"
        function main(): void {
            const answer: int = 41;
            answer = 42;
        }
        "#,
    );

    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .contains("cannot assign to const binding `answer`"));
}

#[test]
fn rejects_assigning_to_const_struct_field() {
    let result = compile_and_run(
        r#"
        struct Counter {
            const value: int;
        }

        attach Counter {
            function reset(mut self): void {
                self.value = 0;
            }
        }

        function main(): void {
            counter: Counter = Counter { value: 7 };
            counter.reset();
        }
        "#,
    );

    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .contains("cannot assign through const-qualified target `int`"));
}

#[test]
fn rejects_reassigning_const_parameter() {
    let result = compile_and_run(
        r#"
        function bump(const n: int): int {
            n = n + 1;
            return n;
        }

        function main(): void {
            print(bump(1));
        }
        "#,
    );

    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .contains("cannot assign to const binding `n`"));
}

#[test]
fn rejects_writing_through_const_slice() {
    let result = compile_and_run(
        r#"
        function overwrite(values: []const int): void {
            values[0] = 7;
        }

        function main(): void {
            heap: Allocator = System::allocator();
            values: []int = []int::alloc(heap, 1);
            overwrite(values);
            heap.free(values);
        }
        "#,
    );

    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .contains("cannot assign through const-qualified target `int`"));
}

#[test]
fn rejects_writing_through_const_pointer() {
    let result = compile_and_run(
        r#"
        struct Point {
            x: int;
        }

        function set_x(point: *const Point): void {
            point.x = 9;
        }

        function main(): void {
            heap: Allocator = System::allocator();
            point: *Point = Point::create(heap);
            set_x(point);
            heap.destroy(point);
        }
        "#,
    );

    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .contains("cannot assign through const-qualified target `int`"));
}

#[test]
fn runs_compiled_arena_destroy_and_free_program() {
    let stdout = compile_and_run(
        r#"
        struct Point {
            x: int;
        }

        function main(): void {
            heap: Allocator = System::allocator();
            arena: Arena = Arena::init(heap);
            alloc: Allocator = arena.allocator();

            p: *Point = Point::create(alloc);
            p.x = 2;
            print(p.x);
            alloc.destroy(p);

            values: []int = []int::alloc(alloc, 2);
            values[1] = 8;
            print(values[1]);
            alloc.free(values);

            arena.reset();
            arena.deinit();
        }
        "#,
    )
    .unwrap();

    assert_eq!(stdout, "2\n8\n");
}

#[test]
fn compiles_generic_enum_to_ir() {
    let program = crate::pipeline::check_source(
        "compiler-test.skunk",
        r#"
        enum Option[T] {
            None;
            Some(T);
        }

        function main(): void {
            value: Option[int] = Option[int]::Some(7);
            match (value) {
                case None: {
                    print(0);
                }
                case Some(v): {
                    print(v);
                }
            }
        }
        "#,
    )
    .unwrap();

    let ir = compile_to_llvm_ir(&program).unwrap();
    assert!(ir.contains("%enum.Option__int = type { i32, i32 }"));
    assert!(ir.contains("switch i32"));
}

#[test]
fn runs_compiled_generic_enum_match_program() {
    let stdout = compile_and_run(
        r#"
        enum Option[T] {
            None;
            Some(T);
        }

        function unwrap(value: Option[int]): int {
            match (value) {
                case None: {
                    return 0;
                }
                case Some(v): {
                    return v + 1;
                }
            }
        }

        function main(): void {
            print(unwrap(Option[int]::None()));
            print(unwrap(Option[int]::Some(41)));
        }
        "#,
    )
    .unwrap();

    assert_eq!(stdout, "0\n42\n");
}

#[test]
fn runs_inferred_generic_enum_constructors() {
    let stdout = compile_and_run(
        r#"
        enum Outcome[T, E] {
            Ok(T);
            Err(E);
        }

        enum AppError {
            Failed(string);
        }

        function make_value(ok: bool): Outcome[int, AppError] {
            if (ok) {
                return Outcome::Ok(41);
            }
            return Outcome::Err(AppError::Failed("failed"));
        }

        function consume(value: Outcome[int, AppError]): int {
            match (value) {
                case Ok(number): {
                    return number;
                }
                case Err(error): {
                    return -1;
                }
            }
        }

        function make_wide_value(): Outcome[long, AppError] {
            value: int = 44;
            return Outcome::Ok(value);
        }

        function consume_wide(value: Outcome[long, AppError]): long {
            match (value) {
                case Ok(number): {
                    return number;
                }
                case Err(error): {
                    return -1L;
                }
            }
        }

        function main(): void {
            print(consume(make_value(true)));
            print(consume(Outcome::Ok(42)));
            failure: Outcome[int, AppError] = Outcome::Err(
                AppError::Failed("no value")
            );
            print(consume(failure));
            print(consume_wide(make_wide_value()));
        }
        "#,
    )
    .unwrap();

    assert_eq!(stdout, "41\n42\n-1\n44\n");
}

#[test]
fn runs_compiled_multi_payload_enum_match_program() {
    let stdout = compile_and_run(
        r#"
        enum PairOrNone[A, B] {
            None;
            Pair(A, B);
        }

        function unwrap(value: PairOrNone[int, int]): int {
            match (value) {
                case None: {
                    return 0;
                }
                case Pair(a, b): {
                    return a + b;
                }
            }
        }

        function main(): void {
            print(unwrap(PairOrNone[int, int]::None()));
            print(unwrap(PairOrNone[int, int]::Pair(20, 22)));
        }
        "#,
    )
    .unwrap();

    assert_eq!(stdout, "0\n42\n");
}

#[test]
fn runs_compiled_imported_generic_enum_module_program() {
    let stdout = compile_project_and_run(
        &[
            (
                "mylib/option.skunk",
                r#"
                module mylib.option;

                enum Option[T] {
                    None;
                    Some(T);
                }

                function wrap[T](value: T): Option[T] {
                    return Option[T]::Some(value);
                }
                "#,
            ),
            (
                "main.skunk",
                r#"
                import mylib.option;

                function main(): void {
                    value: Option[int] = wrap(9);
                    match (value) {
                        case None: {
                            print(0);
                        }
                        case Some(v): {
                            print(v);
                        }
                    }
                }
                "#,
            ),
        ],
        "main.skunk",
    )
    .unwrap();

    assert_eq!(stdout, "9\n");
}

#[test]
fn runs_compiled_trait_bound_program() {
    let stdout = compile_and_run(
        r#"
        trait Writer {
            function write(mut self, value: int): int;
        }

        trait Resettable {
            function reset(mut self): void;
        }

        struct Counter {
            value: int;
        }

        conform Writer for Counter {
            function write(mut self, value: int): int {
                self.value = self.value + value;
                return self.value;
            }
        }

        conform Resettable for Counter {
            function reset(mut self): void {
                self.value = 0;
            }
        }

        function use_counter[T: Writer & Resettable](counter: *T): int {
            counter.reset();
            return counter.write(41);
        }

        function main(): void {
            heap: Allocator = System::allocator();
            counter: *Counter = Counter::create(heap);
            counter.value = 9;
            print(use_counter(counter));
            heap.destroy(counter);
        }
        "#,
    )
    .unwrap();

    assert_eq!(stdout, "41\n");
}

#[test]
fn runs_compiled_where_clause_trait_bound_program() {
    let stdout = compile_and_run(
        r#"
        trait Writer {
            function write(mut self, value: int): int;
        }

        trait Resettable {
            function reset(mut self): void;
        }

        struct Counter {
            value: int;
        }

        conform Writer for Counter {
            function write(mut self, value: int): int {
                self.value = self.value + value;
                return self.value;
            }
        }

        conform Resettable for Counter {
            function reset(mut self): void {
                self.value = 0;
            }
        }

        function use_counter[T](counter: *T): int
        where T: Writer & Resettable {
            counter.reset();
            return counter.write(41);
        }

        function main(): void {
            heap: Allocator = System::allocator();
            counter: *Counter = Counter::create(heap);
            counter.value = 9;
            print(use_counter(counter));
            heap.destroy(counter);
        }
        "#,
    )
    .unwrap();

    assert_eq!(stdout, "41\n");
}

#[test]
fn runs_compiled_shape_bound_program() {
    let stdout = compile_and_run(
        r#"
        shape WriterLike {
            function write(mut self, value: int): int;
        }

        struct BufferWriter {
            value: int;
        }

        attach BufferWriter {
            function write(mut self, value: int): int {
                self.value = self.value + value;
                return self.value;
            }
        }

        function use_writer_like[T: WriterLike](writer: *T): int {
            return writer.write(5);
        }

        function main(): void {
            heap: Allocator = System::allocator();
            writer: *BufferWriter = BufferWriter::create(heap);
            writer.value = 7;
            print(use_writer_like(writer));
            heap.destroy(writer);
        }
        "#,
    )
    .unwrap();

    assert_eq!(stdout, "12\n");
}

#[test]
fn runs_compiled_trait_object_dynamic_dispatch_program() {
    let stdout = compile_and_run(
        r#"
        trait Writer {
            function write(mut self, value: int): int;
        }

        struct Counter {
            value: int;
        }

        conform Writer for Counter {
            function write(mut self, value: int): int {
                self.value = self.value + value;
                return self.value;
            }
        }

        function main(): void {
            writer: Writer = Counter { value: 1 };
            print(writer.write(4));
            print(writer.write(7));
        }
        "#,
    )
    .unwrap();

    assert_eq!(stdout, "5\n12\n");
}

#[test]
fn runs_compiled_trait_object_borrowed_local_program() {
    let stdout = compile_and_run(
        r#"
        trait Writer {
            function write(mut self, value: int): int;
        }

        struct IntWriter {
            i: int;
        }

        attach IntWriter {
            function get_i(self): int {
                return self.i;
            }
        }

        conform Writer for IntWriter {
            function write(mut self, value: int): int {
                self.i = value;
                return self.i;
            }
        }

        function main(): void {
            iw: IntWriter = IntWriter { i: 0 };
            w: Writer = iw;
            w.write(1);
            print(iw.get_i());
            print(iw.i);
        }
        "#,
    )
    .unwrap();

    assert_eq!(stdout, "1\n1\n");
}

#[test]
fn runs_compiled_trait_object_parameter_program() {
    let stdout = compile_and_run(
        r#"
        trait Writer {
            function write(mut self, value: int): int;
        }

        struct Counter {
            value: int;
        }

        conform Writer for Counter {
            function write(mut self, value: int): int {
                self.value = self.value + value;
                return self.value;
            }
        }

        function use_writer(writer: Writer): int {
            return writer.write(9);
        }

        function main(): void {
            writer: Writer = Counter { value: 3 };
            print(use_writer(writer));
            print(writer.write(1));
        }
        "#,
    )
    .unwrap();

    assert_eq!(stdout, "12\n13\n");
}

#[test]
fn runs_compiled_safe_reference_program() {
    let stdout = compile_and_run(
        r#"
        struct Counter {
            value: int;
        }

        attach Counter {
            function get(self): int {
                return self.value;
            }
        }

        function bump(counter: &mut Counter): void {
            counter.value = counter.value + 1;
        }

        function read(counter: &Counter): int {
            return counter.get();
        }

        function main(): void {
            counter: Counter = Counter { value: 4 };
            bump(&mut counter);

            reader: &Counter = &counter;
            print(read(reader));
            print(reader.*.value);
        }
        "#,
    )
    .unwrap();

    assert_eq!(stdout, "5\n5\n");
}

#[test]
fn rejects_mutating_through_immutable_reference() {
    let result = compile_and_run(
        r#"
        struct Counter {
            value: int;
        }

        function bump(counter: &Counter): void {
            counter.value = counter.value + 1;
        }

        function main(): void {
            counter: Counter = Counter { value: 4 };
            bump(&counter);
        }
        "#,
    );

    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .contains("cannot assign through const-qualified target"));
}

#[test]
fn runs_compiled_trait_default_method_program() {
    let stdout = compile_and_run(
        r#"
        trait Writer {
            function write(mut self, value: int): int;

            function write_twice(mut self, value: int): int {
                self.write(value);
                return self.write(value);
            }
        }

        struct Counter {
            value: int;
        }

        conform Writer for Counter {
            function write(mut self, value: int): int {
                self.value = self.value + value;
                return self.value;
            }
        }

        function use_counter[T: Writer](counter: *T): int {
            return counter.write_twice(3);
        }

        function main(): void {
            heap: Allocator = System::allocator();
            counter: *Counter = Counter::create(heap);
            counter.value = 1;
            print(use_counter(counter));

            writer: Writer = Counter { value: 10 };
            print(writer.write_twice(2));

            heap.destroy(counter);
        }
        "#,
    )
    .unwrap();

    assert_eq!(stdout, "7\n14\n");
}

#[test]
fn runs_compiled_supertrait_program() {
    let stdout = compile_and_run(
        r#"
        trait Readable {
            function value(self): int;
        }

        trait Writer: Readable {
            function write(mut self, value: int): int;
        }

        struct Counter {
            value: int;
        }

        conform Writer for Counter {
            function value(self): int {
                return self.value;
            }

            function write(mut self, value: int): int {
                self.value = self.value + value;
                return self.value;
            }
        }

        function read_once[T: Readable](counter: *T): int {
            return counter.value();
        }

        function main(): void {
            heap: Allocator = System::allocator();
            counter: *Counter = Counter::create(heap);
            counter.value = 5;
            print(read_once(counter));

            readable: Readable = Counter { value: 9 };
            print(readable.value());

            heap.destroy(counter);
        }
        "#,
    )
    .unwrap();

    assert_eq!(stdout, "5\n9\n");
}

#[test]
fn runs_compiled_function_returning_trait_object_program() {
    let stdout = compile_and_run(
        r#"
        trait Writer {
            function write(mut self, value: int): int;
        }

        struct Counter {
            value: int;
        }

        conform Writer for Counter {
            function write(mut self, value: int): int {
                self.value = self.value + value;
                return self.value;
            }
        }

        function create_writer(): Writer {
            return Counter { value: 10 };
        }

        function main(): void {
            writer: Writer = create_writer();
            print(writer.write(2));
            print(writer.write(5));
        }
        "#,
    )
    .unwrap();

    assert_eq!(stdout, "12\n17\n");
}

#[test]
fn rejects_trait_object_assignment_for_non_impl_type() {
    let result = compile_and_run(
        r#"
        trait Writer {
            function write(mut self, value: int): int;
        }

        struct Counter {
            value: int;
        }

        conform Writer for Counter {
            function write(mut self, value: int): int {
                self.value = self.value + value;
                return self.value;
            }
        }

        struct Plain {
            value: int;
        }

        function main(): void {
            writer: Writer = Plain { value: 1 };
            print(writer.write(1));
        }
        "#,
    );

    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Writer"));
}

#[test]
fn runs_compiled_unsafe_address_of_and_dereference_program() {
    let stdout = compile_and_run(
        r#"
        function main(): void {
            value: int = 41;
            unsafe {
                ptr: *int = &value;
                print(ptr.*);
                ptr.* = 42;
            }
            print(value);
        }
        "#,
    )
    .unwrap();

    assert_eq!(stdout, "41\n42\n");
}

#[test]
fn runs_compiled_size_of_and_align_of_program() {
    let stdout = compile_and_run(
        r#"
        struct Pair {
            left: int;
            right: int;
        }

        function main(): void {
            print(int::size_of());
            print(int::align_of());
            print(Pair::size_of());
            print(Pair::align_of());
        }
        "#,
    )
    .unwrap();

    assert_eq!(stdout, "4\n4\n8\n4\n");
}

#[test]
fn runs_compiled_unsafe_memory_copy_and_set_program() {
    let stdout = compile_and_run(
        r#"
        function main(): void {
            src: int = 123;
            dst: int = 0;
            bytes: [4]byte;
            unsafe {
                Memory::copy(*byte::cast(&dst), *byte::cast(&src), int::size_of());
                Memory::set(&bytes[0], 7, 4);
            }
            print(dst);
            print(bytes[0]);
            print(bytes[3]);
        }
        "#,
    )
    .unwrap();

    assert_eq!(stdout, "123\n7\n7\n");
}

// The window runtime is macOS-only (skunk_window_runtime.m + Cocoa), so
// programs calling Window/Keyboard APIs cannot link on other platforms.
#[cfg(target_os = "macos")]
#[test]
fn runs_compiled_headless_window_program() {
    let stdout = compile_and_run_with_env(
        r#"
        function main(): void {
            window: Window = Window::create(96, 64, "Headless");
            print(window.is_open());
            window.poll();
            window.clear(Color::black());
            window.draw_rect(4.0, 6.0, 18.0, 10.0, Color::white());
            window.present();
            print(Keyboard::is_down(window, 'w'));
            print(window.delta_time() > 0.0);
            window.close();
            print(window.is_open());
            window.deinit();
        }
        "#,
        &[("SKUNK_WINDOW_HEADLESS", "1")],
    )
    .unwrap();

    assert_eq!(stdout, "true\nfalse\ntrue\nfalse\n");
}

#[cfg(target_os = "macos")]
#[test]
fn runs_headless_pong_example() {
    // Read at runtime (not include_str!) so a missing example file fails
    // this test with a clear message instead of breaking the whole build.
    let pong_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/pong.skunk");
    let pong_source = fs::read_to_string(&pong_path)
        .unwrap_or_else(|err| panic!("failed to read `{}`: {}", pong_path.display(), err));
    let stdout = compile_and_run_with_env(&pong_source, &[("SKUNK_WINDOW_HEADLESS", "1")]).unwrap();

    assert_eq!(stdout, "0\n5\n");
}

#[test]
fn runs_compiled_unsafe_byte_pointer_offset_program() {
    let stdout = compile_and_run(
        r#"
        function main(): void {
            bytes: [4]byte;
            unsafe {
                start: *byte = &bytes[0];
                second: *byte = *byte::offset(start, 1);
                second.* = 9;
            }
            print(bytes[1]);
        }
        "#,
    )
    .unwrap();

    assert_eq!(stdout, "9\n");
}

#[test]
fn rejects_address_of_outside_unsafe_block() {
    let result = compile_and_run(
        r#"
        function main(): void {
            value: int = 1;
            ptr: *int = &value;
            print(ptr.*);
        }
        "#,
    );

    assert!(result.is_err());
    assert!(result.unwrap_err().contains("address-of"));
}

#[test]
fn rejects_memory_copy_outside_unsafe_block() {
    let result = compile_and_run(
        r#"
        function main(): void {
            src: int = 1;
            dst: int = 0;
            Memory::copy(*byte::cast(&dst), *byte::cast(&src), int::size_of());
        }
        "#,
    );

    assert!(result.is_err());
    assert!(result.unwrap_err().contains("unsafe"));
}

#[test]
fn runs_compiled_generic_impl_target_program() {
    let stdout = compile_and_run(
        r#"
        trait SizedThing {
            function size(self): int;
        }

        struct Box[T] {
            value: T;
        }

        conform[T] SizedThing for Box[T] {
            function size(self): int {
                return 1;
            }
        }

        function measure[T: SizedThing](value: T): int {
            return value.size();
        }

        function main(): void {
            print(measure(Box[int] { value: 7 }));
            print(measure(Box[string] { value: "x" }));
        }
        "#,
    )
    .unwrap();

    assert_eq!(stdout, "1\n1\n");
}

#[test]
fn runs_compiled_generic_trait_with_implicit_conformance_binder() {
    let stdout = compile_and_run(
        r#"
        trait Cell[T] {
            function get(self): T;
            function set(mut self, value: T): void;
        }

        struct Box[T] {
            value: T;
        }

        conform Cell[T] for Box[T] {
            function get(self): T {
                return self.value;
            }

            function set(mut self, value: T): void {
                self.value = value;
            }
        }

        function main(): void {
            number: Cell[int] = Box[int] { value: 7 };
            print(number.get());
            number.set(42);
            print(number.get());

            word: Cell[string] = Box[string] { value: "old" };
            print(word.get());
            word.set("new");
            print(word.get());
        }
        "#,
    )
    .unwrap();

    assert_eq!(stdout, "7\n42\nold\nnew\n");
}

#[test]
fn rejects_generic_trait_receiver_mutability_mismatch() {
    let result = compile_and_run(
        r#"
        trait Cell[T] {
            function set(mut self, value: T): void;
        }

        struct Box[T] { value: T; }

        conform Cell[T] for Box[T] {
            function set(self, value: T): void {}
        }

        function main(): void {}
        "#,
    );

    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .contains("trait method `Cell[T].set` expects"));
}

#[test]
fn enforces_generic_trait_bounds_when_specialized() {
    let result = compile_and_run(
        r#"
        trait Printable {
            function print_value(self): void;
        }

        trait PrintableCell[T: Printable] {
            function get(self): T;
        }

        struct Plain { value: int; }
        struct Box[T] { value: T; }

        conform PrintableCell[T] for Box[T] {
            function get(self): T { return self.value; }
        }

        function main(): void {
            cell: PrintableCell[Plain] = Box[Plain] {
                value: Plain { value: 1 }
            };
        }
        "#,
    );

    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .contains("generic trait `PrintableCell` requires `T` to implement trait `Printable`"));
}

#[test]
fn runs_compiled_explicit_generic_function_call_program() {
    let stdout = compile_and_run(
        r#"
        function id[T](value: T): T {
            return value;
        }

        function main(): void {
            print(id[int](42));
        }
        "#,
    )
    .unwrap();

    assert_eq!(stdout, "42\n");
}

#[test]
fn rejects_trait_impl_with_receiver_mutability_mismatch() {
    let result = compile_and_run(
        r#"
        trait Writer {
            function write(mut self, value: int): int;
        }

        struct Counter {
            value: int;
        }

        conform Writer for Counter {
            function write(self, value: int): int {
                return self.value + value;
            }
        }

        function main(): void {}
        "#,
    );

    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .contains("trait method `Writer.write` expects"));
}

#[test]
fn rejects_call_when_trait_bound_is_not_implemented() {
    let result = compile_and_run(
        r#"
        trait Writer {
            function write(mut self, value: int): int;
        }

        struct Counter {
            value: int;
        }

        attach Counter {
            function write(mut self, value: int): int {
                self.value = self.value + value;
                return self.value;
            }
        }

        function use_counter[T: Writer](counter: *T): int {
            return counter.write(41);
        }

        function main(): void {
            heap: Allocator = System::allocator();
            counter: *Counter = Counter::create(heap);
            print(use_counter(counter));
            heap.destroy(counter);
        }
        "#,
    );

    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .contains("generic function `use_counter` requires `T` to implement trait `Writer`"));
}

#[test]
fn runs_compiled_imported_module_with_exported_api_and_private_helper_program() {
    let stdout = compile_project_and_run(
        &[
            (
                "mylib/math.skunk",
                r#"
                module mylib.math;

                function helper(n: int): int {
                    return n + 1;
                }

                export function inc(n: int): int {
                    return helper(n);
                }
                "#,
            ),
            (
                "main.skunk",
                r#"
                import mylib.math;

                function main(): void {
                    print(inc(41));
                }
                "#,
            ),
        ],
        "main.skunk",
    )
    .unwrap();

    assert_eq!(stdout, "42\n");
}

#[test]
fn runs_compiled_imported_generic_trait_program() {
    let stdout = compile_project_and_run(
        &[
            (
                "containers/cell.skunk",
                r#"
                module containers.cell;

                export trait Cell[T] {
                    function get(self): T;
                }
                "#,
            ),
            (
                "main.skunk",
                r#"
                import containers.cell;

                struct Box[T] {
                    value: T;
                }

                conform Cell[T] for Box[T] {
                    function get(self): T { return self.value; }
                }

                function main(): void {
                    cell: Cell[int] = Box[int] { value: 42 };
                    print(cell.get());
                }
                "#,
            ),
        ],
        "main.skunk",
    )
    .unwrap();

    assert_eq!(stdout, "42\n");
}

#[test]
fn rejects_private_imported_module_symbols() {
    let result = compile_project_and_run(
        &[
            (
                "mylib/math.skunk",
                r#"
                module mylib.math;

                function helper(n: int): int {
                    return n + 1;
                }

                export function inc(n: int): int {
                    return helper(n);
                }
                "#,
            ),
            (
                "main.skunk",
                r#"
                import mylib.math;

                function main(): void {
                    print(helper(41));
                }
                "#,
            ),
        ],
        "main.skunk",
    );

    assert!(result.is_err());
    assert!(result.unwrap_err().contains("unknown function `helper`"));
}

#[test]
fn runs_compiled_imported_module_program() {
    let stdout = compile_project_and_run(
        &[
            (
                "mylib/math.skunk",
                r#"
                module mylib.math;

                function inc(n: int): int {
                    return n + 1;
                }
                "#,
            ),
            (
                "main.skunk",
                r#"
                import mylib.math;

                function main(): void {
                    print(inc(41));
                }
                "#,
            ),
        ],
        "main.skunk",
    )
    .unwrap();

    assert_eq!(stdout, "42\n");
}

#[test]
fn runs_compiled_imported_generic_module_program() {
    let stdout = compile_project_and_run(
        &[
            (
                "mylib/box.skunk",
                r#"
                module mylib.box;

                struct Box[T] {
                    value: T;
                }

                function wrap[T](value: T): Box[T] {
                    return Box[T] { value: value };
                }
                "#,
            ),
            (
                "main.skunk",
                r#"
                import mylib.box;

                function main(): void {
                    value: Box[int] = wrap(7);
                    print(value.value);
                }
                "#,
            ),
        ],
        "main.skunk",
    )
    .unwrap();

    assert_eq!(stdout, "7\n");
}

#[test]
fn runs_compiled_exported_type_alias_program() {
    let stdout = compile_project_and_run(
        &[
            (
                "types.skunk",
                r#"
                module types;
                export type Value = string | int;
                "#,
            ),
            (
                "main.skunk",
                r#"
                import types;

                function consume(value: Value): void {
                    print(7);
                }

                function main(): void {
                    consume("exported");
                }
                "#,
            ),
        ],
        "main.skunk",
    )
    .unwrap();
    assert_eq!(stdout, "7\n");
}

#[test]
fn runs_compiled_union_and_generic_alias_program() {
    let stdout = compile_and_run(
        r#"
        type Value = string | int;
        type Either[T] = T | string;

        function forward(value: Value): Either[int] {
            return value;
        }

        function main(): void {
            first: Value = 7;
            second: Value = "skunk";
            a: Either[int] = forward(first);
            b: Either[int] = forward(second);
            print(42);
        }
        "#,
    )
    .unwrap();
    assert_eq!(stdout, "42\n");
}

#[test]
fn runs_compiled_generic_subtype_bounds_program() {
    let stdout = compile_and_run(
        r#"
        trait Animal {}

        struct Dog {}
        struct Cat {}

        conform Animal for Dog {}
        conform Animal for Cat {}

        function choose[T <: Animal](left: T, right: T, first: bool): T {
            if (first) {
                return left;
            }
            return right;
        }

        function widen[A, B >: A <: Animal](left: A, right: B): B {
            return left;
        }

        function main(): void {
            first: Dog | Cat = choose(Dog {}, Cat {}, true);
            second: Dog | Cat = widen(Dog {}, Cat {});
            print(42);
        }
        "#,
    )
    .unwrap();
    assert_eq!(stdout, "42\n");
}

#[test]
fn runs_compiled_trait_intersection_alias_program() {
    let stdout = compile_and_run(
        r#"
        trait Writer {
            function write(mut self, value: int): int;
        }

        trait Resettable {
            function reset(mut self): void;
        }

        type Service = Writer & Resettable;

        struct Counter {
            value: int;
        }

        conform Writer for Counter {
            function write(mut self, value: int): int {
                self.value = self.value + value;
                return self.value;
            }
        }

        conform Resettable for Counter {
            function reset(mut self): void {
                self.value = 0;
            }
        }

        function use_service(service: Service): int {
            service.reset();
            return service.write(41);
        }

        function main(): void {
            service: Service = Counter { value: 9 };
            print(use_service(service));
        }
        "#,
    )
    .unwrap();
    assert_eq!(stdout, "41\n");
}

#[test]
fn extern_declaration_emits_unmangled_declare_and_call() {
    let program = crate::pipeline::check_source(
        "compiler-test.skunk",
        r#"
        extern "C" function cos(value: double): double;

        function main(): void {
            print(cos(0.0));
        }
        "#,
    )
    .unwrap();
    let ir = compile_to_llvm_ir(&program).unwrap();

    assert!(
        ir.contains("declare double @cos(double)"),
        "missing extern declare in IR:\n{}",
        ir
    );
    assert!(
        ir.contains("call double @cos("),
        "extern call should use the unmangled symbol:\n{}",
        ir
    );
}

#[test]
fn extern_declaration_runs_natively() {
    let stdout = compile_and_run(
        r#"
        extern "C" function cos(value: double): double;

        function main(): void {
            print(cos(0.0));
        }
        "#,
    )
    .unwrap();
    assert_eq!(stdout.trim(), "1.000000");
}

#[test]
fn extern_declaration_rejects_reserved_runtime_symbols() {
    let program = crate::pipeline::check_source(
        "compiler-test.skunk",
        r#"
        extern "C" function malloc(size: long): *byte;

        function main(): void {
            print(1);
        }
        "#,
    )
    .unwrap();
    let error = compile_to_llvm_ir(&program).unwrap_err();
    assert!(
        error.contains("reserved runtime symbol"),
        "unexpected error: {}",
        error
    );
}

#[test]
fn extern_declaration_rejects_non_abi_safe_types() {
    let error = crate::pipeline::check_source(
        "compiler-test.skunk",
        r#"
        struct Point { x: int; y: int; }

        extern "C" function takes_struct(point: Point): void;

        function main(): void {
            print(1);
        }
        "#,
    )
    .unwrap_err();
    assert!(
        error.contains("non C-ABI-safe"),
        "unexpected error: {}",
        error
    );
}

#[test]
fn native_test_runner_reports_results() {
    let program = loaded_source(
        r#"
        function add(a: int, b: int): int {
            return a + b;
        }

        test "addition" {
            Testing::expect(add(2, 2) == 4);
            Testing::expect_eq(4, add(2, 2));
        }

        test "more addition" {
            Testing::expect(add(1, 2) == 3);
        }
        "#,
    );
    let (test_program, count) = crate::testing::build_test_program(program, None).unwrap();
    assert_eq!(count, 2);
    let test_program = crate::pipeline::check_loaded(test_program).unwrap();

    let id = Uuid::new_v4().to_string();
    let source_path = env::temp_dir().join(format!("skunk_test_runner_{}.skunk", id));
    let output_path = env::temp_dir().join(format!("skunk_test_runner_{}", id));
    fs::write(&source_path, "generated").unwrap();
    let artifact = compile_to_executable(&test_program, &source_path, &output_path).unwrap();
    let output = Command::new(&artifact.binary_path).output().unwrap();
    let _ = fs::remove_file(&source_path);
    let _ = fs::remove_file(&artifact.llvm_ir_path);
    let _ = fs::remove_file(&artifact.binary_path);

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    assert!(output.status.success(), "test runner failed: {}", stdout);
    assert!(stdout.contains("PASS addition"), "stdout: {}", stdout);
    assert!(stdout.contains("PASS more addition"), "stdout: {}", stdout);
    assert!(
        stdout.contains("2 tests, 2 passed, 0 failed"),
        "stdout: {}",
        stdout
    );
}

#[test]
fn native_test_runner_fails_with_nonzero_exit() {
    let program = loaded_source(
        r#"
        test "broken" {
            Testing::expect_eq(1, 2);
        }
        "#,
    );
    let (test_program, _) = crate::testing::build_test_program(program, None).unwrap();
    let test_program = crate::pipeline::check_loaded(test_program).unwrap();

    let id = Uuid::new_v4().to_string();
    let source_path = env::temp_dir().join(format!("skunk_test_runner_{}.skunk", id));
    let output_path = env::temp_dir().join(format!("skunk_test_runner_{}", id));
    fs::write(&source_path, "generated").unwrap();
    let artifact = compile_to_executable(&test_program, &source_path, &output_path).unwrap();
    let output = Command::new(&artifact.binary_path).output().unwrap();
    let _ = fs::remove_file(&source_path);
    let _ = fs::remove_file(&artifact.llvm_ir_path);
    let _ = fs::remove_file(&artifact.binary_path);

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    assert!(!output.status.success(), "runner should fail: {}", stdout);
    assert!(stdout.contains("FAIL broken"), "stdout: {}", stdout);
    assert!(stdout.contains("expected 1, got 2"), "stdout: {}", stdout);
}

#[test]
fn std_math_resolves_from_embedded_sdk() {
    let stdout = compile_project_and_run(
        &[(
            "main.skunk",
            r#"
            import std.math;

            function main(): void {
                print(sqrt(16.0));
                print(max(3, 9));
                print(abs(0 - 7));
            }
            "#,
        )],
        "main.skunk",
    )
    .unwrap();
    assert_eq!(stdout, "4.000000\n9\n7\n");
}

#[test]
fn project_module_can_import_std_math() {
    let stdout = compile_project_and_run(
        &[
            (
                "main.skunk",
                r#"
                import calc.ops;

                function main(): void {
                    print(hypotenuse(3.0, 4.0));
                }
                "#,
            ),
            (
                "calc/ops.skunk",
                r#"
                module calc.ops;

                import std.math;

                export function hypotenuse(a: double, b: double): double {
                    return sqrt(a * a + b * b);
                }
                "#,
            ),
        ],
        "main.skunk",
    )
    .unwrap();
    assert_eq!(stdout, "5.000000\n");
}

#[test]
fn project_module_preserves_private_generic_enum_attach() {
    let stdout = compile_project_and_run(
        &[
            (
                "main.skunk",
                r#"
                import maybe.ops;

                function main(): void {
                    print(present_or(99));
                    print(missing_or(99));
                }
                "#,
            ),
            (
                "maybe/ops.skunk",
                r#"
                module maybe.ops;

                enum Maybe[T] {
                    Present(T);
                    Missing;
                }

                attach[T] Maybe[T] {
                    function unwrap_or(self, fallback: T): T {
                        match (self) {
                            case Present(value): {
                                return value;
                            }
                            case Missing: {
                                return fallback;
                            }
                        }
                    }
                }

                export function present_or(fallback: int): int {
                    value: Maybe[int] = Maybe::Present(7);
                    return value.unwrap_or(fallback);
                }

                export function missing_or(fallback: int): int {
                    value: Maybe[int] = Maybe::Missing();
                    return value.unwrap_or(fallback);
                }
                "#,
            ),
        ],
        "main.skunk",
    )
    .unwrap();

    assert_eq!(stdout, "7\n99\n");
}

#[test]
fn test_declarations_are_ignored_outside_skunk_test() {
    let stdout = compile_and_run(
        r#"
        function main(): void {
            print(7);
        }

        test "not compiled in normal builds" {
            Testing::expect(true);
        }
        "#,
    )
    .unwrap();
    assert_eq!(stdout, "7\n");
}
