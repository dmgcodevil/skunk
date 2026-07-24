//! Native test support for `skunk test`.
//!
//! A loaded program may contain `test "name" { ... }` declarations. This
//! module rewrites such a program into an ordinary executable program:
//!
//! - each test block becomes a plain function `__skunk_test_<i>`,
//! - any user `main` is dropped,
//! - a testing prelude (extern runtime hooks plus the `Testing` API) is
//!   prepended,
//! - a generated `main` runs every selected test between
//!   `skunk_test_begin`/`skunk_test_end` and exits with
//!   `skunk_test_summary()` (non-zero when any test failed).
//!
//! The assertion helpers are ordinary Skunk code layered on extern "C"
//! hooks implemented in `runtime/skunk_runtime.c`, so the test harness
//! dogfoods the C interop path.

use crate::ast::{self, Node, Type};

/// Skunk source prepended to every test program. Declares the C hooks and
/// the user-facing `Testing` API (`Testing::expect(...)`, etc.).
pub const TESTING_PRELUDE: &str = r#"
extern "C" function skunk_test_begin(name: string): void;
extern "C" function skunk_test_end(): void;
extern "C" function skunk_test_summary(): int;
extern "C" function skunk_test_expect(condition: bool): void;
extern "C" function skunk_test_expect_eq_int(expected: int, actual: int): void;
extern "C" function skunk_test_fail(): void;

struct Testing {
    _reserved: int;
}

attach Testing {
    function expect(condition: bool): void {
        skunk_test_expect(condition);
    }

    function expect_eq(expected: int, actual: int): void {
        skunk_test_expect_eq_int(expected, actual);
    }

    function fail(): void {
        skunk_test_fail();
    }
}
"#;

/// Escapes a test name so it can be embedded in a generated string literal.
fn escape_string_literal(value: &str) -> String {
    let mut output = String::with_capacity(value.len());
    for ch in value.chars() {
        match ch {
            '\\' => output.push_str("\\\\"),
            '"' => output.push_str("\\\""),
            '\n' => output.push_str("\\n"),
            '\r' => output.push_str("\\r"),
            '\t' => output.push_str("\\t"),
            '\0' => output.push_str("\\0"),
            other => output.push(other),
        }
    }
    output
}

/// Rewrites a loaded program into a native test runner program.
///
/// Returns the runner program and the number of selected tests. `filter`
/// keeps only tests whose name contains the given substring.
pub fn build_test_program(program: &Node, filter: Option<&str>) -> Result<(Node, usize), String> {
    let Node::Program { statements } = program else {
        return Err("expected a program root when building tests".to_string());
    };

    let mut output = Vec::<Node>::new();

    // The prelude comes first so extern hooks and `Testing` are declared
    // before any test body that uses them.
    let prelude = ast::try_parse(TESTING_PRELUDE)
        .map_err(|err| format!("internal error: testing prelude failed to parse: {}", err))?;
    let Node::Program {
        statements: prelude_statements,
    } = prelude
    else {
        return Err("internal error: testing prelude is not a program".to_string());
    };
    for statement in prelude_statements {
        if !matches!(statement, Node::EOI) {
            output.push(statement);
        }
    }

    let mut tests = Vec::<(String, String)>::new();
    for statement in statements {
        match statement {
            Node::TestDeclaration { name, body } => {
                if let Some(filter) = filter {
                    if !name.contains(filter) {
                        continue;
                    }
                }
                let function_name = format!("__skunk_test_{}", tests.len());
                tests.push((function_name.clone(), name.clone()));
                output.push(Node::FunctionDeclaration {
                    name: function_name,
                    parameters: Vec::new(),
                    return_type: Type::Void,
                    body: body.clone(),
                    lambda: false,
                });
            }
            // The test binary provides its own entry point.
            Node::FunctionDeclaration {
                name,
                lambda: false,
                ..
            } if name == "main" => {}
            Node::EOI => {}
            other => output.push(other.clone()),
        }
    }

    if tests.is_empty() {
        return Err(match filter {
            Some(filter) => format!("no tests match filter `{}`", filter),
            None => "no tests found".to_string(),
        });
    }

    // Generate the runner `main` as source text and parse it, so the AST it
    // produces always matches what the parser would build.
    let mut main_source = String::from("function main(): int {\n");
    for (function_name, display_name) in &tests {
        main_source.push_str(&format!(
            "    skunk_test_begin(\"{}\");\n",
            escape_string_literal(display_name)
        ));
        main_source.push_str(&format!("    {}();\n", function_name));
        main_source.push_str("    skunk_test_end();\n");
    }
    main_source.push_str("    return skunk_test_summary();\n}\n");

    let runner = ast::try_parse(&main_source).map_err(|err| {
        format!(
            "internal error: generated test runner failed to parse: {}",
            err
        )
    })?;
    let Node::Program {
        statements: runner_statements,
    } = runner
    else {
        return Err("internal error: generated test runner is not a program".to_string());
    };
    for statement in runner_statements {
        if !matches!(statement, Node::EOI) {
            output.push(statement);
        }
    }

    output.push(Node::EOI);
    Ok((Node::Program { statements: output }, tests.len()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_program(source: &str) -> Node {
        ast::try_parse(source).expect("test source parses")
    }

    fn program_statements(program: &Node) -> &[Node] {
        match program {
            Node::Program { statements } => statements,
            _ => panic!("expected program"),
        }
    }

    #[test]
    fn converts_test_blocks_into_functions_and_runner_main() {
        let program = parse_program(
            r#"
            test "addition works" {
                Testing::expect(1 + 1 == 2);
            }

            test "subtraction works" {
                Testing::expect_eq(1, 2 - 1);
            }
            "#,
        );

        let (rewritten, count) = build_test_program(&program, None).unwrap();
        assert_eq!(count, 2);

        let statements = program_statements(&rewritten);
        let function_names: Vec<&str> = statements
            .iter()
            .filter_map(|statement| match statement {
                Node::FunctionDeclaration { name, .. } => Some(name.as_str()),
                _ => None,
            })
            .collect();
        assert!(function_names.contains(&"__skunk_test_0"));
        assert!(function_names.contains(&"__skunk_test_1"));
        assert!(function_names.contains(&"main"));

        let extern_names: Vec<&str> = statements
            .iter()
            .filter_map(|statement| match statement {
                Node::ExternFunctionDeclaration { name, .. } => Some(name.as_str()),
                _ => None,
            })
            .collect();
        assert!(extern_names.contains(&"skunk_test_begin"));
        assert!(extern_names.contains(&"skunk_test_expect"));
    }

    #[test]
    fn filter_selects_matching_tests() {
        let program = parse_program(
            r#"
            test "math addition" {
                Testing::expect(true);
            }

            test "strings" {
                Testing::expect(true);
            }
            "#,
        );

        let (_, count) = build_test_program(&program, Some("math")).unwrap();
        assert_eq!(count, 1);
    }

    #[test]
    fn filter_with_no_matches_errors() {
        let program = parse_program(
            r#"
            test "math addition" {
                Testing::expect(true);
            }
            "#,
        );

        assert!(build_test_program(&program, Some("nope")).is_err());
    }

    #[test]
    fn user_main_is_dropped() {
        let program = parse_program(
            r#"
            function main(): void {
                print(1);
            }

            test "one" {
                Testing::expect(true);
            }
            "#,
        );

        let (rewritten, _) = build_test_program(&program, None).unwrap();
        let mains = program_statements(&rewritten)
            .iter()
            .filter(|statement| {
                matches!(statement, Node::FunctionDeclaration { name, .. } if name == "main")
            })
            .count();
        assert_eq!(mains, 1, "only the generated runner main should remain");
    }

    #[test]
    fn programs_without_tests_error() {
        let program = parse_program("function main(): void { print(1); }\n");
        assert!(build_test_program(&program, None).is_err());
    }
}
