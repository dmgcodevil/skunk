//! Native test-runner construction for `skunk test`.
//!
//! Test declarations are transformed within the canonical syntax AST. The
//! runtime prelude and generated entry point are parsed like ordinary source,
//! so test compilation follows the same front-end path as user programs.

use crate::syntax::ast::{
    BuiltinType, FunctionDecl, TopLevelKind, TypeSyntax, TypeSyntaxKind, Visibility,
};
use crate::syntax::loader::LoadedProgram;

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

pub fn build_test_program(
    mut program: LoadedProgram,
    filter: Option<&str>,
) -> Result<(LoadedProgram, usize), String> {
    let mut retained = Vec::new();
    let mut tests = Vec::new();

    for mut entry in program.module.entries {
        match entry.kind {
            TopLevelKind::Test(test) => {
                if filter.is_some_and(|filter| !test.name.contains(filter)) {
                    continue;
                }
                let function_name = format!("__skunk_test_{}", tests.len());
                tests.push((function_name.clone(), test.name));
                entry.visibility = Visibility::Private;
                entry.kind = TopLevelKind::Function(FunctionDecl {
                    name: function_name,
                    generic_parameters: Vec::new(),
                    parameters: Vec::new(),
                    return_type: TypeSyntax {
                        id: entry.id,
                        span: entry.span,
                        kind: TypeSyntaxKind::Builtin(BuiltinType::Void),
                    },
                    body: test.body,
                });
                retained.push(entry);
            }
            TopLevelKind::Function(ref function) if function.name == "main" => {}
            _ => retained.push(entry),
        }
    }

    if tests.is_empty() {
        return Err(match filter {
            Some(filter) => format!("no tests match filter `{filter}`"),
            None => "no tests found".to_string(),
        });
    }

    let mut generated = String::from(TESTING_PRELUDE);
    generated.push_str("\nfunction main(): int {\n");
    for (function_name, display_name) in &tests {
        generated.push_str(&format!(
            "    skunk_test_begin(\"{}\");\n",
            escape_string_literal(display_name)
        ));
        generated.push_str(&format!("    {function_name}();\n"));
        generated.push_str("    skunk_test_end();\n");
    }
    generated.push_str("    return skunk_test_summary();\n}\n");

    let generated_file = program
        .sources
        .add_file("<skunk-test-runner>", generated)
        .map_err(|error| error.to_string())?;
    let generated_module = crate::syntax::parser::parse_module(&program.sources, generated_file)
        .map_err(|diagnostics| render_diagnostics(diagnostics, &program.sources))?;
    let generated_module = crate::syntax::normalize::normalize(generated_module)
        .map_err(|diagnostics| render_diagnostics(diagnostics, &program.sources))?;

    let mut entries = generated_module.entries;
    entries.extend(retained);
    program.module.entries = entries;
    Ok((program, tests.len()))
}

fn escape_string_literal(value: &str) -> String {
    let mut output = String::with_capacity(value.len());
    for character in value.chars() {
        match character {
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

fn render_diagnostics(
    diagnostics: Vec<crate::diagnostic::Diagnostic>,
    sources: &crate::source_map::SourceMap,
) -> String {
    diagnostics
        .into_iter()
        .map(|diagnostic| diagnostic.render(sources))
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source_map::SourceMap;

    fn loaded(source: &str) -> LoadedProgram {
        let mut sources = SourceMap::default();
        let file = sources.add_file("tests.skunk", source).unwrap();
        let module = crate::syntax::parser::parse_module(&sources, file).unwrap();
        let module = crate::syntax::normalize::normalize(module).unwrap();
        LoadedProgram { module, sources }
    }

    fn function_names(program: &LoadedProgram) -> Vec<&str> {
        program
            .module
            .entries
            .iter()
            .filter_map(|entry| match &entry.kind {
                TopLevelKind::Function(function) => Some(function.name.as_str()),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn constructs_runner_in_the_syntax_ast() {
        let program = loaded(
            r#"
                test "addition works" { Testing::expect(1 + 1 == 2); }
                test "subtraction works" { Testing::expect_eq(1, 2 - 1); }
            "#,
        );
        let (program, count) = build_test_program(program, None).unwrap();

        assert_eq!(count, 2);
        let names = function_names(&program);
        assert!(names.contains(&"__skunk_test_0"));
        assert!(names.contains(&"__skunk_test_1"));
        assert!(names.contains(&"main"));
    }

    #[test]
    fn filters_tests_and_replaces_user_main() {
        let program = loaded(
            r#"
                function main(): void { print(99); }
                test "fast" { Testing::expect(true); }
                test "slow" { Testing::fail(); }
            "#,
        );
        let (program, count) = build_test_program(program, Some("fast")).unwrap();

        assert_eq!(count, 1);
        assert_eq!(
            function_names(&program)
                .iter()
                .filter(|name| **name == "main")
                .count(),
            1
        );
    }

    #[test]
    fn reports_empty_selection() {
        let error = build_test_program(loaded("test \"one\" {}"), Some("missing")).unwrap_err();
        assert!(error.contains("no tests match"));
    }
}
