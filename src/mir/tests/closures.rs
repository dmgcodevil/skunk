use super::*;

#[test]
fn captures_only_outer_locals_referenced_by_a_lambda() {
    let (module, semantics) = lower_source(
        r#"
        function main(): int {
            used: int = 41;
            unused: int = 9;
            closure: () -> int = function(): int {
                return used + 1;
            };
            return closure();
        }
        "#,
    );
    let closure = module
        .functions
        .iter()
        .find(|function| function.origin == FunctionOrigin::Closure)
        .expect("lambda should become a nested MIR function");

    assert_eq!(closure.captures.len(), 1);
    let source = closure.locals[closure.captures[0].index()]
        .source
        .expect("capture should retain its HIR local identity");
    assert_eq!(semantics.resolutions.locals[source.index()].name, "used");
}

#[test]
fn captured_mutation_targets_a_capture_local() {
    let checked = crate::pipeline::check_source(
        "closure.skunk",
        r#"
        function counter(): () -> int {
            value: int = 0;
            return function(): int {
                value = value + 1;
                return value;
            };
        }

        function main(): int {
            count: () -> int = counter();
            return count();
        }
        "#,
    )
    .unwrap();
    let module = crate::pipeline::lower_to_mir(&checked).unwrap();
    let closure = module
        .functions
        .iter()
        .find(|function| function.origin == FunctionOrigin::Closure)
        .expect("counter should contain a nested closure");
    let capture = closure.captures[0];

    assert!(closure
        .blocks
        .iter()
        .any(|block| block.statements.iter().any(|statement| matches!(
            statement.kind,
            StatementKind::Assign {
                destination: Place {
                    base: PlaceBase::Local(local),
                    ..
                },
                ..
            } if local == capture
        ))));
}

#[test]
fn recursive_lambda_captures_its_own_binding() {
    let checked = crate::pipeline::check_source(
        "recursive-closure.skunk",
        r#"
        function main(): int {
            factorial: (int) -> int = function(value: int): int {
                if (value == 0) {
                    return 1;
                }
                return value * factorial(value - 1);
            };
            return factorial(5);
        }
        "#,
    )
    .unwrap();
    let module = crate::pipeline::lower_to_mir(&checked).unwrap();
    let closure = module
        .functions
        .iter()
        .find(|function| function.origin == FunctionOrigin::Closure)
        .expect("factorial should become a nested MIR function");

    assert_eq!(closure.captures.len(), 1);
}

#[test]
fn validator_rejects_a_closure_with_missing_captures() {
    let (mut module, semantics) = lower_source(
        r#"
        function main(): int {
            value: int = 7;
            closure: () -> int = function(): int { return value; };
            return closure();
        }
        "#,
    );
    let rvalue = module
        .functions
        .iter_mut()
        .flat_map(|function| &mut function.blocks)
        .flat_map(|block| &mut block.statements)
        .find_map(|statement| match &mut statement.kind {
            StatementKind::Assign {
                value:
                    Rvalue {
                        kind: RvalueKind::Closure { captures, .. },
                        ..
                    },
                ..
            } => Some(captures),
            _ => None,
        })
        .expect("main should construct a closure");
    rvalue.clear();

    let diagnostics = validate::validate(&module, &semantics).unwrap_err();
    assert!(diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == Some("E5153")));
}
