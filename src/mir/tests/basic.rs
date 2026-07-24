use super::*;

#[test]
fn lowers_locals_and_arithmetic_into_three_address_statements() {
    let (module, semantics) = lower_source(
        r#"
        function add(left: int, right: int): int {
            result: int = left + right;
            return result;
        }
        "#,
    );
    let function = function_named(&module, &semantics, "add");

    assert_eq!(function.parameters.len(), 2);
    assert_eq!(function.locals.len(), 4);
    assert_eq!(function.blocks.len(), 1);
    assert_eq!(function.blocks[0].statements.len(), 2);
    assert!(matches!(
        function.blocks[0].statements[0].kind,
        StatementKind::Assign {
            value: Rvalue {
                kind: RvalueKind::Binary { .. },
                ..
            },
            ..
        }
    ));
    assert!(matches!(
        function.blocks[0].terminator.kind,
        TerminatorKind::Return(Some(Operand {
            kind: OperandKind::Copy(_),
            ..
        }))
    ));
}

#[test]
fn lowers_if_else_into_explicit_control_flow() {
    let (module, semantics) = lower_source(
        r#"
        function absolute(value: int): int {
            if (value < 0) {
                return -value;
            } else {
                return value;
            }
        }
        "#,
    );
    let function = function_named(&module, &semantics, "absolute");

    assert_eq!(function.blocks.len(), 4);
    let TerminatorKind::If {
        then_target,
        else_target,
        ..
    } = function.blocks[function.entry.index()].terminator.kind
    else {
        panic!("entry block should branch");
    };
    assert!(matches!(
        function.blocks[then_target.index()].terminator.kind,
        TerminatorKind::Return(Some(_))
    ));
    assert!(matches!(
        function.blocks[else_target.index()].terminator.kind,
        TerminatorKind::Return(Some(_))
    ));
    assert!(function
        .blocks
        .iter()
        .any(|block| matches!(block.terminator.kind, TerminatorKind::Unreachable)));
}

#[test]
fn lowers_else_if_as_a_chain_of_boolean_terminators() {
    let (module, semantics) = lower_source(
        r#"
        function classify(value: int): int {
            if (value < 0) {
                return -1;
            } else if (value == 0) {
                return 0;
            } else {
                return 1;
            }
        }
        "#,
    );
    let function = function_named(&module, &semantics, "classify");

    assert_eq!(
        function
            .blocks
            .iter()
            .filter(|block| matches!(block.terminator.kind, TerminatorKind::If { .. }))
            .count(),
        2
    );
}

#[test]
fn lowers_calls_without_losing_argument_groups_or_result_types() {
    let (module, semantics) = lower_source(
        r#"
        function increment(value: int): int {
            return value + 1;
        }

        function main(): int {
            result: int = increment(41);
            return result;
        }
        "#,
    );
    let function = function_named(&module, &semantics, "main");
    let call = function.blocks[0]
        .statements
        .iter()
        .find_map(|statement| match &statement.kind {
            StatementKind::Call {
                argument_groups, ..
            } => Some(argument_groups),
            _ => None,
        })
        .expect("main should contain a MIR call");

    assert_eq!(call.len(), 1);
    assert_eq!(call[0].len(), 1);
}

#[test]
fn lowers_non_capturing_lambdas_as_nested_functions() {
    let (module, _) = lower_source(
        r#"
        function main(): void {
            identity: (int) -> int = function(value: int): int {
                return value;
            };
        }
        "#,
    );
    let closure = module
        .functions
        .iter()
        .find(|function| function.definition.is_none())
        .expect("lambda should become a nested MIR function");

    assert!(closure.captures.is_empty());
    assert_eq!(closure.parameters.len(), 1);
    assert!(module.functions.iter().any(|function| function
        .blocks
        .iter()
        .flat_map(|block| &block.statements)
        .any(|statement| matches!(
            statement.kind,
            StatementKind::Assign {
                value: Rvalue {
                    kind: RvalueKind::Closure { .. },
                    ..
                },
                ..
            }
        ))));
}

#[test]
fn validator_rejects_an_unknown_control_flow_target() {
    let (mut module, semantics) = lower_source(
        r#"
        function main(): void {}
        "#,
    );
    module.functions[0].blocks[0].terminator.kind = TerminatorKind::Goto {
        target: MirBlockId::new(99),
    };

    let diagnostics = validate::validate(&module, &semantics).unwrap_err();
    assert!(diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == Some("E5117")));
}

#[test]
fn pipeline_exposes_checked_hir_to_mir_lowering() {
    let checked =
        crate::pipeline::check_source("pipeline.skunk", "function main(): int { return 42; }")
            .unwrap();
    let mir = crate::pipeline::lower_to_mir(&checked).unwrap();

    assert_eq!(mir.functions.len(), 1);
    assert!(matches!(
        mir.functions[0].blocks[0].terminator.kind,
        TerminatorKind::Return(Some(_))
    ));
}

#[test]
fn void_calls_do_not_create_void_temporaries() {
    let (module, semantics) = lower_source(
        r#"
        function ping(): void {}

        function main(): void {
            ping();
        }
        "#,
    );
    let function = function_named(&module, &semantics, "main");

    assert!(function.locals.is_empty());
    assert!(matches!(
        function.blocks[0].statements[0].kind,
        StatementKind::Call {
            destination: None,
            ..
        }
    ));
}
