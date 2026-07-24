use super::*;

#[test]
fn lowers_for_loops_into_cfg_back_edges() {
    let (module, semantics) = lower_source(
        r#"
        function sum_to_three(): int {
            total: int = 0;
            for (i: int = 0; i < 3; i = i + 1) {
                total = total + i;
            }
            return total;
        }
        "#,
    );
    let function = function_named(&module, &semantics, "sum_to_three");

    assert!(function
        .blocks
        .iter()
        .any(|block| matches!(block.terminator.kind, TerminatorKind::If { .. })));
    assert!(function.blocks.iter().any(|block| matches!(
        block.terminator.kind,
        TerminatorKind::Goto { target } if target.index() < block.id.index()
    )));
}

#[test]
fn lowers_struct_destructure_bindings_as_field_projections() {
    let (module, semantics) = lower_source(
        r#"
        struct Point {
            x: int;
            y: int;
        }

        function sum(point: Point): int {
            Point { x, y } = point;
            return x + y;
        }
        "#,
    );
    let function = function_named(&module, &semantics, "sum");

    assert_eq!(
        function.blocks[0]
            .statements
            .iter()
            .filter(|statement| matches!(
                &statement.kind,
                StatementKind::Assign {
                    value: Rvalue {
                        kind: RvalueKind::Use(Operand {
                            kind: OperandKind::Copy(Place { projections, .. }),
                            ..
                        }),
                        ..
                    },
                    ..
                } if matches!(
                    projections.last().map(|projection| &projection.kind),
                    Some(ProjectionKind::Field(_))
                )
            ))
            .count(),
        2
    );
}

#[test]
fn lowers_enum_matches_and_payload_bindings() {
    let (module, semantics) = lower_source(
        r#"
        enum Maybe {
            None;
            Some(int);
        }

        function unwrap(value: Maybe): int {
            match (value) {
                case None: {
                    return 0;
                }
                case Some(inner): {
                    return inner;
                }
            }
        }
        "#,
    );
    let function = function_named(&module, &semantics, "unwrap");
    let switch = function
        .blocks
        .iter()
        .find_map(|block| match &block.terminator.kind {
            TerminatorKind::SwitchEnum { targets, .. } => Some(targets),
            _ => None,
        })
        .expect("unwrap should contain an enum switch");

    assert_eq!(switch.len(), 2);
    assert!(function.blocks.iter().any(|block| {
        block.statements.iter().any(|statement| {
            matches!(
                &statement.kind,
                StatementKind::Assign {
                    value: Rvalue {
                        kind: RvalueKind::Use(Operand {
                            kind: OperandKind::Copy(Place { projections, .. }),
                            ..
                        }),
                        ..
                    },
                    ..
                } if matches!(
                    projections.last().map(|projection| &projection.kind),
                    Some(ProjectionKind::VariantField { .. })
                )
            )
        })
    }));
}

#[test]
fn lowers_struct_matches_as_irrefutable_branches() {
    let (module, semantics) = lower_source(
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
        "#,
    );
    let function = function_named(&module, &semantics, "sum");

    assert!(matches!(
        function.blocks[function.entry.index()].terminator.kind,
        TerminatorKind::Goto { .. }
    ));
    assert_eq!(
        function
            .blocks
            .iter()
            .flat_map(|block| &block.statements)
            .filter(|statement| matches!(
                &statement.kind,
                StatementKind::Assign {
                    value: Rvalue {
                        kind: RvalueKind::Use(Operand {
                            kind: OperandKind::Copy(Place { projections, .. }),
                            ..
                        }),
                        ..
                    },
                    ..
                } if matches!(
                    projections.last().map(|projection| &projection.kind),
                    Some(ProjectionKind::Field(_))
                )
            ))
            .count(),
        2
    );
}

#[test]
fn emits_deferred_calls_in_lifo_order_on_every_exit() {
    let (module, semantics) = lower_source(
        r#"
        function log(value: int): void {}

        function finish(early: boolean): int {
            defer log(1);
            if (early) {
                defer log(2);
                return 7;
            }
            return 8;
        }
        "#,
    );
    let function = function_named(&module, &semantics, "finish");
    let mut deferred_arguments = function
        .blocks
        .iter()
        .filter(|block| matches!(block.terminator.kind, TerminatorKind::Return(_)))
        .map(|block| {
            block
                .statements
                .iter()
                .filter_map(|statement| match &statement.kind {
                    StatementKind::Call {
                        argument_groups, ..
                    } => match &argument_groups[0][0].kind {
                        OperandKind::Constant(Constant::Literal(
                            crate::syntax::ast::Literal::Integer(value),
                        )) => Some(*value),
                        _ => None,
                    },
                    _ => None,
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    deferred_arguments.sort_by_key(Vec::len);

    assert_eq!(deferred_arguments, vec![vec![1], vec![2, 1]]);
}

#[test]
fn lowers_io_as_explicit_mir_statements() {
    let (module, semantics) = lower_source(
        r#"
        function main(): void {
            print(7);
            input();
        }
        "#,
    );
    let function = function_named(&module, &semantics, "main");

    assert!(matches!(
        function.blocks[0].statements[0].kind,
        StatementKind::Print(_)
    ));
    assert_eq!(function.blocks[0].statements[1].kind, StatementKind::Input);
}

#[test]
fn validator_rejects_a_non_exhaustive_enum_switch() {
    let (mut module, semantics) = lower_source(
        r#"
        enum Maybe {
            None;
            Some(int);
        }

        function unwrap(value: Maybe): int {
            match (value) {
                case None: { return 0; }
                case Some(inner): { return inner; }
            }
        }
        "#,
    );
    let function = &mut module.functions[0];
    let targets = function
        .blocks
        .iter_mut()
        .find_map(|block| match &mut block.terminator.kind {
            TerminatorKind::SwitchEnum { targets, .. } => Some(targets),
            _ => None,
        })
        .expect("unwrap should contain an enum switch");
    targets.pop();

    let diagnostics = validate::validate(&module, &semantics).unwrap_err();
    assert!(diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == Some("E5141")));
}

#[test]
fn checked_generic_program_lowers_through_mir_control_flow() {
    let checked = crate::pipeline::check_source(
        "generic-match.skunk",
        r#"
        enum Option[T] {
            None;
            Some(T);
        }

        function unwrap(value: Option[int]): int {
            match (value) {
                case None: { return 0; }
                case Some(inner): { return inner; }
            }
        }

        function main(): int {
            total: int = 0;
            for (i: int = 0; i < 2; i = i + 1) {
                defer unwrap(Option[int]::None());
                total = total + unwrap(Option[int]::Some(i));
            }
            return total;
        }
        "#,
    )
    .unwrap();
    let module = crate::pipeline::lower_to_mir(&checked).unwrap();

    assert!(module.functions.iter().any(|function| function
        .blocks
        .iter()
        .any(|block| matches!(block.terminator.kind, TerminatorKind::SwitchEnum { .. }))));
    assert!(module.functions.iter().any(|function| function
        .blocks
        .iter()
        .any(|block| matches!(block.terminator.kind, TerminatorKind::If { .. }))));
}
