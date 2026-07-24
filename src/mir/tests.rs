use super::*;
use crate::analysis::{model, resolver};
use crate::hir;
use crate::ids::MirBlockId;

fn lower_source(source: &str) -> (Module, model::SemanticModel) {
    let syntax = crate::syntax::parser::parse_test_module(source);
    let resolutions = resolver::resolve(&syntax).unwrap();
    let mut semantics = model::analyze_declarations(&syntax, resolutions).unwrap();
    let hir = hir::lower::lower(&syntax, &mut semantics).unwrap();
    hir::validate::validate(&hir, &semantics).unwrap();
    let mir = lower::lower(&hir, &semantics).unwrap();
    validate::validate(&mir, &semantics).unwrap();
    (mir, semantics)
}

fn function_named<'a>(
    module: &'a Module,
    semantics: &model::SemanticModel,
    name: &str,
) -> &'a Function {
    module
        .functions
        .iter()
        .find(|function| {
            function.definition.is_some_and(|definition| {
                semantics.resolutions.definitions[definition.index()].name == name
            })
        })
        .unwrap_or_else(|| panic!("missing MIR function `{name}`"))
}

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
fn unsupported_hir_is_reported_as_a_mir_diagnostic() {
    let syntax = crate::syntax::parser::parse_test_module(
        r#"
        function main(): void {
            print(1);
        }
        "#,
    );
    let resolutions = resolver::resolve(&syntax).unwrap();
    let mut semantics = model::analyze_declarations(&syntax, resolutions).unwrap();
    let hir = hir::lower::lower(&syntax, &mut semantics).unwrap();
    let diagnostics = lower::lower(&hir, &semantics).unwrap_err();

    assert!(diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == Some("E5007")));
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

#[test]
fn lowers_struct_construction_and_field_places() {
    let (module, semantics) = lower_source(
        r#"
        struct Point {
            x: int;
            y: int;
        }

        function update(): int {
            point: Point = Point { x: 1, y: 2 };
            point.x = point.x + 4;
            return point.x;
        }
        "#,
    );
    let function = function_named(&module, &semantics, "update");

    assert!(function.blocks[0]
        .statements
        .iter()
        .any(|statement| matches!(
            statement.kind,
            StatementKind::Assign {
                value: Rvalue {
                    kind: RvalueKind::Aggregate(Aggregate::Struct { .. }),
                    ..
                },
                ..
            }
        )));
    assert!(function.blocks[0]
        .statements
        .iter()
        .any(|statement| matches!(
            &statement.kind,
            StatementKind::Assign {
                destination: Place { projections, .. },
                ..
            } if projections
                .iter()
                .any(|projection| matches!(projection.kind, ProjectionKind::Field(_)))
        )));
}

#[test]
fn lowers_array_index_length_and_slice_operations() {
    let (module, semantics) = lower_source(
        r#"
        function inspect(): int {
            values: [3]int = [1, 2, 3];
            values[1] = 4;
            view: []int = values[1:];
            return view[0] + values.len;
        }
        "#,
    );
    let function = function_named(&module, &semantics, "inspect");
    let statements = &function.blocks[0].statements;

    assert!(statements.iter().any(|statement| matches!(
        statement.kind,
        StatementKind::Assign {
            value: Rvalue {
                kind: RvalueKind::Aggregate(Aggregate::Array(_)),
                ..
            },
            ..
        }
    )));
    assert!(statements.iter().any(|statement| matches!(
        statement.kind,
        StatementKind::Assign {
            value: Rvalue {
                kind: RvalueKind::Slice { .. },
                ..
            },
            ..
        }
    )));
    assert!(statements.iter().any(|statement| matches!(
        statement.kind,
        StatementKind::Assign {
            value: Rvalue {
                kind: RvalueKind::Length(_),
                ..
            },
            ..
        }
    )));
    assert!(statements.iter().any(|statement| match &statement.kind {
        StatementKind::Assign { destination, .. } => destination
            .projections
            .iter()
            .any(|projection| matches!(projection.kind, ProjectionKind::Index(_))),
        _ => false,
    }));
}

#[test]
fn lowers_references_and_dereference_places() {
    let (module, semantics) = lower_source(
        r#"
        function update(): int {
            value: int = 41;
            reference: &mut int = &mut value;
            reference.* = 42;
            return reference.*;
        }
        "#,
    );
    let function = function_named(&module, &semantics, "update");

    assert!(function.blocks[0]
        .statements
        .iter()
        .any(|statement| matches!(
            statement.kind,
            StatementKind::Assign {
                value: Rvalue {
                    kind: RvalueKind::Reference { mutable: true, .. },
                    ..
                },
                ..
            }
        )));
    assert!(function.blocks[0]
        .statements
        .iter()
        .any(|statement| match &statement.kind {
            StatementKind::Assign { destination, .. } => destination
                .projections
                .iter()
                .any(|projection| projection.kind == ProjectionKind::Dereference),
            _ => false,
        }));
}

#[test]
fn preserves_static_and_method_call_dispatch() {
    let checked = crate::pipeline::check_source(
        "calls.skunk",
        r#"
        struct Point {
            x: int;
            y: int;
        }

        attach Point {
            function new(x: int, y: int): Point {
                return Point { x: x, y: y };
            }

            function sum(self): int {
                return self.x + self.y;
            }
        }

        function main(): int {
            point: Point = Point::new(4, 9);
            return point.sum();
        }
        "#,
    )
    .unwrap();
    let module = crate::pipeline::lower_to_mir(&checked).unwrap();
    let function = function_named(&module, &checked.semantics, "main");

    assert!(function.blocks[0]
        .statements
        .iter()
        .any(|statement| matches!(
            statement.kind,
            StatementKind::Call {
                target: CallTarget::Static(StaticCallee::Definition(_)),
                ..
            }
        )));
    assert!(function.blocks[0]
        .statements
        .iter()
        .any(|statement| matches!(
            statement.kind,
            StatementKind::Call {
                target: CallTarget::Method {
                    method: MethodCallee::Definition(_),
                    ..
                },
                ..
            }
        )));
}

#[test]
fn makes_implicit_field_auto_dereference_explicit() {
    let (module, semantics) = lower_source(
        r#"
        struct Counter {
            value: int;
        }

        function read(counter: &Counter): int {
            return counter.value;
        }
        "#,
    );
    let function = function_named(&module, &semantics, "read");
    let TerminatorKind::Return(Some(Operand {
        kind: OperandKind::Copy(place),
        ..
    })) = &function.blocks[0].terminator.kind
    else {
        panic!("read should return a place operand");
    };

    assert_eq!(place.projections.len(), 2);
    assert_eq!(place.projections[0].kind, ProjectionKind::Dereference);
    assert!(matches!(
        place.projections[1].kind,
        ProjectionKind::Field(_)
    ));
}

#[test]
fn lowers_global_reads_and_writes_as_definition_places() {
    let (module, semantics) = lower_source(
        r#"
        counter: int = 1;

        function bump(): int {
            counter = counter + 1;
            return counter;
        }
        "#,
    );
    let function = function_named(&module, &semantics, "bump");

    assert!(function.blocks[0]
        .statements
        .iter()
        .any(|statement| matches!(
            statement.kind,
            StatementKind::Assign {
                destination: Place {
                    base: PlaceBase::Definition(_),
                    ..
                },
                ..
            }
        )));
    assert!(matches!(
        function.blocks[0].terminator.kind,
        TerminatorKind::Return(Some(Operand {
            kind: OperandKind::Copy(Place {
                base: PlaceBase::Definition(_),
                ..
            }),
            ..
        }))
    ));
}

#[test]
fn validator_rejects_an_invalid_index_projection_type() {
    let (mut module, semantics) = lower_source(
        r#"
        function read(): int {
            values: [2]int = [4, 9];
            return values[0];
        }
        "#,
    );
    let function = &mut module.functions[0];
    let array_ty = function.locals[0].ty;
    let TerminatorKind::Return(Some(Operand {
        kind: OperandKind::Copy(place),
        ..
    })) = &mut function.blocks[0].terminator.kind
    else {
        panic!("read should return an indexed place");
    };
    place.projections[0].ty = array_ty;

    let diagnostics = validate::validate(&module, &semantics).unwrap_err();
    assert!(diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == Some("E5134")));
}

#[test]
fn pointer_indexing_does_not_insert_a_scalar_dereference() {
    let (module, semantics) = lower_source(
        r#"
        function read(values: *int): int {
            return values[1];
        }
        "#,
    );
    let function = function_named(&module, &semantics, "read");
    let TerminatorKind::Return(Some(Operand {
        kind: OperandKind::Copy(place),
        ..
    })) = &function.blocks[0].terminator.kind
    else {
        panic!("read should return an indexed place");
    };

    assert_eq!(place.projections.len(), 1);
    assert!(matches!(
        place.projections[0].kind,
        ProjectionKind::Index(_)
    ));
}

#[test]
fn preserves_enum_constructor_targets() {
    let (module, semantics) = lower_source(
        r#"
        enum Maybe {
            Some(int);
            None;
        }

        function make(): Maybe {
            return Maybe::Some(7);
        }
        "#,
    );
    let function = function_named(&module, &semantics, "make");

    assert!(function.blocks[0]
        .statements
        .iter()
        .any(|statement| matches!(
            statement.kind,
            StatementKind::Call {
                target: CallTarget::Static(StaticCallee::Variant(_)),
                ..
            }
        )));
}
