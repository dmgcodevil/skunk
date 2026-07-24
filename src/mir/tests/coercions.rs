use super::*;

#[test]
fn makes_numeric_and_trait_object_coercions_explicit() {
    let (module, _) = lower_source(
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

        function widen(value: byte): long {
            widened: long = value;
            return widened;
        }

        function main(): void {
            writer: Writer = Counter { value: 1 };
        }
        "#,
    );

    let coercions = module
        .functions
        .iter()
        .flat_map(|function| &function.blocks)
        .flat_map(|block| &block.statements)
        .filter(|statement| {
            matches!(
                statement.kind,
                StatementKind::Assign {
                    value: Rvalue {
                        kind: RvalueKind::Coerce(_),
                        ..
                    },
                    ..
                }
            )
        })
        .count();

    assert_eq!(coercions, 2);
}

#[test]
fn validator_rejects_an_invalid_explicit_coercion() {
    let (mut module, semantics) = lower_source(
        r#"
        function widen(value: byte, flag: boolean): long {
            widened: long = value;
            return widened;
        }
        "#,
    );
    let function = &mut module.functions[0];
    let flag = function.parameters[1];
    let flag_ty = function.locals[flag.index()].ty;
    let operand = function.blocks[0]
        .statements
        .iter_mut()
        .find_map(|statement| match &mut statement.kind {
            StatementKind::Assign {
                value:
                    Rvalue {
                        kind: RvalueKind::Coerce(operand),
                        ..
                    },
                ..
            } => Some(operand),
            _ => None,
        })
        .expect("widening should be represented by a MIR coercion");
    *operand = Operand {
        ty: flag_ty,
        kind: OperandKind::Copy(Place::local(flag)),
    };

    let diagnostics = validate::validate(&module, &semantics).unwrap_err();
    assert!(diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == Some("E5179")));
}
