use super::*;

#[test]
fn preserves_the_typed_program_surface_needed_by_codegen() {
    let (module, _) = lower_source(
        r#"
        export struct Counter {
            value: int;
            flag: boolean;
        }

        export enum Maybe {
            None;
            Some(int);
        }

        export trait Writer {
            function write(mut self, value: int): int;
        }

        shape WriterLike {
            function write(mut self, value: int): int;
        }

        conform Writer for Counter {
            function write(mut self, value: int): int {
                self.value = self.value + value;
                return self.value;
            }
        }

        export extern "C" function external(value: int): int;

        function main(): int {
            return 0;
        }
        "#,
    );

    assert!(module.declarations.iter().any(|declaration| matches!(
        declaration.kind,
        DeclarationKind::Struct { .. }
    ) && declaration.exported));
    assert!(module.declarations.iter().any(|declaration| matches!(
        declaration.kind,
        DeclarationKind::Enum { .. }
    ) && declaration.exported));
    assert!(module.declarations.iter().any(|declaration| matches!(
        declaration.kind,
        DeclarationKind::Trait { .. }
    ) && declaration.exported));
    assert!(module
        .declarations
        .iter()
        .any(|declaration| matches!(declaration.kind, DeclarationKind::Shape { .. })));
    assert!(module
        .declarations
        .iter()
        .any(|declaration| matches!(declaration.kind, DeclarationKind::Implementation { .. })));
    assert!(module.declarations.iter().any(|declaration| matches!(
        declaration.kind,
        DeclarationKind::ExternFunction { .. }
    ) && declaration.exported));

    let (owner, methods) = module
        .declarations
        .iter()
        .find_map(|declaration| match &declaration.kind {
            DeclarationKind::Struct {
                definition,
                methods,
                ..
            } => Some((*definition, methods)),
            _ => None,
        })
        .expect("Counter should have a MIR declaration");
    assert_eq!(methods.len(), 1);
    assert!(module.functions.iter().any(|function| {
        function.definition() == Some(methods[0]) && function.owner() == Some(owner)
    }));
}

#[test]
fn preserves_global_storage_metadata_without_embedding_hir() {
    let (module, _) = lower_source(
        r#"
        counter: int = 1;

        function main(): int {
            return counter;
        }
        "#,
    );

    let (global, initializer) = module
        .declarations
        .iter()
        .find_map(|declaration| match declaration.kind {
            DeclarationKind::Global {
                definition,
                mutable: true,
                initializer: Some(initializer),
                ..
            } => Some((definition, initializer)),
            _ => None,
        })
        .expect("counter should retain its global declaration");
    let body = module
        .functions
        .iter()
        .find(|function| function.source == initializer)
        .expect("global initializer should have an executable MIR body");
    assert_eq!(body.origin, FunctionOrigin::GlobalInitializer { global });
    assert!(matches!(
        body.blocks[body.entry.index()].terminator.kind,
        TerminatorKind::Return(Some(_))
    ));
}

#[test]
fn validator_rejects_a_global_with_a_missing_initializer_body() {
    let (mut module, semantics) = lower_source(
        r#"
        counter: int = 1;
        function main(): int { return counter; }
        "#,
    );
    module
        .functions
        .retain(|function| !matches!(function.origin, FunctionOrigin::GlobalInitializer { .. }));

    let diagnostics = validate::validate(&module, &semantics).unwrap_err();
    assert!(diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == Some("E5181")));
}

#[test]
fn validator_rejects_nominal_members_with_corrupted_types() {
    let (mut module, semantics) = lower_source(
        r#"
        struct Pair {
            number: int;
            enabled: boolean;
        }
        "#,
    );

    let DeclarationKind::Struct { fields, .. } = &mut module.declarations[0].kind else {
        panic!("expected a struct declaration");
    };
    fields[0].ty = fields[1].ty;

    let diagnostics = validate::validate(&module, &semantics).unwrap_err();
    assert!(diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == Some("E5163")));
}

#[test]
fn validator_rejects_duplicate_function_source_identities() {
    let (mut module, semantics) = lower_source(
        r#"
        function first(): void {}
        function second(): void {}
        "#,
    );
    module.functions[1].source = module.functions[0].source;

    let diagnostics = validate::validate(&module, &semantics).unwrap_err();
    assert!(diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == Some("E5156")));
}
