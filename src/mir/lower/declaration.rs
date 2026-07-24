use super::*;
use crate::syntax::ast::Visibility;

pub(super) fn lower_module(
    module: &hir::Module,
    model: &SemanticModel,
    functions: &mut Vec<Function>,
    diagnostics: &mut Vec<Diagnostic>,
) -> Vec<Declaration> {
    let mut declarations = Vec::new();
    for item in &module.items {
        let exported = item.visibility == Visibility::Public;
        match &item.kind {
            hir::ItemKind::Function(function) => collect_function(
                function,
                item.source,
                item.span,
                None,
                model,
                functions,
                diagnostics,
            ),
            hir::ItemKind::Struct(declaration) => {
                let Some(definition) = required_definition(item, "struct", diagnostics) else {
                    continue;
                };
                declarations.push(Declaration {
                    source: item.source,
                    span: item.span,
                    exported,
                    kind: DeclarationKind::Struct {
                        definition,
                        fields: declaration
                            .fields
                            .iter()
                            .map(|field| FieldDeclaration {
                                id: field.id,
                                name: field.name.clone(),
                                ty: field.ty,
                                mutable: !field.is_const,
                            })
                            .collect(),
                        methods: method_definitions(&declaration.methods),
                    },
                });
                lower_methods(declaration, definition, model, functions, diagnostics);
            }
            hir::ItemKind::Enum(declaration) => {
                let Some(definition) = required_definition(item, "enum", diagnostics) else {
                    continue;
                };
                declarations.push(Declaration {
                    source: item.source,
                    span: item.span,
                    exported,
                    kind: DeclarationKind::Enum {
                        definition,
                        variants: declaration
                            .variants
                            .iter()
                            .map(|variant| VariantDeclaration {
                                id: variant.id,
                                name: variant.name.clone(),
                                payload: variant.payload.clone(),
                            })
                            .collect(),
                        methods: method_definitions(&declaration.methods),
                    },
                });
                for function in &declaration.methods {
                    collect_function(
                        function,
                        function.body.source,
                        function.body.span,
                        Some(definition),
                        model,
                        functions,
                        diagnostics,
                    );
                }
            }
            hir::ItemKind::Trait(declaration) => {
                let Some(definition) = required_definition(item, "trait", diagnostics) else {
                    continue;
                };
                declarations.push(Declaration {
                    source: item.source,
                    span: item.span,
                    exported,
                    kind: DeclarationKind::Trait {
                        definition,
                        supertraits: declaration.supertraits.clone(),
                        methods: lower_trait_methods(&declaration.methods),
                    },
                });
            }
            hir::ItemKind::Shape(declaration) => {
                let Some(definition) = required_definition(item, "shape", diagnostics) else {
                    continue;
                };
                declarations.push(Declaration {
                    source: item.source,
                    span: item.span,
                    exported,
                    kind: DeclarationKind::Shape {
                        definition,
                        methods: lower_trait_methods(&declaration.methods),
                    },
                });
            }
            hir::ItemKind::Implementation { traits, target } => {
                declarations.push(Declaration {
                    source: item.source,
                    span: item.span,
                    exported,
                    kind: DeclarationKind::Implementation {
                        traits: traits.clone(),
                        target: *target,
                    },
                });
            }
            hir::ItemKind::ExternFunction(signature) => {
                let Some(definition) = required_definition(item, "extern function", diagnostics)
                else {
                    continue;
                };
                declarations.push(Declaration {
                    source: item.source,
                    span: item.span,
                    exported,
                    kind: DeclarationKind::ExternFunction {
                        definition,
                        parameters: signature.parameters.clone(),
                        result: signature.result,
                    },
                });
            }
            hir::ItemKind::Global(global) => {
                let Some(definition) = required_definition(item, "global", diagnostics) else {
                    continue;
                };
                declarations.push(Declaration {
                    source: item.source,
                    span: item.span,
                    exported,
                    kind: DeclarationKind::Global {
                        definition,
                        ty: global.ty,
                        mutable: !global.is_const,
                        initializer: global
                            .initializer
                            .as_ref()
                            .map(|expression| expression.source),
                    },
                });
                if let Some(initializer) = &global.initializer {
                    match FunctionLowerer::new_initializer(
                        initializer,
                        global.ty,
                        definition,
                        initializer.source,
                        initializer.span,
                        model,
                    )
                    .lower()
                    {
                        Ok(mut lowered) => functions.append(&mut lowered),
                        Err(mut errors) => diagnostics.append(&mut errors),
                    }
                }
            }
            // Tests are converted into ordinary functions by the test-program
            // builder. Top-level statements have no runtime declaration.
            hir::ItemKind::Test(_) | hir::ItemKind::Statement(_) => {}
        }
    }
    declarations
}

fn lower_methods(
    declaration: &hir::Struct,
    owner: crate::ids::DefId,
    model: &SemanticModel,
    functions: &mut Vec<Function>,
    diagnostics: &mut Vec<Diagnostic>,
) {
    for function in &declaration.methods {
        collect_function(
            function,
            function.body.source,
            function.body.span,
            Some(owner),
            model,
            functions,
            diagnostics,
        );
    }
}

fn method_definitions(methods: &[hir::Function]) -> Vec<crate::ids::DefId> {
    methods
        .iter()
        .filter_map(|method| method.definition)
        .collect()
}

fn required_definition(
    item: &hir::Item,
    kind: &str,
    diagnostics: &mut Vec<Diagnostic>,
) -> Option<crate::ids::DefId> {
    item.definition.or_else(|| {
        diagnostics.push(
            Diagnostic::error(format!("MIR {kind} declaration has no definition identity"))
                .with_code("E5007")
                .at(item.span),
        );
        None
    })
}

fn lower_trait_methods(methods: &[hir::TraitMethod]) -> Vec<TraitMethodDeclaration> {
    methods
        .iter()
        .map(|method| TraitMethodDeclaration {
            definition: method.definition,
            name: method.name.clone(),
            receiver: method.receiver.map(|receiver| Receiver {
                mutable: receiver.mutable,
                is_const: receiver.is_const,
            }),
            parameters: method.parameters.clone(),
            result: method.result,
            default_body: method.default_body.as_ref().map(|body| body.source),
        })
        .collect()
}
