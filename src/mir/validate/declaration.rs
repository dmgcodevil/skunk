use super::*;
use crate::ids::{DefId, TypeId};
use crate::syntax::ast::Visibility;

impl Validator<'_> {
    pub(super) fn declaration(&mut self, declaration: &Declaration) {
        self.span(declaration.span);
        match &declaration.kind {
            DeclarationKind::Struct {
                definition,
                fields,
                methods,
            } => {
                self.declaration_definition(
                    *definition,
                    DefinitionKind::Struct,
                    declaration.exported,
                    declaration.span,
                );
                let mut seen = HashSet::new();
                for field in fields {
                    self.ty(field.ty, declaration.span);
                    if !seen.insert(field.id) {
                        self.duplicate_member("field", "E5161", declaration.span);
                    }
                    if self.model.field_owners.get(&field.id) != Some(definition) {
                        self.diagnostics.push(
                            Diagnostic::error("MIR field does not belong to its struct")
                                .with_code("E5162")
                                .at(declaration.span),
                        );
                    }
                    if self.model.field_types.get(&field.id) != Some(&field.ty) {
                        self.diagnostics.push(
                            Diagnostic::error("MIR field type differs from semantic analysis")
                                .with_code("E5163")
                                .at(declaration.span),
                        );
                    }
                }
                self.declared_methods(*definition, methods, declaration.span);
            }
            DeclarationKind::Enum {
                definition,
                variants,
                methods,
            } => {
                self.declaration_definition(
                    *definition,
                    DefinitionKind::Enum,
                    declaration.exported,
                    declaration.span,
                );
                let mut seen = HashSet::new();
                for variant in variants {
                    if !seen.insert(variant.id) {
                        self.duplicate_member("variant", "E5164", declaration.span);
                    }
                    for ty in &variant.payload {
                        self.ty(*ty, declaration.span);
                    }
                    if self.model.variant_owners.get(&variant.id) != Some(definition) {
                        self.diagnostics.push(
                            Diagnostic::error("MIR variant does not belong to its enum")
                                .with_code("E5165")
                                .at(declaration.span),
                        );
                    }
                    if self.model.variant_payloads.get(&variant.id) != Some(&variant.payload) {
                        self.diagnostics.push(
                            Diagnostic::error("MIR variant payload differs from semantic analysis")
                                .with_code("E5166")
                                .at(declaration.span),
                        );
                    }
                }
                self.declared_methods(*definition, methods, declaration.span);
            }
            DeclarationKind::Trait {
                definition,
                supertraits,
                methods,
            } => {
                self.declaration_definition(
                    *definition,
                    DefinitionKind::Trait,
                    declaration.exported,
                    declaration.span,
                );
                let mut seen = HashSet::new();
                for supertrait in supertraits {
                    if !seen.insert(*supertrait) {
                        self.duplicate_member("supertrait", "E5167", declaration.span);
                    }
                    self.expect_definition_kind(
                        *supertrait,
                        DefinitionKind::Trait,
                        "supertrait",
                        declaration.span,
                    );
                }
                self.trait_methods(methods, declaration.span);
            }
            DeclarationKind::Shape {
                definition,
                methods,
            } => {
                self.declaration_definition(
                    *definition,
                    DefinitionKind::Shape,
                    declaration.exported,
                    declaration.span,
                );
                self.trait_methods(methods, declaration.span);
            }
            DeclarationKind::Implementation { traits, target } => {
                for ty in traits {
                    self.ty(*ty, declaration.span);
                    self.expect_nominal_kind(
                        *ty,
                        DefinitionKind::Trait,
                        "implementation interface",
                        declaration.span,
                    );
                }
                self.ty(*target, declaration.span);
                self.expect_runtime_nominal(*target, declaration.span);
            }
            DeclarationKind::ExternFunction {
                definition,
                parameters,
                result,
            } => {
                self.declaration_definition(
                    *definition,
                    DefinitionKind::ExternFunction,
                    declaration.exported,
                    declaration.span,
                );
                for parameter in parameters {
                    self.ty(*parameter, declaration.span);
                }
                self.ty(*result, declaration.span);
                self.function_signature(
                    *definition,
                    parameters,
                    *result,
                    false,
                    "extern function",
                    declaration.span,
                );
            }
            DeclarationKind::Global {
                definition,
                ty,
                initializer,
                ..
            } => {
                self.declaration_definition(
                    *definition,
                    DefinitionKind::Global,
                    declaration.exported,
                    declaration.span,
                );
                self.ty(*ty, declaration.span);
                if self.model.definition_types.get(definition) != Some(ty) {
                    self.diagnostics.push(
                        Diagnostic::error("MIR global type differs from semantic analysis")
                            .with_code("E5168")
                            .at(declaration.span),
                    );
                }
                if let Some(initializer) = initializer {
                    match self.functions.get(initializer) {
                        Some(shape)
                            if shape.origin
                                == (FunctionOrigin::GlobalInitializer {
                                    global: *definition,
                                })
                                && shape.captures.is_empty()
                                && shape.parameters.is_empty()
                                && shape.result == *ty => {}
                        Some(_) => self.diagnostics.push(
                            Diagnostic::error(
                                "MIR global initializer has an incompatible body signature",
                            )
                            .with_code("E5180")
                            .at(declaration.span),
                        ),
                        None => self.diagnostics.push(
                            Diagnostic::error("MIR global references a missing initializer body")
                                .with_code("E5181")
                                .at(declaration.span),
                        ),
                    }
                }
            }
        }
    }

    fn declaration_definition(
        &mut self,
        definition: DefId,
        expected: DefinitionKind,
        exported: bool,
        span: Span,
    ) {
        if !self.expect_definition_kind(definition, expected, "declaration", span) {
            return;
        }
        let semantic_exported =
            self.model.resolutions.definitions[definition.index()].visibility == Visibility::Public;
        if exported != semantic_exported {
            self.diagnostics.push(
                Diagnostic::error("MIR declaration visibility differs from name resolution")
                    .with_code("E5169")
                    .at(span),
            );
        }
    }

    fn declared_methods(&mut self, owner: DefId, methods: &[DefId], span: Span) {
        let mut seen = HashSet::new();
        for method in methods {
            if !seen.insert(*method) {
                self.duplicate_member("method", "E5170", span);
            }
            self.expect_definition_kind(*method, DefinitionKind::Method, "method", span);
            if self.function_definitions.get(method) != Some(&Some(owner)) {
                self.diagnostics.push(
                    Diagnostic::error("declared MIR method has no body owned by its nominal type")
                        .with_code("E5171")
                        .at(span),
                );
            }
        }
    }

    fn trait_methods(&mut self, methods: &[TraitMethodDeclaration], span: Span) {
        let mut seen = HashSet::new();
        for method in methods {
            if !seen.insert(method.definition) {
                self.duplicate_member("trait method", "E5172", span);
            }
            self.expect_definition_kind(
                method.definition,
                DefinitionKind::Method,
                "trait method",
                span,
            );
            for parameter in &method.parameters {
                self.ty(*parameter, span);
            }
            self.ty(method.result, span);
            self.function_signature(
                method.definition,
                &method.parameters,
                method.result,
                method.receiver.is_some(),
                "trait method",
                span,
            );
        }
    }

    pub(super) fn function_signature(
        &mut self,
        definition: DefId,
        parameters: &[TypeId],
        result: TypeId,
        allow_receiver_prefix: bool,
        description: &str,
        span: Span,
    ) {
        let Some(ty) = self.model.definition_types.get(&definition).copied() else {
            self.diagnostics.push(
                Diagnostic::error(format!("MIR {description} has no semantic signature"))
                    .with_code("E5173")
                    .at(span),
            );
            return;
        };
        match self.model.types.kind(ty) {
            TypeKind::Function {
                parameters: expected_parameters,
                result: expected_result,
            } if (expected_parameters == parameters
                || (allow_receiver_prefix
                    && expected_parameters
                        .get(1..)
                        .is_some_and(|tail| tail == parameters)))
                && *expected_result == result => {}
            _ => self.diagnostics.push(
                Diagnostic::error(format!(
                    "MIR {description} signature differs from semantic analysis"
                ))
                .with_code("E5174")
                .at(span),
            ),
        }
    }

    fn expect_definition_kind(
        &mut self,
        definition: DefId,
        expected: DefinitionKind,
        description: &str,
        span: Span,
    ) -> bool {
        let Some(record) = self.model.resolutions.definitions.get(definition.index()) else {
            self.invalid_id(description, definition.index(), span);
            return false;
        };
        if record.kind != expected {
            self.diagnostics.push(
                Diagnostic::error(format!("MIR {description} has the wrong semantic kind"))
                    .with_code("E5175")
                    .at(span),
            );
            return false;
        }
        true
    }

    fn expect_nominal_kind(
        &mut self,
        ty: TypeId,
        expected: DefinitionKind,
        description: &str,
        span: Span,
    ) {
        let TypeKind::Nominal { definition, .. } = self.model.types.kind(ty) else {
            self.diagnostics.push(
                Diagnostic::error(format!("MIR {description} is not a nominal type"))
                    .with_code("E5176")
                    .at(span),
            );
            return;
        };
        self.expect_definition_kind(*definition, expected, description, span);
    }

    fn expect_runtime_nominal(&mut self, ty: TypeId, span: Span) {
        let TypeKind::Nominal { definition, .. } = self.model.types.kind(ty) else {
            self.diagnostics.push(
                Diagnostic::error("MIR implementation target is not a nominal type")
                    .with_code("E5177")
                    .at(span),
            );
            return;
        };
        let Some(record) = self.model.resolutions.definitions.get(definition.index()) else {
            self.invalid_id("implementation target", definition.index(), span);
            return;
        };
        if !matches!(record.kind, DefinitionKind::Struct | DefinitionKind::Enum) {
            self.diagnostics.push(
                Diagnostic::error("MIR implementation target is not a struct or enum")
                    .with_code("E5178")
                    .at(span),
            );
        }
    }

    fn duplicate_member(&mut self, description: &str, code: &'static str, span: Span) {
        self.diagnostics.push(
            Diagnostic::error(format!("duplicate MIR {description} declaration"))
                .with_code(code)
                .at(span),
        );
    }
}
