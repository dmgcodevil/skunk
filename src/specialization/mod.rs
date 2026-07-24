//! Verifies and records the concrete program produced by generic expansion.
//!
//! Generic discovery happens before HIR. This phase is the stable post-HIR
//! contract: every runtime declaration and local type must be concrete before
//! code generation.

use crate::analysis::model::SemanticModel;
use crate::analysis::types::TypeKind;
use crate::diagnostic::Diagnostic;
use crate::hir;
use crate::ids::{DefId, TypeId};
use crate::source_map::Span;
use std::collections::HashSet;

pub(crate) mod convert;
pub(crate) mod expand;
pub(crate) mod tree;

#[derive(Debug)]
pub struct SpecializationSet {
    pub definitions: Vec<DefId>,
    pub types: Vec<TypeId>,
}

pub fn seal(
    module: &hir::Module,
    model: &SemanticModel,
) -> Result<SpecializationSet, Vec<Diagnostic>> {
    let fallback_span = module
        .items
        .first()
        .map(|item| item.span)
        .unwrap_or_else(|| Span::empty(crate::ids::FileId::new(0)));
    let mut definitions = Vec::new();
    let mut roots = Vec::new();

    for item in &module.items {
        if let Some(definition) = item.definition {
            definitions.push(definition);
        }
        collect_item_types(item, &mut definitions, &mut roots);
    }
    roots.extend(model.local_types.values().copied());

    definitions.sort_unstable_by_key(|definition| definition.index());
    definitions.dedup();
    roots.sort_unstable_by_key(|ty| ty.index());
    roots.dedup();

    let mut checker = ConcreteTypeChecker {
        model,
        span: fallback_span,
        visited: HashSet::new(),
        diagnostics: Vec::new(),
    };
    for ty in roots {
        checker.check(ty);
    }
    if checker.diagnostics.is_empty() {
        let mut types = checker.visited.into_iter().collect::<Vec<_>>();
        types.sort_unstable_by_key(|ty| ty.index());
        Ok(SpecializationSet { definitions, types })
    } else {
        Err(checker.diagnostics)
    }
}

fn collect_item_types(item: &hir::Item, definitions: &mut Vec<DefId>, types: &mut Vec<TypeId>) {
    match &item.kind {
        hir::ItemKind::Struct(declaration) => {
            types.extend(declaration.fields.iter().map(|field| field.ty));
            for method in &declaration.methods {
                collect_function_types(method, definitions, types);
            }
        }
        hir::ItemKind::Enum(declaration) => {
            for variant in &declaration.variants {
                types.extend(variant.payload.iter().copied());
            }
            for method in &declaration.methods {
                collect_function_types(method, definitions, types);
            }
        }
        hir::ItemKind::Trait(declaration) | hir::ItemKind::Shape(declaration) => {
            definitions.extend(declaration.supertraits.iter().copied());
            for method in &declaration.methods {
                definitions.push(method.definition);
                types.extend(method.parameters.iter().copied());
                types.push(method.result);
            }
        }
        hir::ItemKind::Implementation { traits, target } => {
            types.extend(traits.iter().copied());
            types.push(*target);
        }
        hir::ItemKind::Function(function) => collect_function_types(function, definitions, types),
        hir::ItemKind::ExternFunction(signature) => {
            types.extend(signature.parameters.iter().copied());
            types.push(signature.result);
        }
        hir::ItemKind::Global(global) => types.push(global.ty),
        hir::ItemKind::Test(_) | hir::ItemKind::Statement(_) => {}
    }
}

fn collect_function_types(
    function: &hir::Function,
    definitions: &mut Vec<DefId>,
    types: &mut Vec<TypeId>,
) {
    if let Some(definition) = function.definition {
        definitions.push(definition);
    }
    types.extend(function.parameters.iter().map(|parameter| parameter.ty));
    types.push(function.result);
}

struct ConcreteTypeChecker<'a> {
    model: &'a SemanticModel,
    span: Span,
    visited: HashSet<TypeId>,
    diagnostics: Vec<Diagnostic>,
}

impl ConcreteTypeChecker<'_> {
    fn check(&mut self, ty: TypeId) {
        if !self.visited.insert(ty) {
            return;
        }
        match self.model.types.kind(ty) {
            TypeKind::GenericParameter(definition) => {
                let name = self
                    .model
                    .resolutions
                    .definitions
                    .get(definition.index())
                    .map(|definition| definition.name.as_str())
                    .unwrap_or("<unknown>");
                self.diagnostics.push(
                    Diagnostic::error(format!(
                        "unspecialized generic parameter `{name}` reached backend HIR"
                    ))
                    .with_code("E4100")
                    .at(self.span),
                );
            }
            TypeKind::Nominal { arguments, .. }
            | TypeKind::Union(arguments)
            | TypeKind::Intersection(arguments) => {
                for argument in arguments {
                    self.check(*argument);
                }
            }
            TypeKind::Const(inner) | TypeKind::Pointer(inner) | TypeKind::Slice(inner) => {
                self.check(*inner)
            }
            TypeKind::Array { element, .. } => self.check(*element),
            TypeKind::Reference { target, .. } => self.check(*target),
            TypeKind::Function { parameters, result } => {
                for parameter in parameters {
                    self.check(*parameter);
                }
                self.check(*result);
            }
            TypeKind::Error | TypeKind::Never | TypeKind::Builtin(_) | TypeKind::Intrinsic(_) => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_a_concrete_runtime_surface() {
        let source = "function id(value: int): int { return value; }";
        let module = crate::syntax::parser::parse_test_module(source);
        let resolutions = crate::analysis::resolver::resolve(&module).unwrap();
        let mut model = crate::analysis::model::analyze_declarations(&module, resolutions).unwrap();
        let hir = crate::hir::lower::lower(&module, &mut model).unwrap();

        let specializations = seal(&hir, &model).unwrap();
        assert!(!specializations.definitions.is_empty());
    }
}
