use super::*;
use crate::analysis::{model, resolver};
use crate::hir;
use crate::ids::MirBlockId;

pub(super) fn lower_source(source: &str) -> (Module, model::SemanticModel) {
    let syntax = crate::syntax::parser::parse_test_module(source);
    let resolutions = resolver::resolve(&syntax).unwrap();
    let mut semantics = model::analyze_declarations(&syntax, resolutions).unwrap();
    let hir = hir::lower::lower(&syntax, &mut semantics).unwrap();
    hir::validate::validate(&hir, &semantics).unwrap();
    let mir = lower::lower(&hir, &semantics).unwrap();
    validate::validate(&mir, &semantics).unwrap();
    (mir, semantics)
}

pub(super) fn function_named<'a>(
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

mod basic;
mod closures;
mod control_flow;
mod places;
