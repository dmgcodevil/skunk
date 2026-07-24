//! Stable compiler pipeline facade.
//!
//! Callers use this module rather than depending on individual phase ordering.
//! Phase ordering and internal specialization details are centralized here.

use crate::analysis::{check, model, resolver};
use crate::hir::{lower, validate};
use crate::source_map::SourceMap;
use crate::specialization::{self, convert, expand};
use crate::syntax;
use std::path::Path;

/// A program that passed every front-end phase and is ready for lowering.
///
/// HIR and semantic tables are retained so callers can inspect the checked
/// program; native code generation first lowers them to validated MIR.
#[derive(Debug)]
pub struct CheckedProgram {
    pub hir: crate::hir::Module,
    pub semantics: model::SemanticModel,
    pub specializations: specialization::SpecializationSet,
    pub sources: Option<SourceMap>,
}

/// Parses and checks one source file through the public syntax boundary.
pub fn check_source(path: impl AsRef<Path>, source: &str) -> Result<CheckedProgram, String> {
    let mut sources = SourceMap::default();
    let file = sources
        .add_file(path.as_ref(), source)
        .map_err(|error| error.to_string())?;
    let module = syntax::parser::parse_module(&sources, file)
        .map_err(|diagnostics| render_diagnostics(diagnostics, Some(&sources)))?;
    check_syntax(module, Some(sources))
}

/// Checks a complete multi-file program produced by the syntax loader.
pub fn check_loaded(program: syntax::loader::LoadedProgram) -> Result<CheckedProgram, String> {
    check_syntax(program.module, Some(program.sources))
}

/// Lowers a checked program into typed MIR and verifies its backend-facing
/// invariants. Native compilation consumes only this validated MIR plus the
/// semantic type and definition tables retained by `CheckedProgram`.
pub fn lower_to_mir(program: &CheckedProgram) -> Result<crate::mir::Module, String> {
    let mir = crate::mir::lower::lower(&program.hir, &program.semantics)
        .map_err(|diagnostics| render_diagnostics(diagnostics, program.sources.as_ref()))?;
    crate::mir::validate::validate(&mir, &program.semantics)
        .map_err(|diagnostics| render_diagnostics(diagnostics, program.sources.as_ref()))?;
    Ok(mir)
}

fn check_syntax(
    module: syntax::ast::Module,
    sources: Option<SourceMap>,
) -> Result<CheckedProgram, String> {
    let module = syntax::normalize::normalize(module)
        .map_err(|diagnostics| render_diagnostics(diagnostics, sources.as_ref()))?;
    resolver::resolve(&module)
        .map_err(|diagnostics| render_diagnostics(diagnostics, sources.as_ref()))?;
    let file = module.span.file;
    let source_len = module.span.end as usize;
    let expansion_input = convert::to_tree(&module);
    let prepared = expand::prepare_program(&expansion_input)?;
    check::check(&prepared)?;
    let prepared_syntax = convert::from_tree(&prepared, file, source_len)
        .map_err(|diagnostic| diagnostic.to_string())?;
    let prepared_resolutions = resolver::resolve(&prepared_syntax)
        .map_err(|diagnostics| render_diagnostics(diagnostics, sources.as_ref()))?;
    let mut semantic_model = model::analyze_declarations(&prepared_syntax, prepared_resolutions)
        .map_err(|diagnostics| render_diagnostics(diagnostics, sources.as_ref()))?;
    let hir = lower::lower(&prepared_syntax, &mut semantic_model)
        .map_err(|diagnostics| render_diagnostics(diagnostics, sources.as_ref()))?;
    validate::validate(&hir, &semantic_model)
        .map_err(|diagnostics| render_diagnostics(diagnostics, sources.as_ref()))?;
    let specializations = specialization::seal(&hir, &semantic_model)
        .map_err(|diagnostics| render_diagnostics(diagnostics, sources.as_ref()))?;
    Ok(CheckedProgram {
        hir,
        semantics: semantic_model,
        specializations,
        sources,
    })
}

fn render_diagnostics(
    diagnostics: Vec<crate::diagnostic::Diagnostic>,
    sources: Option<&SourceMap>,
) -> String {
    diagnostics
        .into_iter()
        .map(|diagnostic| match sources {
            Some(sources) => diagnostic.render(sources),
            None => diagnostic.to_string(),
        })
        .collect::<Vec<_>>()
        .join("\n")
}
