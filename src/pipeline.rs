//! Stable compiler pipeline facade.
//!
//! Callers use this module rather than depending on individual phase ordering.
//! During the migration, the final compatibility conversion is deliberately
//! centralized here. It will be removed when lowering consumes typed HIR.

use crate::ast as legacy;
use crate::ids::FileId;
use crate::source_map::SourceMap;
use crate::{
    hir_lowering, hir_validation, monomorphize, resolver, semantics, specialization, syntax,
    type_checker,
};
use std::path::Path;

/// A program that passed every front-end phase and is ready for lowering.
///
/// The typed representation is the authoritative compiler state. The legacy
/// node is retained privately and only exposed to the LLVM compatibility
/// backend while that backend is being migrated to HIR.
#[derive(Debug)]
pub struct CheckedProgram {
    pub hir: crate::hir::Module,
    pub semantics: semantics::SemanticModel,
    pub specializations: specialization::SpecializationSet,
    pub sources: Option<SourceMap>,
    legacy_codegen: legacy::Node,
}

impl CheckedProgram {
    pub(crate) fn legacy_codegen(&self) -> &legacy::Node {
        &self.legacy_codegen
    }
}

/// Converts a loaded program through the new syntax boundary, then prepares
/// and checks the temporary legacy backend input.
pub fn check(program: &legacy::Node) -> Result<CheckedProgram, String> {
    let module = syntax::from_legacy(program, FileId::new(0), 0)
        .map_err(|diagnostic| diagnostic.to_string())?;
    check_syntax(module, None)
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

fn check_syntax(
    module: syntax::ast::Module,
    sources: Option<SourceMap>,
) -> Result<CheckedProgram, String> {
    resolver::resolve(&module)
        .map_err(|diagnostics| render_diagnostics(diagnostics, sources.as_ref()))?;
    let file = module.span.file;
    let source_len = module.span.end as usize;
    let compatibility_input = syntax::to_legacy(&module);
    let prepared = monomorphize::prepare_program(&compatibility_input)?;
    type_checker::check(&prepared)?;
    let prepared_syntax = syntax::from_legacy(&prepared, file, source_len)
        .map_err(|diagnostic| diagnostic.to_string())?;
    let prepared_resolutions = resolver::resolve(&prepared_syntax)
        .map_err(|diagnostics| render_diagnostics(diagnostics, sources.as_ref()))?;
    let mut semantic_model =
        semantics::analyze_declarations(&prepared_syntax, prepared_resolutions)
            .map_err(|diagnostics| render_diagnostics(diagnostics, sources.as_ref()))?;
    let hir = hir_lowering::lower(&prepared_syntax, &mut semantic_model)
        .map_err(|diagnostics| render_diagnostics(diagnostics, sources.as_ref()))?;
    hir_validation::validate(&hir, &semantic_model)
        .map_err(|diagnostics| render_diagnostics(diagnostics, sources.as_ref()))?;
    let specializations = specialization::seal(&hir, &semantic_model)
        .map_err(|diagnostics| render_diagnostics(diagnostics, sources.as_ref()))?;
    Ok(CheckedProgram {
        hir,
        semantics: semantic_model,
        specializations,
        sources,
        legacy_codegen: prepared,
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
