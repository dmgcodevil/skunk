//! Stable compiler pipeline facade.
//!
//! Callers use this module rather than depending on individual phase ordering.
//! Phase ordering and internal specialization details are centralized here.

use crate::analysis::{check, model, resolver};
use crate::hir::{lower, validate};
use crate::source_map::SourceMap;
use crate::specialization::{self, convert, expand};
use crate::syntax;
use std::fmt::Debug;
use std::path::Path;

/// Opt-in console logger for compiler phase outputs.
///
/// The logger writes to stderr so dumps never become part of a compiled
/// program's stdout. Library callers that use the existing pipeline entry
/// points get a disabled logger and therefore retain the previous behavior.
#[derive(Debug, Default)]
pub struct CompilerLogger {
    enabled: bool,
    next_stage: usize,
}

impl CompilerLogger {
    /// Creates a logger that prints every compiler phase to stderr.
    pub fn console() -> Self {
        Self {
            enabled: true,
            next_stage: 1,
        }
    }

    /// Creates a logger that discards all phase output.
    pub fn disabled() -> Self {
        Self::default()
    }

    pub(crate) fn debug(&mut self, phase: &str, representation: &str, value: &dyn Debug) {
        if !self.enabled {
            return;
        }
        eprintln!(
            "\n[skunk debug {:02}] {} ({})",
            self.next_stage, phase, representation
        );
        eprintln!("{value:#?}");
        self.next_stage += 1;
    }

    pub(crate) fn text(&mut self, phase: &str, representation: &str, value: &str) {
        if !self.enabled {
            return;
        }
        eprintln!(
            "\n[skunk debug {:02}] {} ({})",
            self.next_stage, phase, representation
        );
        eprintln!("{value}");
        self.next_stage += 1;
    }
}

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
    let mut logger = CompilerLogger::disabled();
    check_source_with_logger(path, source, &mut logger)
}

/// Parses and checks source while reporting each produced representation.
pub fn check_source_with_logger(
    path: impl AsRef<Path>,
    source: &str,
    logger: &mut CompilerLogger,
) -> Result<CheckedProgram, String> {
    let mut sources = SourceMap::default();
    let file = sources
        .add_file(path.as_ref(), source)
        .map_err(|error| error.to_string())?;
    let module = syntax::parser::parse_module(&sources, file)
        .map_err(|diagnostics| render_diagnostics(diagnostics, Some(&sources)))?;
    check_syntax(module, Some(sources), logger)
}

/// Checks a complete multi-file program produced by the syntax loader.
pub fn check_loaded(program: syntax::loader::LoadedProgram) -> Result<CheckedProgram, String> {
    let mut logger = CompilerLogger::disabled();
    check_loaded_with_logger(program, &mut logger)
}

/// Checks a loaded program while reporting each produced representation.
pub fn check_loaded_with_logger(
    program: syntax::loader::LoadedProgram,
    logger: &mut CompilerLogger,
) -> Result<CheckedProgram, String> {
    check_syntax(program.module, Some(program.sources), logger)
}

/// Lowers a checked program into typed MIR and verifies its backend-facing
/// invariants. Native compilation consumes only this validated MIR plus the
/// semantic type and definition tables retained by `CheckedProgram`.
pub fn lower_to_mir(program: &CheckedProgram) -> Result<crate::mir::Module, String> {
    let mut logger = CompilerLogger::disabled();
    lower_to_mir_with_logger(program, &mut logger)
}

/// Lowers checked HIR to MIR while reporting the resulting representation.
pub fn lower_to_mir_with_logger(
    program: &CheckedProgram,
    logger: &mut CompilerLogger,
) -> Result<crate::mir::Module, String> {
    let mir = crate::mir::lower::lower(&program.hir, &program.semantics)
        .map_err(|diagnostics| render_diagnostics(diagnostics, program.sources.as_ref()))?;
    logger.debug("MIR lowering", "MIR", &mir);
    crate::mir::validate::validate(&mir, &program.semantics)
        .map_err(|diagnostics| render_diagnostics(diagnostics, program.sources.as_ref()))?;
    Ok(mir)
}

fn check_syntax(
    module: syntax::ast::Module,
    sources: Option<SourceMap>,
    logger: &mut CompilerLogger,
) -> Result<CheckedProgram, String> {
    logger.debug("source loading", "syntax AST", &module);
    let module = syntax::normalize::normalize(module)
        .map_err(|diagnostics| render_diagnostics(diagnostics, sources.as_ref()))?;
    logger.debug("syntax normalization", "syntax AST", &module);
    let initial_resolutions = resolver::resolve(&module)
        .map_err(|diagnostics| render_diagnostics(diagnostics, sources.as_ref()))?;
    logger.debug(
        "pre-expansion name resolution",
        "resolution tables",
        &initial_resolutions,
    );
    let file = module.span.file;
    let source_len = module.span.end as usize;
    let expansion_input = convert::to_tree(&module);
    logger.debug(
        "specialization conversion",
        "generic expansion tree",
        &expansion_input,
    );
    let prepared = expand::prepare_program(&expansion_input)?;
    logger.debug(
        "generic specialization",
        "specialized expansion tree",
        &prepared,
    );
    check::check(&prepared)?;
    let prepared_syntax = convert::from_tree(&prepared, file, source_len)
        .map_err(|diagnostic| diagnostic.to_string())?;
    logger.debug(
        "specialization conversion",
        "specialized syntax AST",
        &prepared_syntax,
    );
    let prepared_resolutions = resolver::resolve(&prepared_syntax)
        .map_err(|diagnostics| render_diagnostics(diagnostics, sources.as_ref()))?;
    logger.debug(
        "post-expansion name resolution",
        "resolution tables",
        &prepared_resolutions,
    );
    let mut semantic_model = model::analyze_declarations(&prepared_syntax, prepared_resolutions)
        .map_err(|diagnostics| render_diagnostics(diagnostics, sources.as_ref()))?;
    logger.debug("semantic analysis", "semantic model", &semantic_model);
    let hir = lower::lower(&prepared_syntax, &mut semantic_model)
        .map_err(|diagnostics| render_diagnostics(diagnostics, sources.as_ref()))?;
    logger.debug("HIR lowering", "typed HIR", &hir);
    validate::validate(&hir, &semantic_model)
        .map_err(|diagnostics| render_diagnostics(diagnostics, sources.as_ref()))?;
    let specializations = specialization::seal(&hir, &semantic_model)
        .map_err(|diagnostics| render_diagnostics(diagnostics, sources.as_ref()))?;
    logger.debug(
        "specialization seal",
        "specialization metadata",
        &specializations,
    );
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
