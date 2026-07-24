//! Skunk compiler library.
//!
//! The command-line executable is intentionally a thin client of this crate so
//! parsing, checking, and code generation can also be embedded by tests and
//! future tooling without depending on CLI state.

pub mod ast;
pub mod compiler;
pub mod diagnostic;
pub mod hir;
pub mod hir_lowering;
pub mod hir_validation;
pub mod ids;
pub mod intrinsics;
pub mod manifest;
pub mod monomorphize;
pub mod parser;
pub mod pipeline;
pub mod resolver;
pub mod sdk;
pub mod semantic_types;
pub mod semantics;
pub mod source;
pub mod source_map;
pub mod specialization;
pub mod syntax;
pub mod testing;
pub mod type_checker;
