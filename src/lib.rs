//! Skunk compiler library.
//!
//! The command-line executable is intentionally a thin client of this crate so
//! parsing, checking, and code generation can also be embedded by tests and
//! future tooling without depending on CLI state.

pub mod analysis;
pub mod backend;
pub mod diagnostic;
pub mod hir;
pub mod ids;
pub mod intrinsics;
pub mod manifest;
pub mod mir;
pub mod pipeline;
pub mod sdk;
pub mod source_map;
pub mod specialization;
pub mod syntax;
pub mod testing;

pub use syntax::loader as source;
