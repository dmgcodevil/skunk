//! Source-level syntax.
//!
//! Syntax nodes describe only what the programmer wrote. They deliberately do
//! not contain resolved declarations, inferred types, or backend-specific data.

pub mod ast;
pub mod loader;
pub mod normalize;
pub mod parser;
pub(crate) mod pest;
