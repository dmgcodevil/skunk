//! Source-level syntax.
//!
//! Syntax nodes describe only what the programmer wrote. They deliberately do
//! not contain resolved declarations, inferred types, or backend-specific data.

pub mod ast;
mod legacy_bridge;
pub mod parser;

pub use legacy_bridge::{from_legacy, to_legacy};
