//! Name resolution and semantic type analysis.

pub(crate) mod check;
pub mod model;
pub mod resolver;
pub mod types;

pub use crate::syntax::ast::BuiltinType;
pub use model::{analyze_declarations, SemanticModel};
pub use resolver::{resolve, Resolutions};
pub use types::{TypeKind, TypeStore};
