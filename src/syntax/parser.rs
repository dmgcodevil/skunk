//! Public syntax parser boundary.
//!
//! Pest and its generated `Rule` type remain implementation details. The
//! temporary legacy conversion inside this module is removed once all parser
//! actions construct category-specific syntax nodes directly.

use super::ast::Module;
use crate::diagnostic::Diagnostic;
use crate::ids::FileId;
use crate::source_map::SourceMap;

pub fn parse_module(sources: &SourceMap, file: FileId) -> Result<Module, Vec<Diagnostic>> {
    let source = sources.source(file).ok_or_else(|| {
        vec![
            Diagnostic::error(format!("unknown source file id {}", file.index()))
                .with_code("E0002"),
        ]
    })?;

    let legacy = crate::ast::try_parse(source)
        .map_err(|message| vec![Diagnostic::error(message).with_code("E1000")])?;
    super::from_legacy(&legacy, file, source.len()).map_err(|diagnostic| vec![diagnostic])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::syntax::ast::{TopLevelKind, Visibility};

    #[test]
    fn produces_category_specific_module_nodes() {
        let source = r#"
            module geometry;
            export struct Point { x: int; y: int; }
            function sum(point: Point): int { return point.x + point.y; }
        "#;
        let mut sources = SourceMap::default();
        let file = sources.add_file("geometry.skunk", source).unwrap();
        let module = parse_module(&sources, file).unwrap();

        assert_eq!(module.name.as_ref().unwrap().qualified_name(), "geometry");
        assert!(matches!(
            module.entries[0],
            crate::syntax::ast::TopLevel {
                visibility: Visibility::Public,
                kind: TopLevelKind::Struct(_),
                ..
            }
        ));
        assert!(matches!(module.entries[1].kind, TopLevelKind::Function(_)));
    }

    #[test]
    fn oversized_integer_is_a_diagnostic_not_a_panic() {
        let source = "function main(): void { print(999999999999999999999999999999); }";
        let mut sources = SourceMap::default();
        let file = sources.add_file("large.skunk", source).unwrap();
        let diagnostics = parse_module(&sources, file).unwrap_err();

        assert!(diagnostics[0]
            .message
            .contains("outside the supported 64-bit range"));
    }
}
