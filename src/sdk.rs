//! Embedded SDK: standard library modules shipped inside the compiler binary.
//!
//! Sources under `lib/std/` are embedded with `include_str!` at build time and
//! materialized into `$SKUNK_HOME/lib/<version>/std/` on first use, so an
//! installed `skunk` binary is fully self-contained. The `std.` import prefix
//! is reserved: `import std.math;` always resolves into the SDK, never into
//! project files.

use crate::compiler::{skunk_home, write_if_changed};
use std::fs;
use std::path::PathBuf;

/// Every embedded standard library module, as (relative path, contents).
const STD_MODULES: &[(&str, &str)] = &[("std/math.skunk", include_str!("../lib/std/math.skunk"))];

/// Returns true when an import name is reserved for the standard library.
pub fn is_std_module(module_name: &str) -> bool {
    module_name == "std" || module_name.starts_with("std.")
}

/// The on-disk root the embedded SDK is materialized into, keyed by compiler
/// version so upgrades never mix stdlib versions.
pub fn std_lib_root() -> PathBuf {
    skunk_home().join("lib").join(env!("CARGO_PKG_VERSION"))
}

/// Materializes all embedded standard library modules and returns the SDK
/// root. Idempotent and cheap: files are only rewritten when their contents
/// changed.
pub fn ensure_std_lib() -> Result<PathBuf, String> {
    let root = std_lib_root();
    for (relative_path, contents) in STD_MODULES {
        let path = root.join(relative_path);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .map_err(|err| format!("failed to create `{}`: {}", parent.display(), err))?;
        }
        write_if_changed(&path, contents)?;
    }
    Ok(root)
}

/// Resolves a `std.*` module name to its materialized SDK file.
pub fn std_module_path(module_name: &str) -> Result<PathBuf, String> {
    let root = ensure_std_lib()?;
    let mut path = root.clone();
    for segment in module_name.split('.') {
        path.push(segment);
    }
    path.set_extension("skunk");
    if !path.exists() {
        return Err(format!(
            "unknown standard library module `{}` (the `std.` prefix is reserved for the SDK; searched `{}`)",
            module_name,
            path.display()
        ));
    }
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn std_prefix_detection() {
        assert!(is_std_module("std"));
        assert!(is_std_module("std.math"));
        assert!(is_std_module("std.collections.list"));
        assert!(!is_std_module("stdlib"));
        assert!(!is_std_module("app.std"));
    }

    #[test]
    fn materializes_and_resolves_std_math() {
        let path = std_module_path("std.math").unwrap();
        assert!(path.exists());
        let contents = fs::read_to_string(&path).unwrap();
        assert!(contents.contains("module std.math;"));
    }

    #[test]
    fn unknown_std_module_errors() {
        let error = std_module_path("std.no_such_module").unwrap_err();
        assert!(error.contains("unknown standard library module"));
    }
}
