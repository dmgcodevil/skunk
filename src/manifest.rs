//! `skunk.toml` project manifest.
//!
//! The manifest is a deliberately small TOML subset: `[section]` headers and
//! `key = value` pairs where a value is a quoted string, a boolean, or a
//! single-line array of quoted strings. That covers the whole surface:
//!
//! ```toml
//! [package]
//! name = "demo"
//! entry = "src/main.skunk"
//!
//! [build]
//! optimize = true
//! libraries = ["sqlite3"]
//! frameworks = ["Cocoa"]
//! ```

use std::fs;
use std::path::{Path, PathBuf};

pub const MANIFEST_FILE: &str = "skunk.toml";

#[derive(Debug, Clone, PartialEq)]
pub struct Manifest {
    pub name: String,
    pub entry: PathBuf,
    pub optimize: bool,
    pub libraries: Vec<String>,
    pub frameworks: Vec<String>,
}

impl Default for Manifest {
    fn default() -> Self {
        Manifest {
            name: "app".to_string(),
            entry: PathBuf::from("src/main.skunk"),
            optimize: true,
            libraries: Vec::new(),
            frameworks: Vec::new(),
        }
    }
}

/// Loads and parses `skunk.toml` from the given path.
pub fn load_manifest(path: &Path) -> Result<Manifest, String> {
    let contents = fs::read_to_string(path)
        .map_err(|err| format!("failed to read `{}`: {}", path.display(), err))?;
    parse_manifest(&contents)
}

/// Parses manifest contents. See the module docs for the accepted subset.
pub fn parse_manifest(contents: &str) -> Result<Manifest, String> {
    let mut manifest = Manifest::default();
    let mut section = String::new();

    for (index, raw_line) in contents.lines().enumerate() {
        let line_number = index + 1;
        let line = strip_comment(raw_line).trim().to_string();
        if line.is_empty() {
            continue;
        }
        if line.starts_with('[') {
            if !line.ends_with(']') {
                return Err(format!(
                    "skunk.toml line {}: malformed section header `{}`",
                    line_number, line
                ));
            }
            section = line[1..line.len() - 1].trim().to_string();
            continue;
        }
        let Some((key, value)) = line.split_once('=') else {
            return Err(format!(
                "skunk.toml line {}: expected `key = value`, found `{}`",
                line_number, line
            ));
        };
        let key = key.trim();
        let value = value.trim();
        match (section.as_str(), key) {
            ("package", "name") => manifest.name = parse_string(value, line_number)?,
            ("package", "entry") => {
                manifest.entry = PathBuf::from(parse_string(value, line_number)?)
            }
            ("build", "optimize") => manifest.optimize = parse_bool(value, line_number)?,
            ("build", "libraries") => manifest.libraries = parse_string_array(value, line_number)?,
            ("build", "frameworks") => {
                manifest.frameworks = parse_string_array(value, line_number)?
            }
            // Unknown keys are ignored so newer manifests still load in
            // older compilers.
            _ => {}
        }
    }

    if manifest.name.is_empty() {
        return Err("skunk.toml: `package.name` must not be empty".to_string());
    }
    Ok(manifest)
}

/// Removes a `#` comment unless the `#` appears inside a quoted string.
fn strip_comment(line: &str) -> &str {
    let mut in_string = false;
    for (index, ch) in line.char_indices() {
        match ch {
            '"' => in_string = !in_string,
            '#' if !in_string => return &line[..index],
            _ => {}
        }
    }
    line
}

fn parse_string(value: &str, line_number: usize) -> Result<String, String> {
    let inner = value
        .strip_prefix('"')
        .and_then(|v| v.strip_suffix('"'))
        .ok_or_else(|| {
            format!(
                "skunk.toml line {}: expected a quoted string, found `{}`",
                line_number, value
            )
        })?;
    if inner.contains('"') {
        return Err(format!(
            "skunk.toml line {}: escaped quotes are not supported",
            line_number
        ));
    }
    Ok(inner.to_string())
}

fn parse_bool(value: &str, line_number: usize) -> Result<bool, String> {
    match value {
        "true" => Ok(true),
        "false" => Ok(false),
        other => Err(format!(
            "skunk.toml line {}: expected `true` or `false`, found `{}`",
            line_number, other
        )),
    }
}

fn parse_string_array(value: &str, line_number: usize) -> Result<Vec<String>, String> {
    let inner = value
        .strip_prefix('[')
        .and_then(|v| v.strip_suffix(']'))
        .ok_or_else(|| {
            format!(
                "skunk.toml line {}: expected a single-line array like [\"name\"], found `{}`",
                line_number, value
            )
        })?;
    let inner = inner.trim();
    if inner.is_empty() {
        return Ok(Vec::new());
    }
    inner
        .split(',')
        .map(|item| parse_string(item.trim(), line_number))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_full_manifest() {
        let manifest = parse_manifest(
            r#"
            # A demo project
            [package]
            name = "demo"
            entry = "src/app.skunk"

            [build]
            optimize = false
            libraries = ["sqlite3", "z"]
            frameworks = ["Metal"]
            "#,
        )
        .unwrap();

        assert_eq!(manifest.name, "demo");
        assert_eq!(manifest.entry, PathBuf::from("src/app.skunk"));
        assert!(!manifest.optimize);
        assert_eq!(manifest.libraries, vec!["sqlite3", "z"]);
        assert_eq!(manifest.frameworks, vec!["Metal"]);
    }

    #[test]
    fn defaults_apply_when_keys_are_missing() {
        let manifest = parse_manifest("[package]\nname = \"tiny\"\n").unwrap();

        assert_eq!(manifest.name, "tiny");
        assert_eq!(manifest.entry, PathBuf::from("src/main.skunk"));
        assert!(manifest.optimize);
        assert!(manifest.libraries.is_empty());
        assert!(manifest.frameworks.is_empty());
    }

    #[test]
    fn ignores_comments_and_unknown_keys() {
        let manifest = parse_manifest(
            "[package]\nname = \"x\" # inline comment\nfuture_key = \"whatever\"\n",
        )
        .unwrap();

        assert_eq!(manifest.name, "x");
    }

    #[test]
    fn rejects_unquoted_strings() {
        assert!(parse_manifest("[package]\nname = demo\n").is_err());
    }

    #[test]
    fn rejects_malformed_lines() {
        assert!(parse_manifest("[package]\nname\n").is_err());
    }

    #[test]
    fn parses_empty_array() {
        let manifest = parse_manifest("[build]\nlibraries = []\n").unwrap();
        assert!(manifest.libraries.is_empty());
    }
}
