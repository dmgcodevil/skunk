use crate::ast::{self, Node};
use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

pub fn load_program(entry_path: &Path) -> Result<Node, String> {
    let entry_path = fs::canonicalize(entry_path)
        .map_err(|err| format!("failed to resolve `{}`: {}", entry_path.display(), err))?;
    let module_root = entry_path
        .parent()
        .ok_or_else(|| format!("`{}` has no parent directory", entry_path.display()))?
        .to_path_buf();
    let mut loader = ProgramLoader::new(module_root);
    let mut statements = loader.load_file(&entry_path, None, true)?;
    statements.push(Node::EOI);
    Ok(Node::Program { statements })
}

struct ProgramLoader {
    module_root: PathBuf,
    visited: HashSet<PathBuf>,
    loading: HashSet<PathBuf>,
}

impl ProgramLoader {
    fn new(module_root: PathBuf) -> Self {
        Self {
            module_root,
            visited: HashSet::new(),
            loading: HashSet::new(),
        }
    }

    fn load_file(
        &mut self,
        file_path: &Path,
        expected_module: Option<&str>,
        is_entry: bool,
    ) -> Result<Vec<Node>, String> {
        let file_path = fs::canonicalize(file_path)
            .map_err(|err| format!("failed to resolve `{}`: {}", file_path.display(), err))?;

        if self.visited.contains(&file_path) {
            return Ok(Vec::new());
        }

        if !self.loading.insert(file_path.clone()) {
            return Err(format!(
                "cyclic import detected while loading `{}`",
                file_path.display()
            ));
        }

        let contents = fs::read_to_string(&file_path)
            .map_err(|err| format!("failed to read `{}`: {}", file_path.display(), err))?;
        let program = ast::parse(&contents);
        let Node::Program { statements } = program else {
            unreachable!("parse always returns a program node")
        };

        let declared_module = extract_module_name(&statements)?;
        if let Some(expected_module) = expected_module {
            match declared_module.as_deref() {
                Some(actual) if actual == expected_module => {}
                Some(actual) => {
                    return Err(format!(
                        "module declaration mismatch in `{}`: expected `{}`, found `{}`",
                        file_path.display(),
                        expected_module,
                        actual
                    ))
                }
                None => {
                    return Err(format!(
                        "imported file `{}` must declare `module {};`",
                        file_path.display(),
                        expected_module
                    ))
                }
            }
        } else if !is_entry && declared_module.is_none() {
            return Err(format!(
                "imported file `{}` is missing a module declaration",
                file_path.display()
            ));
        }

        let mut output = Vec::new();
        for statement in statements {
            match statement {
                Node::Module { .. } => {}
                Node::Import { name } => {
                    let import_path = self.module_path(&name);
                    let imported = self.load_file(&import_path, Some(&name), false)?;
                    output.extend(imported);
                }
                Node::EOI => {}
                other => output.push(other),
            }
        }

        self.loading.remove(&file_path);
        self.visited.insert(file_path);
        Ok(output)
    }

    fn module_path(&self, module_name: &str) -> PathBuf {
        let mut path = self.module_root.clone();
        for segment in module_name.split('.') {
            path.push(segment);
        }
        path.set_extension("skunk");
        path
    }
}

fn extract_module_name(statements: &[Node]) -> Result<Option<String>, String> {
    let mut module_name = None;
    for statement in statements {
        if let Node::Module { name } = statement {
            if module_name.is_some() {
                return Err("a file may declare at most one module".to_string());
            }
            module_name = Some(name.clone());
        }
    }
    Ok(module_name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    use uuid::Uuid;

    fn write_file(path: &Path, contents: &str) {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(path, contents).unwrap();
    }

    #[test]
    fn loads_imported_module_before_entry_statements() {
        let root = env::temp_dir().join(format!("skunk_modules_{}", Uuid::new_v4()));
        let entry = root.join("main.skunk");
        let module = root.join("std").join("math.skunk");

        write_file(
            &module,
            r#"
            module std.math;

            function inc(n: int): int {
                return n + 1;
            }
            "#,
        );
        write_file(
            &entry,
            r#"
            import std.math;

            function main(): void {
                print(inc(6));
            }
            "#,
        );

        let loaded = load_program(&entry).unwrap();
        let Node::Program { statements } = loaded else {
            panic!("expected program node");
        };

        assert!(statements.iter().any(|statement| matches!(
            statement,
            Node::FunctionDeclaration { name, .. } if name == "inc"
        )));
        assert!(statements.iter().any(|statement| matches!(
            statement,
            Node::FunctionDeclaration { name, .. } if name == "main"
        )));

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn rejects_module_name_mismatch() {
        let root = env::temp_dir().join(format!("skunk_modules_{}", Uuid::new_v4()));
        let entry = root.join("main.skunk");
        let module = root.join("std").join("math.skunk");

        write_file(
            &module,
            r#"
            module std.wrong;

            function inc(n: int): int {
                return n + 1;
            }
            "#,
        );
        write_file(
            &entry,
            r#"
            import std.math;

            function main(): void {}
            "#,
        );

        let err = load_program(&entry).unwrap_err();
        assert!(err.contains("module declaration mismatch"));

        let _ = fs::remove_dir_all(root);
    }
}
