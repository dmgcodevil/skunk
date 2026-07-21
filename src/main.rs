mod ast;
mod compiler;
mod manifest;
mod monomorphize;
mod parser;
mod sdk;
mod source;
mod testing;
mod type_checker;
use colored::*;
use std::env;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const USAGE: &str = "Usage: skunk <file_path>\n       skunk run <file_path>\n       skunk compile <file_path> [output_path]\n       skunk test [file_path] [--filter <substring>]\n       skunk build\n       skunk new <project_name>\n       skunk --version\n       skunk versions\n       skunk use <version>";

#[derive(Debug, PartialEq)]
enum CommandKind {
    Run {
        source: String,
    },
    Compile {
        source: String,
        output: Option<PathBuf>,
    },
    Test {
        source: Option<String>,
        filter: Option<String>,
    },
    Build,
    New {
        name: String,
    },
    Version,
    Versions,
    Use {
        version: String,
    },
}

/// Parses the command line into a high-level command.
fn parse_cli(args: &[String]) -> Result<CommandKind, String> {
    if args.len() < 2 {
        return Err(USAGE.to_string());
    }
    match args[1].as_str() {
        "run" => {
            if args.len() != 3 {
                Err(USAGE.to_string())
            } else {
                Ok(CommandKind::Run {
                    source: args[2].clone(),
                })
            }
        }
        "compile" => {
            if args.len() < 3 || args.len() > 4 {
                Err(USAGE.to_string())
            } else {
                Ok(CommandKind::Compile {
                    source: args[2].clone(),
                    output: args.get(3).map(PathBuf::from),
                })
            }
        }
        "test" => {
            let mut source = None;
            let mut filter = None;
            let mut rest = args[2..].iter();
            while let Some(arg) = rest.next() {
                if arg == "--filter" {
                    match rest.next() {
                        Some(value) => filter = Some(value.clone()),
                        None => return Err("--filter expects a value".to_string()),
                    }
                } else if source.is_none() {
                    source = Some(arg.clone());
                } else {
                    return Err(USAGE.to_string());
                }
            }
            Ok(CommandKind::Test { source, filter })
        }
        "build" => {
            if args.len() != 2 {
                Err(USAGE.to_string())
            } else {
                Ok(CommandKind::Build)
            }
        }
        "new" => {
            if args.len() != 3 {
                Err(USAGE.to_string())
            } else {
                Ok(CommandKind::New {
                    name: args[2].clone(),
                })
            }
        }
        "--version" | "-V" | "version" => {
            if args.len() != 2 {
                Err(USAGE.to_string())
            } else {
                Ok(CommandKind::Version)
            }
        }
        "versions" => {
            if args.len() != 2 {
                Err(USAGE.to_string())
            } else {
                Ok(CommandKind::Versions)
            }
        }
        "use" => {
            if args.len() != 3 {
                Err(USAGE.to_string())
            } else {
                Ok(CommandKind::Use {
                    version: args[2].clone(),
                })
            }
        }
        _ if args.len() == 2 => Ok(CommandKind::Run {
            source: args[1].clone(),
        }),
        _ => Err(USAGE.to_string()),
    }
}

/// Chooses the default binary path for `skunk compile` when the caller does not
/// provide one explicitly.
fn default_output_path(source_path: &Path) -> PathBuf {
    let stem = source_path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("out");
    source_path.with_file_name(stem)
}

/// Chooses a unique temporary binary path for native `run` commands.
fn temporary_run_output_path() -> PathBuf {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    env::temp_dir().join(format!("skunk_run_{}_{}", std::process::id(), timestamp))
}

/// Loads, monomorphizes, and type-checks a program from disk.
fn load_and_check(file_path: &Path) -> Result<ast::Node, String> {
    let node = source::load_program(file_path)?;
    prepare_and_check(&node)
}

/// Monomorphizes and type-checks an already-loaded program.
fn prepare_and_check(node: &ast::Node) -> Result<ast::Node, String> {
    let node = monomorphize::prepare_program(node)?;
    type_checker::check(&node)?;
    Ok(node)
}

/// Compiles and executes a program natively, then removes its temporary build artifacts.
fn run_native(program: &ast::Node, source_path: &Path) -> Result<ExitStatus, String> {
    run_native_with_options(
        program,
        source_path,
        &compiler::BuildOptions::default(),
    )
}

/// Compiles and executes a program with explicit linker and optimization
/// options, then removes its temporary build artifacts.
fn run_native_with_options(
    program: &ast::Node,
    source_path: &Path,
    options: &compiler::BuildOptions,
) -> Result<ExitStatus, String> {
    let output_path = temporary_run_output_path();
    let llvm_ir_path = output_path.with_extension("ll");
    let artifact = match compiler::compile_to_executable_with_options(
        program,
        source_path,
        &output_path,
        options,
    ) {
        Ok(artifact) => artifact,
        Err(err) => {
            let _ = fs::remove_file(&llvm_ir_path);
            let _ = fs::remove_file(&output_path);
            return Err(err);
        }
    };

    let status = Command::new(&artifact.binary_path).status().map_err(|err| {
        format!(
            "failed to run compiled program {}: {}",
            artifact.binary_path.display(),
            err
        )
    });
    let _ = fs::remove_file(&artifact.llvm_ir_path);
    let _ = fs::remove_file(&artifact.binary_path);
    status
}

/// Converts a project manifest's build table into compiler options.
fn manifest_build_options(manifest: &manifest::Manifest) -> compiler::BuildOptions {
    compiler::BuildOptions {
        optimize: manifest.optimize,
        libraries: manifest.libraries.clone(),
        frameworks: manifest.frameworks.clone(),
    }
}

/// Prints non-fatal manifest diagnostics such as ignored future keys.
fn emit_manifest_warnings(manifest: &manifest::Manifest) {
    for warning in &manifest.warnings {
        eprintln!("warning: {}", warning);
    }
}

/// Resolves the source and build options for `skunk test`: an explicit path
/// uses defaults, while a project test inherits the complete `[build]` table.
fn resolve_test_configuration(
    source: Option<String>,
) -> Result<(PathBuf, compiler::BuildOptions), String> {
    if let Some(source) = source {
        return Ok((
            PathBuf::from(source),
            compiler::BuildOptions::default(),
        ));
    }
    let manifest_path = PathBuf::from(manifest::MANIFEST_FILE);
    if manifest_path.exists() {
        let manifest = manifest::load_manifest(&manifest_path)?;
        emit_manifest_warnings(&manifest);
        let options = manifest_build_options(&manifest);
        Ok((manifest.entry, options))
    } else {
        Err(format!(
            "no source file given and no `{}` found in the current directory\n{}",
            manifest::MANIFEST_FILE,
            USAGE
        ))
    }
}

/// Runs `skunk test`: rewrites test declarations into a native runner, builds
/// it, executes it, and returns its exit status.
fn run_tests(source: Option<String>, filter: Option<String>) -> Result<ExitStatus, String> {
    let (source_path, options) = resolve_test_configuration(source)?;
    let program = source::load_program(&source_path)?;
    let (test_program, test_count) =
        testing::build_test_program(&program, filter.as_deref())?;
    let test_program = prepare_and_check(&test_program)?;
    println!(
        "running {} test{} from {}\n",
        test_count,
        if test_count == 1 { "" } else { "s" },
        source_path.display()
    );
    run_native_with_options(&test_program, &source_path, &options)
}

/// Runs `skunk build`: compiles the manifest entry into `target/<name>`.
fn run_build() -> Result<PathBuf, String> {
    let manifest_path = PathBuf::from(manifest::MANIFEST_FILE);
    if !manifest_path.exists() {
        return Err(format!(
            "`skunk build` requires a `{}` in the current directory; run `skunk new <name>` to create a project",
            manifest::MANIFEST_FILE
        ));
    }
    let manifest = manifest::load_manifest(&manifest_path)?;
    emit_manifest_warnings(&manifest);
    let node = load_and_check(&manifest.entry)?;
    let target_dir = PathBuf::from("target");
    fs::create_dir_all(&target_dir)
        .map_err(|err| format!("failed to create `{}`: {}", target_dir.display(), err))?;
    let output_path = target_dir.join(&manifest.name);
    let options = manifest_build_options(&manifest);
    let artifact = compiler::compile_to_executable_with_options(
        &node,
        &manifest.entry,
        &output_path,
        &options,
    )?;
    Ok(artifact.binary_path)
}

/// Runs `skunk new`: scaffolds a project directory with a manifest and entry file.
fn run_new(name: &str) -> Result<PathBuf, String> {
    let valid = !name.is_empty()
        && name
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || ch == '_' || ch == '-')
        && name
            .chars()
            .next()
            .is_some_and(|ch| ch.is_ascii_alphabetic() || ch == '_');
    if !valid {
        return Err(format!(
            "invalid project name `{}`: use letters, digits, `_`, and `-`, starting with a letter or `_`",
            name
        ));
    }
    let root = PathBuf::from(name);
    if root.exists() {
        return Err(format!("`{}` already exists", root.display()));
    }
    let src_dir = root.join("src");
    fs::create_dir_all(&src_dir)
        .map_err(|err| format!("failed to create `{}`: {}", src_dir.display(), err))?;

    let manifest_contents = format!(
        "[package]\nname = \"{}\"\nentry = \"src/main.skunk\"\n\n[build]\noptimize = true\nlibraries = []\nframeworks = []\n",
        name
    );
    fs::write(root.join(manifest::MANIFEST_FILE), manifest_contents)
        .map_err(|err| format!("failed to write skunk.toml: {}", err))?;

    let main_contents = format!(
        "function main(): void {{\n    print(\"hello from {}\");\n}}\n\ntest \"it works\" {{\n    Testing::expect(1 + 1 == 2);\n}}\n",
        name
    );
    fs::write(src_dir.join("main.skunk"), main_contents)
        .map_err(|err| format!("failed to write src/main.skunk: {}", err))?;

    fs::write(root.join(".gitignore"), "/target\n*.ll\n")
        .map_err(|err| format!("failed to write .gitignore: {}", err))?;

    Ok(root)
}

/// The directory versioned binaries live in: `$SKUNK_HOME/bin`.
fn bin_dir() -> PathBuf {
    compiler::skunk_home().join("bin")
}

/// Returns the version the `skunk` symlink in the bin directory points at,
/// when it is a symlink to a `skunk-<version>` binary.
fn active_installed_version(dir: &Path) -> Option<String> {
    let target = fs::read_link(dir.join("skunk")).ok()?;
    let file_name = target.file_name()?.to_string_lossy().to_string();
    file_name.strip_prefix("skunk-").map(str::to_string)
}

/// Runs `skunk versions`: lists `skunk-<version>` binaries in the bin
/// directory and marks the one the `skunk` symlink points at.
fn list_versions() -> Result<(), String> {
    let dir = bin_dir();
    let mut versions = Vec::new();
    if let Ok(entries) = fs::read_dir(&dir) {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if let Some(version) = name.strip_prefix("skunk-") {
                // Skip leftovers like editor backups; versions start with a digit.
                if version.chars().next().is_some_and(|ch| ch.is_ascii_digit()) {
                    versions.push(version.to_string());
                }
            }
        }
    }
    versions.sort();

    if versions.is_empty() {
        println!("no versioned installs found in {}", dir.display());
        println!("this binary is skunk {}", env!("CARGO_PKG_VERSION"));
        println!("see RELEASE.md for the versioned install layout");
        return Ok(());
    }

    let active = active_installed_version(&dir);
    for version in &versions {
        if active.as_ref() == Some(version) {
            println!("* {} (active)", version);
        } else {
            println!("  {}", version);
        }
    }
    if active.is_none() {
        println!(
            "\nnote: `{}` is not a symlink to a versioned binary; run `skunk use <version>` to manage it",
            dir.join("skunk").display()
        );
    }
    Ok(())
}

/// Runs `skunk use <version>`: repoints the `skunk` symlink in the bin
/// directory at the requested versioned binary.
fn use_version(version: &str) -> Result<(), String> {
    let dir = bin_dir();
    let target = dir.join(format!("skunk-{}", version));
    if !target.exists() {
        return Err(format!(
            "skunk {} is not installed (expected `{}`); run `skunk versions` to list installed versions",
            version,
            target.display()
        ));
    }
    let link = dir.join("skunk");
    #[cfg(unix)]
    {
        if fs::symlink_metadata(&link).is_ok() {
            fs::remove_file(&link)
                .map_err(|err| format!("failed to remove `{}`: {}", link.display(), err))?;
        }
        std::os::unix::fs::symlink(&target, &link).map_err(|err| {
            format!(
                "failed to link `{}` -> `{}`: {}",
                link.display(),
                target.display(),
                err
            )
        })?;
        println!("now using skunk {}", version);
        Ok(())
    }
    #[cfg(not(unix))]
    {
        let _ = link;
        Err("`skunk use` is only supported on macOS and Linux".to_string())
    }
}

fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("{}", USAGE);
        std::process::exit(1);
    }
    let command = match parse_cli(&args) {
        Ok(parsed) => parsed,
        Err(err) => {
            eprintln!("{}", err.red());
            std::process::exit(1);
        }
    };

    match command {
        CommandKind::Run { source } => {
            let source_path = Path::new(&source);
            let node = match load_and_check(source_path) {
                Ok(node) => node,
                Err(err) => {
                    eprintln!("Error: {}", err.red());
                    std::process::exit(1);
                }
            };
            match run_native(&node, source_path) {
                Ok(status) if status.success() => {}
                Ok(status) => std::process::exit(status.code().unwrap_or(1)),
                Err(err) => {
                    eprintln!("Run error: {}", err.red());
                    std::process::exit(1);
                }
            }
        }
        CommandKind::Compile { source, output } => {
            let source_path = Path::new(&source);
            let node = match load_and_check(source_path) {
                Ok(node) => node,
                Err(err) => {
                    eprintln!("Error: {}", err.red());
                    std::process::exit(1);
                }
            };
            let output_path = output.unwrap_or_else(|| default_output_path(source_path));
            let now = Instant::now();
            match compiler::compile_to_executable(&node, source_path, &output_path) {
                Ok(artifact) => {
                    let elapsed = now.elapsed();
                    println!(
                        "Compiled {} -> {}",
                        artifact.llvm_ir_path.display(),
                        artifact.binary_path.display()
                    );
                    println!("Elapsed: {:.2?}", elapsed);
                }
                Err(err) => {
                    eprintln!("Compile error: {}", err.red());
                    std::process::exit(1);
                }
            }
        }
        CommandKind::Test { source, filter } => match run_tests(source, filter) {
            Ok(status) => std::process::exit(status.code().unwrap_or(1)),
            Err(err) => {
                eprintln!("Test error: {}", err.red());
                std::process::exit(1);
            }
        },
        CommandKind::Build => match run_build() {
            Ok(binary_path) => {
                println!("Built {}", binary_path.display());
            }
            Err(err) => {
                eprintln!("Build error: {}", err.red());
                std::process::exit(1);
            }
        },
        CommandKind::Version => {
            println!("skunk {}", env!("CARGO_PKG_VERSION"));
        }
        CommandKind::Versions => {
            if let Err(err) = list_versions() {
                eprintln!("Error: {}", err.red());
                std::process::exit(1);
            }
        }
        CommandKind::Use { version } => {
            if let Err(err) = use_version(&version) {
                eprintln!("Error: {}", err.red());
                std::process::exit(1);
            }
        }
        CommandKind::New { name } => match run_new(&name) {
            Ok(root) => {
                println!("Created project `{}`", root.display());
                println!("  cd {} && skunk build", root.display());
                println!("  skunk test");
            }
            Err(err) => {
                eprintln!("Error: {}", err.red());
                std::process::exit(1);
            }
        },
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| value.to_string()).collect()
    }

    #[test]
    fn bare_source_path_uses_native_run() {
        let args = args(&["skunk", "main.skunk"]);
        let command = parse_cli(&args).unwrap();

        assert_eq!(
            command,
            CommandKind::Run {
                source: "main.skunk".to_string()
            }
        );
    }

    #[test]
    fn explicit_run_uses_native_run() {
        let args = args(&["skunk", "run", "main.skunk"]);
        let command = parse_cli(&args).unwrap();

        assert_eq!(
            command,
            CommandKind::Run {
                source: "main.skunk".to_string()
            }
        );
    }

    #[test]
    fn legacy_interpret_command_is_rejected() {
        let args = args(&["skunk", "interpret", "main.skunk"]);

        assert!(parse_cli(&args).is_err());
    }

    #[test]
    fn compile_accepts_optional_output() {
        let args = args(&["skunk", "compile", "main.skunk", "out"]);
        let command = parse_cli(&args).unwrap();

        assert_eq!(
            command,
            CommandKind::Compile {
                source: "main.skunk".to_string(),
                output: Some(PathBuf::from("out")),
            }
        );
    }

    #[test]
    fn test_command_parses_filter() {
        let args = args(&["skunk", "test", "main.skunk", "--filter", "shorthand"]);
        let command = parse_cli(&args).unwrap();

        assert_eq!(
            command,
            CommandKind::Test {
                source: Some("main.skunk".to_string()),
                filter: Some("shorthand".to_string()),
            }
        );
    }

    #[test]
    fn test_command_allows_no_source() {
        let args = args(&["skunk", "test"]);
        let command = parse_cli(&args).unwrap();

        assert_eq!(
            command,
            CommandKind::Test {
                source: None,
                filter: None,
            }
        );
    }

    #[test]
    fn test_command_rejects_dangling_filter() {
        let args = args(&["skunk", "test", "--filter"]);

        assert!(parse_cli(&args).is_err());
    }

    #[test]
    fn project_test_build_options_include_native_link_settings() {
        let project = manifest::parse_manifest(
            r#"
            [package]
            name = "native-tests"
            entry = "src/main.skunk"

            [build]
            optimize = false
            libraries = ["sqlite3", "z"]
            frameworks = ["Cocoa"]
            "#,
        )
        .unwrap();
        let options = manifest_build_options(&project);

        assert!(!options.optimize);
        assert_eq!(options.libraries, vec!["sqlite3", "z"]);
        assert_eq!(options.frameworks, vec!["Cocoa"]);
    }

    #[test]
    fn explicit_test_file_uses_default_build_options() {
        let (source, options) =
            resolve_test_configuration(Some("tests/math.skunk".to_string())).unwrap();

        assert_eq!(source, PathBuf::from("tests/math.skunk"));
        assert!(options.optimize);
        assert!(options.libraries.is_empty());
        assert!(options.frameworks.is_empty());
    }

    #[test]
    fn new_command_requires_a_name() {
        assert!(parse_cli(&args(&["skunk", "new"])).is_err());
        assert_eq!(
            parse_cli(&args(&["skunk", "new", "demo"])).unwrap(),
            CommandKind::New {
                name: "demo".to_string()
            }
        );
    }

    #[test]
    fn version_flag_is_recognized() {
        assert_eq!(
            parse_cli(&args(&["skunk", "--version"])).unwrap(),
            CommandKind::Version
        );
        assert_eq!(
            parse_cli(&args(&["skunk", "-V"])).unwrap(),
            CommandKind::Version
        );
        assert_eq!(
            parse_cli(&args(&["skunk", "version"])).unwrap(),
            CommandKind::Version
        );
    }

    #[test]
    fn versions_and_use_are_recognized() {
        assert_eq!(
            parse_cli(&args(&["skunk", "versions"])).unwrap(),
            CommandKind::Versions
        );
        assert_eq!(
            parse_cli(&args(&["skunk", "use", "0.1.0"])).unwrap(),
            CommandKind::Use {
                version: "0.1.0".to_string()
            }
        );
        assert!(parse_cli(&args(&["skunk", "use"])).is_err());
    }

    #[test]
    fn build_takes_no_arguments() {
        assert_eq!(
            parse_cli(&args(&["skunk", "build"])).unwrap(),
            CommandKind::Build
        );
        assert!(parse_cli(&args(&["skunk", "build", "extra"])).is_err());
    }
}
