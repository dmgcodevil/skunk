mod ast;
mod compiler;
mod monomorphize;
mod parser;
mod source;
mod type_checker;
use colored::*;
use std::env;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const USAGE: &str = "Usage: skunk <file_path>\n       skunk run <file_path>\n       skunk compile <file_path> [output_path]";

enum CommandKind {
    Run,
    Compile { output: Option<PathBuf> },
}

/// Parses the command line into a high-level command and source path.
fn parse_cli(args: &[String]) -> Result<(CommandKind, &str), String> {
    match args.len() {
        2 => Ok((CommandKind::Run, &args[1])),
        _ => match args[1].as_str() {
            "run" => {
                if args.len() != 3 {
                    Err(USAGE.to_string())
                } else {
                    Ok((CommandKind::Run, &args[2]))
                }
            }
            "compile" => {
                if args.len() < 3 || args.len() > 4 {
                    Err(USAGE.to_string())
                } else {
                    Ok((
                        CommandKind::Compile {
                            output: args.get(3).map(PathBuf::from),
                        },
                        &args[2],
                    ))
                }
            }
            _ => Err(USAGE.to_string()),
        },
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
    env::temp_dir().join(format!(
        "skunk_run_{}_{}",
        std::process::id(),
        timestamp
    ))
}

/// Compiles and executes a program natively, then removes its temporary build artifacts.
fn run_native(program: &ast::Node, source_path: &Path) -> Result<ExitStatus, String> {
    let output_path = temporary_run_output_path();
    let llvm_ir_path = output_path.with_extension("ll");
    let artifact = match compiler::compile_to_executable(program, source_path, &output_path) {
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

fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();
    let type_checker_enabled: bool = true;
    if args.len() < 2 {
        eprintln!("{}", USAGE);
        std::process::exit(1);
    }
    let (command, file_path) = match parse_cli(&args) {
        Ok(parsed) => parsed,
        Err(err) => {
            eprintln!("{}", err.red());
            std::process::exit(1);
        }
    };
    let node = match source::load_program(Path::new(file_path)) {
        Ok(node) => node,
        Err(err) => {
            eprintln!("Error: {}", err.red());
            std::process::exit(1);
        }
    };
    let node = match monomorphize::prepare_program(&node) {
        Ok(node) => node,
        Err(err) => {
            eprintln!("Error: {}", err.red());
            std::process::exit(1);
        }
    };
    if type_checker_enabled {
        match type_checker::check(&node) {
            Ok(_) => (),
            Err(e) => {
                eprintln!("Error: {}", e.red());
                std::process::exit(1);
            }
        };
    }

    match command {
        CommandKind::Run => match run_native(&node, Path::new(file_path)) {
            Ok(status) if status.success() => {}
            Ok(status) => std::process::exit(status.code().unwrap_or(1)),
            Err(err) => {
                eprintln!("Run error: {}", err.red());
                std::process::exit(1);
            }
        }
        CommandKind::Compile { output } => {
            let source_path = Path::new(file_path);
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
        let (command, source) = parse_cli(&args).unwrap();

        assert!(matches!(command, CommandKind::Run));
        assert_eq!(source, "main.skunk");
    }

    #[test]
    fn explicit_run_uses_native_run() {
        let args = args(&["skunk", "run", "main.skunk"]);
        let (command, source) = parse_cli(&args).unwrap();

        assert!(matches!(command, CommandKind::Run));
        assert_eq!(source, "main.skunk");
    }

    #[test]
    fn legacy_interpret_command_is_rejected() {
        let args = args(&["skunk", "interpret", "main.skunk"]);

        assert!(parse_cli(&args).is_err());
    }
}
