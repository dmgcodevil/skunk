mod ast;
mod compiler;
mod interpreter;
mod parser;
mod type_checker;
use colored::*;
use std::env;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::time::Instant;

enum CommandKind {
    Interpret,
    Compile { output: Option<PathBuf> },
}

fn parse_cli(args: &[String]) -> Result<(CommandKind, &str), String> {
    match args.len() {
        2 => Ok((CommandKind::Interpret, &args[1])),
        _ => match args[1].as_str() {
            "interpret" => {
                if args.len() != 3 {
                    Err("Usage: skunk interpret <file_path>".to_string())
                } else {
                    Ok((CommandKind::Interpret, &args[2]))
                }
            }
            "compile" => {
                if args.len() < 3 || args.len() > 4 {
                    Err("Usage: skunk compile <file_path> [output_path]".to_string())
                } else {
                    Ok((
                        CommandKind::Compile {
                            output: args.get(3).map(PathBuf::from),
                        },
                        &args[2],
                    ))
                }
            }
            _ => Err(
                "Usage: skunk <file_path>\n       skunk interpret <file_path>\n       skunk compile <file_path> [output_path]"
                    .to_string(),
            ),
        },
    }
}

fn default_output_path(source_path: &Path) -> PathBuf {
    let stem = source_path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("out");
    source_path.with_file_name(stem)
}

fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();
    let type_checker_enabled: bool = true;
    if args.len() < 2 {
        eprintln!("Usage: skunk <file_path>");
        std::process::exit(1);
    }
    let (command, file_path) = match parse_cli(&args) {
        Ok(parsed) => parsed,
        Err(err) => {
            eprintln!("{}", err.red());
            std::process::exit(1);
        }
    };
    let contents = fs::read_to_string(file_path)?;
    let node = ast::parse(&contents);
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
        CommandKind::Interpret => {
            let now = Instant::now();
            let _result = interpreter::evaluate(&node);
            let elapsed = now.elapsed();
            println!("Elapsed: {:.2?}", elapsed);
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
