mod ast;
mod interpreter;
mod parser;
mod type_checker;
use colored::*;
use std::env;
use std::fs;
use std::io;
use std::ops::Deref;

fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 1 {
        eprintln!("Usage: <file_path>");
        std::process::exit(1);
    }
    let file_path = &args[1];
    let contents = fs::read_to_string(file_path)?;
    println!("{}", contents);
    let node = ast::parse(&contents);
    println!("{:#?}", node);
    match type_checker::check(&node) {
        Ok(_) => (),
        Err(e) => {
            eprintln!("Error: {}", e.red());
            std::process::exit(1);
        }
    };
    let result = interpreter::evaluate(&node);
    let res_ref = result.borrow();
    println!("Result:");
    println!("{}", res_ref);
    Ok(())
}
