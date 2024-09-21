mod ast;
mod interpreter;
mod parser;

fn main() -> rustyline::Result<()> {
    interpreter::repl()
}
