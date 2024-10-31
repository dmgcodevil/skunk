use crate::parser_demo::{DemoParser, Rule};

mod ast;
mod interpreter;
mod parser;
mod parser_demo;
use pest::Parser;
fn main() {
    println!("Hello, world!");
    match DemoParser::parse(Rule::member_access, r#"a[0].b"#) {
        Ok(pairs) => {
            println!("{:?}", pairs);
        }
        Err(e) => {
            panic!("Error while parsing: {:?}", e);
        }
    }
}
