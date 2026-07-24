use pest_derive::Parser;

#[derive(Parser)]
#[grammar = "syntax/grammar.pest"]
pub struct SkunkParser;
