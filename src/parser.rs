use pest::Parser;
use pest_derive::Parser;

#[derive(Parser)]
#[grammar = "grammar.pest"] // Specifies the grammar file
pub struct SkunkParser;
