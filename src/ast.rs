use crate::parser::{Rule, SkunkParser};
use pest::iterators::Pair;
use pest::pratt_parser::{Assoc, Op, PrattParser};
use pest::Parser;

#[derive(Debug, PartialEq)]
pub enum Node {
    Program {
        statements: Vec<Node>,
    },
    // Statements
    StructDeclaration {
        name: String,
        declarations: Vec<Node>,
    },
    VariableDeclaration {
        var_type: Type,
        name: String,
        value: Box<Node>, // The value is an expression node
    },
    FunctionDeclaration {
        name: String,
        parameters: Vec<(String, Type)>,
        return_type: Type,
        body: Vec<Node>, // The function body is a list of nodes (statements or expressions)
    },
    Assignment {
        identifier: String,
        value: Box<Node>,
    },
    If {
        condition: Box<Node>,          // The condition of the `if` or `else if`
        body: Vec<Node>,               // The body of the `if` or `else if`
        else_if_blocks: Vec<Node>,     // List of else if blocks
        else_block: Option<Vec<Node>>, // Optional else block
    },
    For {
        init: Option<Box<Node>>,      // Initialization is a statement node
        condition: Option<Box<Node>>, // The condition is an expression node
        update: Option<Box<Node>>,    // Update is a statement node
        body: Vec<Node>,              // The body is a list of nodes
    },
    Return(Box<Node>), // The return value is an expression node
    Print(Box<Node>),  // The print expression is an expression node
    Input,             // Read data from keyboard

    // Expressions
    Literal(Literal),   // Represents a literal value (int, string, bool)
    Identifier(String), // Represents a variable or function name
    BinaryOp {
        left: Box<Node>,    // The left operand is an expression node
        operator: Operator, // The operator
        right: Box<Node>,   // The right operand is an expression node
    },
    FunctionCall {
        name: String,         // The function name
        arguments: Vec<Node>, // The arguments are a list of expression nodes
    },
    EOI,
    EMPTY,
}

#[derive(Debug, PartialEq)]
pub enum Literal {
    Integer(i64),
    StringLiteral(String),
    Boolean(bool),
}

#[derive(Debug, PartialEq)]
pub enum Operator {
    Add,
    Subtract,
    Multiply,
    Divide,
    Power,
    Equals,
    NotEquals,
    LessThan,
    GreaterThan,
    LessThanOrEqual,
    GreaterThanOrEqual,
    And,
    Or,
    Not,
}

#[derive(Debug, PartialEq)]
pub enum Type {
    Void,
    Int,
    String,
    Boolean,
    Custom(String), // Custom types like structs
}

fn create_ast(pair: Pair<Rule>) -> Node {
    let r = pair.as_rule();
    match r {
        Rule::program => {
            let mut statements: Vec<Node> = Vec::new();
            for inner_pair in pair.into_inner() {
                statements.push(create_ast(inner_pair));
            }
            Node::Program { statements }
        }
        Rule::statement => {
            let mut pairs = pair.into_inner();
            let inner = pairs.next().unwrap();
            create_ast(inner)
        }
        Rule::expression => create_expression(pair),
        Rule::assignment => create_assignment(pair),
        Rule::struct_decl => create_struct_decl(pair),
        Rule::var_decl => create_var_decl(pair),
        Rule::func_decl => create_func_decl(pair),
        Rule::literal => create_literal(pair),
        Rule::primary => {
            let mut pairs = pair.into_inner();
            create_ast(pairs.next().unwrap())
        }
        Rule::IDENTIFIER => Node::Identifier(pair.as_str().to_string()),
        Rule::member_access => Node::Identifier(create_identifier(pair)),
        Rule::func_call => create_function_call(pair),
        Rule::sk_return => {
            let mut pairs = pair.into_inner();
            Node::Return(Box::new(create_ast(pairs.next().unwrap())))
        }
        Rule::io => {
            let p = pair.into_inner().next().unwrap();
            match p.as_rule() {
                Rule::print => Node::Print(Box::new(create_ast(p.into_inner().next().unwrap()))),
                Rule::input => Node::Input,
                _ => panic!("unsupported IO rule"),
            }
        }
        Rule::control_flow => {
            let p = pair.into_inner().next().unwrap();
            match p.as_rule() {
                Rule::if_expr => create_if_expr(p),
                Rule::for_expr => create_for_classic(p),
                _ => panic!("unsupported control flow"),
            }
        }
        Rule::EOI => Node::EOI,
        _ => {
            panic!("Unexpected Node: {:?}", pair);
        }
    }
}

lazy_static::lazy_static! {
    // Define operator precedence and associativity
      static ref PRATT_PARSER: PrattParser<Rule> = {
        use Rule::*;
        use Assoc::*;

        PrattParser::new()
            .op(Op::infix(add, Left) | Op::infix(subtract, Left))
            .op(Op::infix(multiply, Left) | Op::infix(divide, Left))
            .op(Op::infix(power, Right))
            .op(Op::infix(or, Left))
            .op(Op::infix(and, Left))
            .op(Op::infix(lt, Left))
            .op(Op::infix(gt, Left))
            .op(Op::infix(eq, Left))
};
}

fn create_literal(pair: Pair<Rule>) -> Node {
    let literal = pair.into_inner().next().unwrap();
    match literal.as_rule() {
        Rule::STRING_LITERAL => Node::Literal(Literal::StringLiteral(literal.as_str().to_string())),
        Rule::INTEGER => Node::Literal(Literal::Integer(literal.as_str().parse::<i64>().unwrap())),
        Rule::BOOLEAN_LITERAL => {
            Node::Literal(Literal::Boolean(literal.as_str().parse::<bool>().unwrap()))
        }
        _ => panic!("unsupported rule {:?}", literal),
    }
}

fn create_function_call(pair: Pair<Rule>) -> Node {
    let mut inner_pairs = pair.into_inner();
    let name = inner_pairs.next().unwrap().as_str().to_string();
    let mut arguments = Vec::new();
    if let Some(arg_pair) = inner_pairs.next() {
        arguments.push(create_ast(arg_pair));
        for arg in inner_pairs {
            arguments.push(create_ast(arg));
        }
    }
    Node::FunctionCall { name, arguments }
}

fn create_type_from_str(s: &str) -> Type {
    match s {
        "int" => Type::Int,
        "string" => Type::String,
        "boolean" => Type::Boolean,
        "void" => Type::Void,
        _ => Type::Custom(s.to_string()),
    }
}

fn create_type(pair: Pair<Rule>) -> Type {
    create_type_from_str(pair.as_str())
}

fn create_struct_decl(pair: Pair<Rule>) -> Node {
    let mut inner_pairs = pair.into_inner();
    let name = inner_pairs.next().unwrap().as_str().to_string();
    let mut declarations: Vec<Node> = Vec::new();
    while let Some(p) = inner_pairs.next() {
        match p.as_rule() {
            Rule::struct_field_decl => declarations.push(create_var_decl(p)),
            _ => panic!("unsupported rule {}", p),
        }
    }
    Node::StructDeclaration { name, declarations }
}

fn create_var_decl(pair: Pair<Rule>) -> Node {
    let mut inner_pairs = pair.into_inner();
    let var_name = inner_pairs.next().unwrap().as_str().to_string();
    let var_type = create_type_from_str(inner_pairs.next().unwrap().as_str());

    let body = if let Some(body_pair) = inner_pairs.next() {
        create_ast(body_pair)
    } else {
        Node::EMPTY
    };

    Node::VariableDeclaration {
        var_type,
        name: var_name,
        value: Box::new(body),
    }
}

/// returns params as a vector of (String, Type) pairs
fn create_param_list(pair: Pair<Rule>) -> Vec<(String, Type)> {
    match pair.as_rule() {
        Rule::empty_params => Vec::new(),
        Rule::param_list => {
            let mut result: Vec<(String, Type)> = Vec::new();
            let pairs: Vec<Pair<Rule>> = pair.into_inner().collect();
            assert_eq!(pairs.len() % 2, 0);
            pairs.chunks(2).for_each(|chunk| {
                result.push((
                    chunk[0].as_str().to_string(),
                    create_type_from_str(chunk[1].as_str()),
                ));
            });
            result
        }
        _ => panic!("unexpected  rule {}", pair),
    }
}

fn stub() -> Node {
    Node::EMPTY
}

fn create_expression(pair: Pair<Rule>) -> Node {
    PRATT_PARSER
        .map_primary(|primary| create_ast(primary))
        .map_infix(|lhs, op, rhs| {
            let operator = match op.as_rule() {
                // Rule::assign => Operator::Assign,
                Rule::add => Operator::Add,
                Rule::subtract => Operator::Subtract,
                Rule::multiply => Operator::Multiply,
                Rule::divide => Operator::Divide,
                Rule::power => Operator::Power,
                Rule::and => Operator::And,
                Rule::or => Operator::Or,
                Rule::lt => Operator::LessThan,
                Rule::gt => Operator::GreaterThan,
                Rule::eq => Operator::Equals,
                _ => unreachable!(),
            };

            Node::BinaryOp {
                left: Box::new(lhs),
                operator,
                right: Box::new(rhs),
            }
        })
        .parse(pair.into_inner())
}

/// return identifier as String out of the pair
/// panics if rule != Rule::IDENTIFIER
fn create_identifier(pair: Pair<Rule>) -> String {
    match pair.as_rule() {
        Rule::IDENTIFIER => {
            pair.as_str().to_string()
        }
        Rule::member_access => {
            let mut v: Vec<String> = Vec::new();
            for p in pair.into_inner() {
                v.push(create_identifier(p));
            }
            v.join(".")
        }
        _ => panic!("not identifier rule {}", pair),
    }
}

/// creates Node::FunctionDeclaration from the pair
fn create_func_decl(pair: Pair<Rule>) -> Node {
    let mut inner_pairs = pair.into_inner();
    let name = create_identifier(inner_pairs.next().unwrap());
    let parameters = create_param_list(inner_pairs.next().unwrap());
    let return_type = match inner_pairs.peek() {
        Some(p) => {
            if p.as_rule() == Rule::return_type {
                // return_type -> type
                create_type_from_str(
                    inner_pairs
                        .next()
                        .unwrap()
                        .into_inner()
                        .next()
                        .unwrap()
                        .as_str(),
                )
            } else {
                Type::Void
            }
        }
        _ => Type::Void,
    };
    let mut body: Vec<Node> = Vec::new();
    while let Some(statement) = inner_pairs.next() {
        body.push(create_ast(statement))
    }
    Node::FunctionDeclaration {
        name,
        parameters,
        return_type,
        body,
    }
}

/// Recursive function to pretty print the parse tree
fn pretty_print(pair: Pair<Rule>, depth: usize) {
    println!(
        "{:indent$}{:?}: `{}`",
        "",
        pair.as_rule(),
        pair.as_str(),
        indent = depth * 2
    );
    for inner_pair in pair.into_inner() {
        pretty_print(inner_pair, depth + 1);
    }
}

fn create_if_expr(pair: Pair<Rule>) -> Node {
    let mut inner_pairs = pair.into_inner();
    let condition = create_ast(inner_pairs.next().unwrap());
    let body = create_body(inner_pairs.next().unwrap().into_inner());

    // Parse optional `else if` blocks
    let mut else_if_blocks = Vec::new();
    while let Some(else_if_pair) = inner_pairs.clone().peek() {
        if else_if_pair.as_rule() == Rule::else_if_expr {
            inner_pairs.next(); // Consume the peeked pair
            let mut elif_pairs = else_if_pair.into_inner(); // condition + body
            let else_if_condition = create_ast(elif_pairs.next().unwrap());
            let else_if_body = create_body(elif_pairs.next().unwrap().into_inner());
            else_if_blocks.push(Node::If {
                condition: Box::new(else_if_condition),
                body: else_if_body,
                else_if_blocks: Vec::new(),
                else_block: None,
            });
        } else {
            break;
        }
    }

    // Parse optional `else` block
    let else_block = if let Some(else_pair) = inner_pairs.next() {
        Some(create_body(else_pair.into_inner()))
    } else {
        None
    };

    Node::If {
        condition: Box::new(condition),
        body,
        else_if_blocks,
        else_block,
    }
}

fn create_body(pairs: pest::iterators::Pairs<Rule>) -> Vec<Node> {
    pairs.map(create_ast).collect()
}

fn create_assignment(pair: Pair<Rule>) -> Node {
    assert_eq!(Rule::assignment, pair.as_rule());
    let mut inner_pairs = pair.into_inner();
    let identifier = create_identifier(inner_pairs.next().unwrap());
    let value = create_ast(inner_pairs.next().unwrap());
    Node::Assignment {
        identifier,
        value: Box::new(value),
    }
}

fn create_for_classic(pair: Pair<Rule>) -> Node {
    let mut inner_pairs = pair.into_inner();
    assert_eq!(Rule::for_classic, inner_pairs.peek().unwrap().as_rule());
    let mut for_decl = inner_pairs.next().unwrap().into_inner();
    let init_pair = for_decl.next().unwrap();
    let init: Option<Node> = if !init_pair.as_str().is_empty() {
        let init_kind = init_pair.into_inner().next().unwrap();
        match init_kind.as_rule() {
            Rule::assignment => Some(create_assignment(init_kind)),
            Rule::var_decl => Some(create_var_decl(init_kind)),
            _ => panic!("unsupported 'for' init rule {:?}", init_kind),
        }
    } else {
        None
    };

    let condition_pair = for_decl.next().unwrap();
    let condition: Option<Node> = if !condition_pair.as_str().is_empty() {
        // todo add cond rule. create_ast is too generic
        Some(create_ast(condition_pair.into_inner().next().unwrap()))
    } else {
        None
    };
    let update_pair = for_decl.next().unwrap();
    let update: Option<Node> = if !update_pair.as_str().is_empty() {
        Some(create_assignment(update_pair.into_inner().next().unwrap()))
    } else {
        None
    };
    println!("for-init = {:?}", init);
    println!("for-cond = {:?}", condition);
    println!("for-update= {:?}", update);

    let body = create_body(inner_pairs.next().unwrap().into_inner());
    Node::For {
        init: init.map(Box::new),
        condition: condition.map(Box::new),
        update: update.map(Box::new),
        body,
    }
}

fn parse(code: &str) -> Node {
    match SkunkParser::parse(Rule::program, code) {
        Ok(pairs) => {
            pretty_print(pairs.clone().next().unwrap(), 0);
            create_ast(pairs.clone().next().unwrap())
        }
        Err(e) => {
            panic!("Error while parsing: {}", e);
        }
    }
}

mod tests {
    use super::*;

    fn int_var_decl(name: &str, value: i64) -> Node {
        Node::VariableDeclaration {
            name: name.to_string(),
            var_type: Type::Int,
            value: Box::new(Node::Literal(Literal::Integer(value))),
        }
    }

    fn int_var_assign(name: &str, value: i64) -> Node {
        Node::Assignment {
            identifier: name.to_string(),
            value: Box::new(Node::Literal(Literal::Integer(value))),
        }
    }

    fn var_less_than_int(name: &str, value: i64) -> Node {
        Node::BinaryOp {
            left: Box::new(Node::Identifier(name.to_string())),
            operator: Operator::LessThan,
            right: Box::new(Node::Literal(Literal::Integer(value))),
        }
    }

    fn inc_int_var(name: &str) -> Node {
        Node::Assignment {
            identifier: name.to_string(),
            value: Box::new(Node::BinaryOp {
                left: Box::new(Node::Identifier(name.to_string())),
                operator: Operator::Add,
                right: Box::new(Node::Literal(Literal::Integer(1))),
            }),
        }
    }

    fn print_int(i: i64) -> Node {
        Node::Print(Box::new(Node::Literal(Literal::Integer(i))))
    }
    fn print_var(name: &str) -> Node {
        Node::Print(Box::new(Node::Identifier(name.to_string())))
    }

    #[test]
    fn test_fn_decl_no_args_return_void() {
        let source_code = r#"
        function main():void {}
        "#;
        assert_eq!(
            Node::Program {
                statements: Vec::from([
                    Node::FunctionDeclaration {
                        name: "main".to_string(),
                        parameters: Vec::new(),
                        return_type: Type::Void,
                        body: Vec::new(),
                    },
                    Node::EOI
                ])
            },
            parse(source_code)
        );
    }

    #[test]
    fn test_fn_decl_no_args_return_nothing() {
        let source_code = r#"
        function main() {}
        "#;
        assert_eq!(
            Node::Program {
                statements: Vec::from([
                    Node::FunctionDeclaration {
                        name: "main".to_string(),
                        parameters: Vec::new(),
                        return_type: Type::Void,
                        body: Vec::new(),
                    },
                    Node::EOI
                ])
            },
            parse(source_code)
        );
    }

    #[test]
    fn test_fn_decl_with_args_return_int() {
        let source_code = r#"
        function main(a:int, b:int):int {}
        "#;
        assert_eq!(
            Node::Program {
                statements: Vec::from([
                    Node::FunctionDeclaration {
                        name: "main".to_string(),
                        parameters: Vec::from([
                            ("a".to_string(), Type::Int),
                            ("b".to_string(), Type::Int)
                        ]),
                        return_type: Type::Int,
                        body: Vec::new(),
                    },
                    Node::EOI
                ])
            },
            parse(source_code)
        );
    }

    #[test]
    fn test_var_dec_literal() {
        let source_code = r#"
        a:int = 0;
        "#;
        assert_eq!(
            Node::Program {
                statements: Vec::from([
                    Node::VariableDeclaration {
                        name: "a".to_string(),
                        var_type: Type::Int,
                        value: Box::new(Node::Literal(Literal::Integer(0)))
                    },
                    Node::EOI
                ])
            },
            parse(source_code)
        )
    }

    #[test]
    fn test_update_var() {
        let source_code = r#"
        i = i + 1;
        "#;
        let add = Node::BinaryOp {
            left: Box::new(Node::Identifier("i".to_string())),
            operator: Operator::Add,
            right: Box::new(Node::Literal(Literal::Integer(1))),
        };

        assert_eq!(
            Node::Program {
                statements: Vec::from([
                    Node::Assignment {
                        identifier: "i".to_string(),
                        value: Box::new(add)
                    },
                    Node::EOI
                ])
            },
            parse(source_code)
        )
    }

    #[test]
    fn test_var_dec_arithmetic_exp() {
        let source_code = r#"
        a:int = 2 + 3;
        "#;
        assert_eq!(
            Node::Program {
                statements: Vec::from([
                    Node::VariableDeclaration {
                        name: "a".to_string(),
                        var_type: Type::Int,
                        value: Box::new(Node::BinaryOp {
                            left: Box::new(Node::Literal(Literal::Integer(2))),
                            operator: Operator::Add,
                            right: Box::new(Node::Literal(Literal::Integer(3)))
                        })
                    },
                    Node::EOI
                ])
            },
            parse(source_code)
        )
    }
    #[test]
    fn test_var_dec_arithmetic_exp2() {
        let source_code = r#"
        a: int = 2 + 3 / 4;
        "#;
        assert_eq!(
            Node::Program {
                statements: Vec::from([
                    Node::VariableDeclaration {
                        name: "a".to_string(),
                        var_type: Type::Int,
                        value: Box::new(Node::BinaryOp {
                            left: Box::new(Node::Literal(Literal::Integer(2))),
                            operator: Operator::Add,
                            right: Box::new(Node::BinaryOp {
                                left: Box::new(Node::Literal(Literal::Integer(3))),
                                operator: Operator::Divide,
                                right: Box::new(Node::Literal(Literal::Integer(4))),
                            })
                        })
                    },
                    Node::EOI
                ])
            },
            parse(source_code)
        )
    }
    #[test]
    fn test_var_dec_arithmetic_exp3() {
        let source_code = r#"
        a:int = (2 + 3) / 4;
        "#;
        assert_eq!(
            Node::Program {
                statements: Vec::from([
                    Node::VariableDeclaration {
                        name: "a".to_string(),
                        var_type: Type::Int,
                        value: Box::new(Node::BinaryOp {
                            left: Box::new(Node::BinaryOp {
                                left: Box::new(Node::Literal(Literal::Integer(2))),
                                operator: Operator::Add,
                                right: Box::new(Node::Literal(Literal::Integer(3)))
                            }),
                            operator: Operator::Divide,
                            right: Box::new(Node::Literal(Literal::Integer(4)))
                        })
                    },
                    Node::EOI
                ])
            },
            parse(source_code)
        );
    }

    #[test]
    fn test_function_call() {
        let source_code = r#"
        function sum(a: int, b: int): int {
            return a + b;
        }

        function main() {
           res: int = sum(1, 2);
        }
        "#;
        assert_eq!(
            Node::Program {
                statements: Vec::from([
                    Node::FunctionDeclaration {
                        name: "sum".to_string(),
                        parameters: Vec::from([
                            ("a".to_string(), Type::Int),
                            ("b".to_string(), Type::Int)
                        ]),
                        return_type: Type::Int,
                        body: Vec::from([Node::Return(Box::new(Node::BinaryOp {
                            left: Box::new(Node::Identifier("a".to_string())),
                            operator: Operator::Add,
                            right: Box::new(Node::Identifier("b".to_string()))
                        }))]),
                    },
                    Node::FunctionDeclaration {
                        name: "main".to_string(),
                        parameters: Vec::new(),
                        return_type: Type::Void,
                        body: Vec::from([Node::VariableDeclaration {
                            name: "res".to_string(),
                            var_type: Type::Int,
                            value: Box::new(Node::FunctionCall {
                                name: "sum".to_string(),
                                arguments: Vec::from([
                                    Node::Literal(Literal::Integer(1)),
                                    Node::Literal(Literal::Integer(2))
                                ])
                            })
                        }]),
                    },
                    Node::EOI
                ])
            },
            parse(source_code)
        )
    }

    #[test]
    fn test_struct_decl() {
        let source_code = r#"
        struct Foo {
            i: int;
            s: string;
            b: boolean;
        }
        "#;
        assert_eq!(
            Node::Program {
                statements: Vec::from([
                    Node::StructDeclaration {
                        name: "Foo".to_string(),
                        declarations: Vec::from([
                            Node::VariableDeclaration {
                                name: "i".to_string(),
                                var_type: Type::Int,
                                value: Box::new(Node::EMPTY)
                            },
                            Node::VariableDeclaration {
                                name: "s".to_string(),
                                var_type: Type::String,
                                value: Box::new(Node::EMPTY)
                            },
                            Node::VariableDeclaration {
                                name: "b".to_string(),
                                var_type: Type::Boolean,
                                value: Box::new(Node::EMPTY)
                            },
                        ])
                    },
                    Node::EOI
                ])
            },
            parse(source_code)
        )
    }
    #[test]
    fn test_print() {
        let source_code = r#"
        print("test");
        "#;
        assert_eq!(
            Node::Program {
                statements: Vec::from([
                    Node::Print(Box::new(Node::Literal(Literal::StringLiteral(
                        "\"test\"".to_string()
                    )))),
                    Node::EOI
                ])
            },
            parse(source_code)
        );
    }

    #[test]
    fn test_input() {
        let source_code = r#"
        input();
        "#;
        assert_eq!(
            Node::Program {
                statements: Vec::from([Node::Input, Node::EOI])
            },
            parse(source_code)
        );
    }

    #[test]
    fn test_and() {
        let source_code = r#"
            c: boolean = a && b;
        "#;
        let and = Node::BinaryOp {
            left: Box::new(Node::Identifier("a".to_string())),
            operator: Operator::And,
            right: Box::new(Node::Identifier("b".to_string())),
        };
        let var_decl = Node::VariableDeclaration {
            var_type: Type::Boolean,
            name: "c".to_string(),
            value: Box::new(and),
        };

        assert_eq!(
            Node::Program {
                statements: Vec::from([var_decl, Node::EOI])
            },
            parse(source_code)
        )
    }
    #[test]
    fn test_or() {
        let source_code = r#"
            c: boolean = a || b;
        "#;
        let and = Node::BinaryOp {
            left: Box::new(Node::Identifier("a".to_string())),
            operator: Operator::Or,
            right: Box::new(Node::Identifier("b".to_string())),
        };
        let var_declr = Node::VariableDeclaration {
            var_type: Type::Boolean,
            name: "c".to_string(),
            value: Box::new(and),
        };

        assert_eq!(
            Node::Program {
                statements: Vec::from([var_declr, Node::EOI])
            },
            parse(source_code)
        )
    }

    #[test]
    fn test_or_and() {
        let source_code = r#"
            d: boolean = a || b && c;
        "#;
        let and = Node::BinaryOp {
            left: Box::new(Node::Identifier("b".to_string())),
            operator: Operator::And,
            right: Box::new(Node::Identifier("c".to_string())),
        };
        let or = Node::BinaryOp {
            left: Box::new(Node::Identifier("a".to_string())),
            operator: Operator::Or,
            right: Box::new(and),
        };
        let var_decl = Node::VariableDeclaration {
            var_type: Type::Boolean,
            name: "d".to_string(),
            value: Box::new(or),
        };

        assert_eq!(
            Node::Program {
                statements: Vec::from([var_decl, Node::EOI])
            },
            parse(source_code)
        )
    }

    #[test]
    fn test_if() {
        let source_code = r#"
            if (a < b) {
                print("a < b");
            }
        "#;
        let lt = Node::BinaryOp {
            left: Box::new(Node::Identifier("a".to_string())),
            operator: Operator::LessThan,
            right: Box::new(Node::Identifier("b".to_string())),
        };
        let body = Vec::from([Node::Print(Box::new(Node::Literal(
            Literal::StringLiteral("\"a < b\"".to_string()),
        )))]);
        let if_statement = Node::If {
            condition: Box::new(lt),
            body,
            else_if_blocks: Vec::new(),
            else_block: Option::None,
        };

        assert_eq!(
            Node::Program {
                statements: Vec::from([if_statement, Node::EOI])
            },
            parse(source_code)
        );
    }
    #[test]
    fn test_if_elif() {
        let source_code = r#"
            if (a < b) {
                print("a < b");
            } else if (a > b) {
                print("a > b");
            }
        "#;
        let lt = Node::BinaryOp {
            left: Box::new(Node::Identifier("a".to_string())),
            operator: Operator::LessThan,
            right: Box::new(Node::Identifier("b".to_string())),
        };
        let gt = Node::BinaryOp {
            left: Box::new(Node::Identifier("a".to_string())),
            operator: Operator::GreaterThan,
            right: Box::new(Node::Identifier("b".to_string())),
        };
        let if_body = Vec::from([Node::Print(Box::new(Node::Literal(
            Literal::StringLiteral("\"a < b\"".to_string()),
        )))]);
        let else_if_body = Vec::from([Node::Print(Box::new(Node::Literal(
            Literal::StringLiteral("\"a > b\"".to_string()),
        )))]);
        let if_statement = Node::If {
            condition: Box::new(lt),
            body: if_body,
            else_if_blocks: Vec::from([Node::If {
                condition: Box::new(gt),
                body: else_if_body,
                else_if_blocks: Vec::new(),
                else_block: Option::None,
            }]),
            else_block: Option::None,
        };
        assert_eq!(
            Node::Program {
                statements: Vec::from([if_statement, Node::EOI])
            },
            parse(source_code)
        );
    }

    #[test]
    fn test_if_elif_else() {
        let source_code = r#"
            if (a < b) {
                print("a < b");
            } else if(a > b) {
                print("a > b");
            } else {
                print("a == b");
            }
        "#;
        let lt = Node::BinaryOp {
            left: Box::new(Node::Identifier("a".to_string())),
            operator: Operator::LessThan,
            right: Box::new(Node::Identifier("b".to_string())),
        };
        let gt = Node::BinaryOp {
            left: Box::new(Node::Identifier("a".to_string())),
            operator: Operator::GreaterThan,
            right: Box::new(Node::Identifier("b".to_string())),
        };
        let eq = Node::BinaryOp {
            left: Box::new(Node::Identifier("a".to_string())),
            operator: Operator::Equals,
            right: Box::new(Node::Identifier("b".to_string())),
        };
        let if_body = Vec::from([Node::Print(Box::new(Node::Literal(
            Literal::StringLiteral("\"a < b\"".to_string()),
        )))]);
        let else_if_body = Vec::from([Node::Print(Box::new(Node::Literal(
            Literal::StringLiteral("\"a > b\"".to_string()),
        )))]);
        let else_body = Vec::from([Node::Print(Box::new(Node::Literal(
            Literal::StringLiteral("\"a == b\"".to_string()),
        )))]);
        let if_statement = Node::If {
            condition: Box::new(lt),
            body: if_body,
            else_if_blocks: Vec::from([Node::If {
                condition: Box::new(gt),
                body: else_if_body,
                else_if_blocks: Vec::new(),
                else_block: Option::None,
            }]),
            else_block: Option::Some(else_body),
        };
        assert_eq!(
            Node::Program {
                statements: Vec::from([if_statement, Node::EOI])
            },
            parse(source_code)
        );
    }

    #[test]
    fn test_for() {
        let source_code = r#"
            for (i:int=0; i < 10; i = i + 1) {
                print(i);
            }
        "#;
        assert_eq!(
            Node::Program {
                statements: Vec::from([
                    Node::For {
                        init: Some(Box::new(int_var_decl("i", 0))),
                        condition: Some(Box::new(var_less_than_int("i", 10))),
                        update: Some(Box::new(inc_int_var("i"))),
                        body: Vec::from([print_var("i")])
                    },
                    Node::EOI
                ])
            },
            parse(source_code)
        );
    }
    #[test]
    fn test_for_outer_var() {
        let source_code = r#"
            for (i=0; i < 10; i = i + 1) {
                print(i);
            }
        "#;
        assert_eq!(
            Node::Program {
                statements: Vec::from([
                    Node::For {
                        init: Some(Box::new(int_var_assign("i", 0))),
                        condition: Some(Box::new(var_less_than_int("i", 10))),
                        update: Some(Box::new(inc_int_var("i"))),
                        body: Vec::from([print_var("i")])
                    },
                    Node::EOI
                ])
            },
            parse(source_code)
        );
    }

    #[test]
    fn test_for_optional_init() {
        let source_code = r#"
            for (;i < 10; i = i + 1) {
                print(i);
            }
        "#;
        assert_eq!(
            Node::Program {
                statements: Vec::from([
                    Node::For {
                        init: None,
                        condition: Some(Box::new(var_less_than_int("i", 10))),
                        update: Some(Box::new(inc_int_var("i"))),
                        body: Vec::from([print_var("i")])
                    },
                    Node::EOI
                ])
            },
            parse(source_code)
        );
    }

    #[test]
    fn test_for_optional_cond() {
        let source_code = r#"
            for (i:int=0; ; i = i + 1) {
                print(i);
            }
        "#;
        assert_eq!(
            Node::Program {
                statements: Vec::from([
                    Node::For {
                        init: Some(Box::new(int_var_decl("i", 0))),
                        condition: None,
                        update: Some(Box::new(inc_int_var("i"))),
                        body: Vec::from([print_var("i")])
                    },
                    Node::EOI
                ])
            },
            parse(source_code)
        );
    }

    #[test]
    fn test_for_optional_update() {
        let source_code = r#"
            for (i:int=0; i<10;) {
                print(i);
            }
        "#;
        assert_eq!(
            Node::Program {
                statements: Vec::from([
                    Node::For {
                        init: Some(Box::new(int_var_decl("i", 0))),
                        condition: Some(Box::new(var_less_than_int("i", 10))),
                        update: None,
                        body: Vec::from([print_var("i")])
                    },
                    Node::EOI
                ])
            },
            parse(source_code)
        );
    }

    #[test]
    fn test_for_optional_init_cond() {
        let source_code = r#"
            for (; ; i = i + 1) {
                print(i);
            }
        "#;
        assert_eq!(
            Node::Program {
                statements: Vec::from([
                    Node::For {
                        init: None,
                        condition: None,
                        update: Some(Box::new(inc_int_var("i"))),
                        body: Vec::from([print_var("i")])
                    },
                    Node::EOI
                ])
            },
            parse(source_code)
        );
    }
    #[test]
    fn test_for_optional_init_cond_update() {
        let source_code = r#"
            for (;;) {
                print(i);
            }
        "#;
        assert_eq!(
            Node::Program {
                statements: Vec::from([
                    Node::For {
                        init: None,
                        condition: None,
                        update: None,
                        body: Vec::from([print_var("i")])
                    },
                    Node::EOI
                ])
            },
            parse(source_code)
        );
    }

    #[test]
    fn test_member_access() {
        let source_code = r#"
            f: Foo = Foo();
            f.a = 1;
        "#;
        println!("{:?}", parse(source_code));
        assert_eq!(
            Node::Program {
                statements: Vec::from([
                    Node::VariableDeclaration {
                        var_type: Type::Custom("Foo".to_string()),
                        name: "f".to_string(),
                        value: Box::new(Node::FunctionCall {
                            name: "Foo".to_string(),
                            arguments: Vec::new()
                        })
                    }, Node::Assignment {
                        identifier: "f.a".to_string(),
                        value: Box::new(Node::Literal(Literal::Integer(1)))
                    }, Node::EOI])
            },
            parse(source_code)
        );
    }
}
