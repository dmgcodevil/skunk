use crate::ast::Node::{Identifier, MemberAccess, StructInitialization};
use crate::parser::{Rule, SkunkParser};
use pest::iterators::Pair;
use pest::pratt_parser::{Assoc, Op, PrattParser};
use pest::Parser;

#[derive(Debug, PartialEq, Clone)]
pub enum Node {
    Program {
        statements: Vec<Node>,
    },
    // Statements
    StructDeclaration {
        name: String,
        declarations: Vec<(String, Type)>,
    },
    VariableDeclaration {
        var_type: Type,
        name: String,
        value: Option<Box<Node>>,
    },
    FunctionDeclaration {
        name: String,
        parameters: Vec<(String, Type)>,
        return_type: Type,
        body: Vec<Node>, // The function body is a list of nodes (statements or expressions)
    },
    Assignment {
        var: Box<Node>,
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
    UnaryOp {
        operator: UnaryOperator,
        operand: Box<Node>,
    },
    FunctionCall {
        name: String,         // The function name
        arguments: Vec<Node>, // The arguments are a list of expression nodes
    },
    ArrayAccess {
        coordinates: Vec<Node>,
    },
    MemberAccess {
        member: Box<Node>, // field, function
    },
    Access {
        nodes: Vec<Node>,
    },
    StructInitialization {
        name: String,
        fields: Vec<(String, Node)>, // List of field initializations (name, value)
    },
    StaticFunctionCall {
        _type: Type,
        name: String,
        arguments: Vec<Node>,
    },
    EOI,
    EMPTY,
}

#[derive(Debug, PartialEq, Clone)]
pub enum Literal {
    Integer(i64),
    StringLiteral(String),
    Boolean(bool),
}

#[derive(Debug, PartialEq, Clone)]
pub enum Operator {
    Add,
    Subtract,
    Multiply,
    Divide,
    Mod,
    Power,
    Equals,
    NotEquals,
    LessThan,
    GreaterThan,
    LessThanOrEqual,
    GreaterThanOrEqual,
    And,
    Or,
    // Not,
}

#[derive(Debug, PartialEq, Clone)]
pub enum UnaryOperator {
    Plus,
    Minus,
    Negate,
}

#[derive(Debug, PartialEq, Clone)]
pub enum Type {
    Void,
    Int,
    String,
    Boolean,
    Array {
        elem_type: Box<Type>,
        dimensions: Vec<i64>,
    },
    Custom(String), // Custom types like structs
}

pub fn type_to_string(t: &Type) -> String {
    match t {
        Type::Void => "void".to_string(),
        Type::Int => "int".to_string(),
        Type::String => "string".to_string(),
        Type::Boolean => "boolean".to_string(),
        Type::Custom(v) => v.to_string(),
        _ => panic!("unsupported type {:?}", t),
    }
}

pub fn extract_struct_field(field: &Node) -> (String, Type) {
    match field {
        Node::VariableDeclaration {
            name,
            var_type,
            value,
        } => ((*name).clone(), (*var_type).clone()),
        _ => panic!("expected VariableDeclaration node but: {:?}", field),
    }
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
        // Rule::array_alloc => create_array_alloc(pair),
        Rule::var_decl => create_var_decl(pair),
        Rule::func_decl => create_func_decl(pair),
        // Rule::not => {
        //     let mut pairs = pair.into_inner();
        //     let inner = pairs.next().unwrap();
        //     Node::Not {
        //         body: Box::new(create_ast(inner)),
        //     }
        // }
        Rule::literal => create_literal(pair),
        Rule::size => create_literal(pair),
        Rule::primary => {
            let mut pairs = pair.into_inner();
            create_ast(pairs.next().unwrap())
        }
        Rule::IDENTIFIER => Node::Identifier(pair.as_str().to_string()),
        Rule::access => create_access(pair),
        Rule::func_call => create_function_call(pair),
        Rule::static_func_call => create_static_func_call(pair),
        Rule::struct_init => create_struct_init(pair),
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
            .op(Op::infix(eq, Left))
            .op(Op::infix(not_eq, Left))
            .op(Op::infix(lt, Left))
            .op(Op::infix(lte, Left))
            .op(Op::infix(gt, Left))
            .op(Op::infix(gte, Left))
            .op(Op::infix(add, Left) | Op::infix(subtract, Left))
            .op(Op::infix(multiply, Left) | Op::infix(divide, Left) | Op::infix(modulus, Left))
            .op(Op::infix(power, Right))
            .op(Op::infix(or, Left))
            .op(Op::infix(and, Left))
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

fn create_array_access(pair: Pair<Rule>) -> Node {
    assert_eq!(Rule::array_access, pair.as_rule());
    let mut pairs = pair.into_inner();
    let mut coordinates: Vec<Node> = Vec::new();
    while let Some(dim_expr) = pairs.next() {
        coordinates.push(create_ast(dim_expr));
    }
    Node::ArrayAccess { coordinates }
}

fn create_member_access(pair: Pair<Rule>) -> Node {
    assert_eq!(Rule::member_access, pair.as_rule());
    let mut pairs = pair.into_inner();
    if let Some(inner) = pairs.next() {
        MemberAccess {
            member: Box::new(match inner.as_rule() {
                Rule::func_call => create_function_call(inner),
                Rule::IDENTIFIER => Identifier(inner.as_str().to_string()),
                _ => panic!("unsupported member access rule: {:?}", inner),
            }),
        }
    } else {
        panic!("member access tree is empty")
    }
}

fn create_chained_access(pair: Pair<Rule>) -> Vec<Node> {
    let mut nodes: Vec<Node> = Vec::new();
    let mut inner_pairs = pair.into_inner();
    while let Some(inner_pair) = inner_pairs.next() {
        match inner_pair.as_rule() {
            Rule::IDENTIFIER => nodes.push(create_identifier(inner_pair)),
            Rule::member_access => nodes.push(create_member_access(inner_pair)),
            Rule::array_access => nodes.push(create_array_access(inner_pair)),
            _ => panic!("unsupported chained access node: {:?}", inner_pair),
        }
    }
    nodes
}

fn create_access(pair: Pair<Rule>) -> Node {
    let mut nodes: Vec<Node> = Vec::new();
    let mut inner_pairs = pair.into_inner();
    while let Some(inner_pair) = inner_pairs.next() {
        match inner_pair.as_rule() {
            Rule::chained_access => nodes.extend(create_chained_access(inner_pair)),
            Rule::IDENTIFIER => nodes.push(create_identifier(inner_pair)),
            _ => panic!("unsupported rule {:?}", inner_pair),
        }
    }
    Node::Access { nodes }
}

fn create_function_call(pair: Pair<Rule>) -> Node {
    let mut inner_pairs = pair.into_inner();
    let name = inner_pairs.next().unwrap().as_str().to_string();
    let mut arguments = Vec::new();
    while let Some(arg_pair) = inner_pairs.next() {
        arguments.push(create_ast(arg_pair));
    }
    Node::FunctionCall { name, arguments }
}

fn create_static_func_call(pair: Pair<Rule>) -> Node {
    let mut inner_pairs = pair.into_inner();
    let _type = create_type(inner_pairs.next().unwrap());
    let name = inner_pairs.next().unwrap().as_str().to_string();
    let mut arguments = Vec::new();
    while let Some(arg_pair) = inner_pairs.next() {
        arguments.push(create_ast(arg_pair));
    }

    Node::StaticFunctionCall {
        _type,
        name,
        arguments,
    }
}

fn create_struct_init(pair: Pair<Rule>) -> Node {
    println!("{:?}", pair);
    let mut inner_pairs = pair.into_inner();
    let name_pair = inner_pairs.next().unwrap();
    let name = create_identifier(name_pair);
    let mut fields: Vec<(String, Node)> = Vec::new();
    if let Some(mut init_field_list) = inner_pairs.next().map(|p| p.into_inner()) {
        while let Some(p) = init_field_list.next() {
            match p.as_rule() {
                Rule::init_field => {
                    let mut init_field_pairs = p.into_inner();
                    let field_name = init_field_pairs.next().unwrap().as_str().to_string();
                    let body = create_ast(init_field_pairs.next().unwrap());
                    fields.push((field_name, body));
                }
                _ => panic!("unsupported rule {}", p),
            }
        }
    }

    if let Identifier(s) = name {
        StructInitialization { name: s, fields } // todo
    } else {
        unreachable!()
    }
}

fn create_base_type_from_str(s: &str) -> Type {
    match s {
        "int" => Type::Int,
        "string" => Type::String,
        "boolean" => Type::Boolean,
        "void" => Type::Void,
        _ => Type::Custom(s.to_string()),
    }
}

fn create_type(pair: Pair<Rule>) -> Type {
    match pair.as_rule() {
        Rule::base_type => create_base_type_from_str(pair.as_str()),
        Rule::array_type => {
            let mut inner_pairs = pair.into_inner();
            let elem_type = create_type(
                inner_pairs
                    .next()
                    .filter(|x| matches!(x.as_rule(), Rule::base_type))
                    .unwrap_or_else(|| panic!("array type is missing")),
            );
            let mut dimensions: Vec<i64> = Vec::new();
            while let Some(dim_pair) = inner_pairs.next() {
                let dim = create_ast(dim_pair.into_inner().next().unwrap());
                match dim {
                    Node::Literal(Literal::Integer(v)) => dimensions.push(v),
                    _ => panic!("incorrect array size literal: {:?}", dim),
                }
            }
            Type::Array {
                elem_type: Box::new(elem_type),
                dimensions,
            }
        }
        Rule::_type => create_type(pair.into_inner().next().unwrap()),
        _ => panic!("unexpected pair {:?}", pair),
    }
}

fn create_struct_decl(pair: Pair<Rule>) -> Node {
    let mut inner_pairs = pair.into_inner();
    let name = inner_pairs.next().unwrap().as_str().to_string();
    let mut declarations: Vec<(String, Type)> = Vec::new();
    while let Some(p) = inner_pairs.next() {
        match p.as_rule() {
            Rule::struct_field_decl => declarations.push(create_struct_field_dec(p)),
            _ => panic!("unsupported rule {}", p),
        }
    }
    Node::StructDeclaration { name, declarations }
}

fn create_struct_field_dec(pair: Pair<Rule>) -> (String, Type) {
    let mut inner_pairs = pair.into_inner();
    let field_name = inner_pairs.next().unwrap().as_str().to_string();
    let field_type = create_type(inner_pairs.next().unwrap());
    (field_name, field_type)
}

fn create_var_decl(pair: Pair<Rule>) -> Node {
    let mut inner_pairs = pair.into_inner();
    let var_name = inner_pairs.next().unwrap().as_str().to_string();
    let var_type = create_type(inner_pairs.next().unwrap());

    let body = if let Some(body_pair) = inner_pairs.next() {
        Some(Box::new(create_ast(body_pair)))
    } else {
        None
    };

    Node::VariableDeclaration {
        var_type,
        name: var_name,
        value: body,
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
                result.push((chunk[0].as_str().to_string(), create_type(chunk[1].clone())));
            });
            result
        }
        _ => panic!("unexpected  rule {}", pair),
    }
}

/*

      expression: `!a`
        primary: `!a`
          unary_op: `!a`
            negate: `!`
            primary: `a`
              access: `a`
                IDENTIFIER: `a`
*/

fn create_expression(pair: Pair<Rule>) -> Node {
    PRATT_PARSER
        .map_primary(|primary| {
            let mut inner_pairs = primary.clone().into_inner();
            let inner = inner_pairs.next().unwrap();
            match inner.as_rule() {
                Rule::unary_op => {
                    let mut unary_pairs = inner.into_inner();
                    let unary_op_pair = unary_pairs.next().unwrap();
                    let unary_operand_pair = unary_pairs.next().unwrap();
                    match unary_op_pair.as_rule() {
                        Rule::unary_plus => Node::UnaryOp {
                            operator: UnaryOperator::Plus,
                            operand: Box::new(create_ast(unary_operand_pair)),
                        },
                        Rule::unary_minus => Node::UnaryOp {
                            operator: UnaryOperator::Minus,
                            operand: Box::new(create_ast(unary_operand_pair)),
                        },
                        Rule::negate => Node::UnaryOp {
                            operator: UnaryOperator::Negate,
                            operand: Box::new(create_ast(unary_operand_pair)),
                        },
                        _ => panic!("unsupported unary operator {:?}", unary_op_pair),
                    }
                }

                _ => create_ast(primary), // todo verify should we pass inner or primary
            }
        })
        .map_infix(|lhs, op, rhs| {
            let operator = match op.as_rule() {
                // Rule::assign => Operator::Assign,
                Rule::add => Operator::Add,
                Rule::subtract => Operator::Subtract,
                Rule::multiply => Operator::Multiply,
                Rule::divide => Operator::Divide,
                Rule::modulus => Operator::Mod,
                Rule::power => Operator::Power,
                Rule::and => Operator::And,
                Rule::or => Operator::Or,
                Rule::eq => Operator::Equals,
                Rule::not_eq => Operator::NotEquals,
                // Rule::not => Operator::Not,
                Rule::lt => Operator::LessThan,
                Rule::lte => Operator::LessThanOrEqual,
                Rule::gt => Operator::GreaterThan,
                Rule::gte => Operator::GreaterThanOrEqual,
                _ => panic!("unsupported operator {:?}", op),
            };

            Node::BinaryOp {
                left: Box::new(lhs),
                operator,
                right: Box::new(rhs),
            }
        })
        .parse(pair.into_inner())
}

fn create_identifier(pair: Pair<Rule>) -> Node {
    match pair.as_rule() {
        Rule::IDENTIFIER => Identifier(pair.as_str().to_string()),
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
                create_type(inner_pairs.next().unwrap().into_inner().next().unwrap())
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
    if let Identifier(s) = name {
        // todo
        Node::FunctionDeclaration {
            name: s,
            parameters,
            return_type,
            body,
        }
    } else {
        unreachable!()
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
    let var = Box::new(create_access(inner_pairs.next().unwrap()));
    let value = Box::new(create_ast(inner_pairs.next().unwrap()));
    Node::Assignment { var, value }
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

    let body = create_body(inner_pairs);
    Node::For {
        init: init.map(Box::new),
        condition: condition.map(Box::new),
        update: update.map(Box::new),
        body,
    }
}

pub fn parse(code: &str) -> Node {
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
    use crate::ast::Node::Access;

    fn int_var_decl(name: &str, value: i64) -> Node {
        Node::VariableDeclaration {
            name: name.to_string(),
            var_type: Type::Int,
            value: Some(Box::new(Node::Literal(Literal::Integer(value)))),
        }
    }

    fn int_var_assign(name: &str, value: i64) -> Node {
        Node::Assignment {
            var: Box::new(access_var(name)),
            value: Box::new(Node::Literal(Literal::Integer(value))),
        }
    }

    fn var_less_than_int(name: &str, value: i64) -> Node {
        Node::BinaryOp {
            left: Box::new(access_var(name)),
            operator: Operator::LessThan,
            right: Box::new(Node::Literal(Literal::Integer(value))),
        }
    }

    fn inc_int_var(name: &str) -> Node {
        Node::Assignment {
            var: Box::new(access_var(name)),
            value: Box::new(Node::BinaryOp {
                left: Box::new(access_var(name)),
                operator: Operator::Add,
                right: Box::new(Node::Literal(Literal::Integer(1))),
            }),
        }
    }

    fn access_var(name: &str) -> Node {
        Access {
            nodes: name
                .split(".")
                .map(|i| Node::Identifier(i.to_string()))
                .collect(),
        }
    }

    fn field_access(name: &str, field_name: &str) -> Node {
        Access {
            nodes: [
                Identifier(name.to_string()),
                MemberAccess {
                    member: Box::new(Identifier(field_name.to_string())),
                },
            ]
            .to_vec(),
        }
    }

    fn print_int(i: i64) -> Node {
        Node::Print(Box::new(Access {
            nodes: [Node::Literal(Literal::Integer(i))].to_vec(),
        }))
    }
    fn print_var(name: &str) -> Node {
        Node::Print(Box::new(access_var(name)))
    }

    #[test]
    fn test_test_unary_minus() {
        let source_code = r#"
           -1;
        "#;
        assert_eq!(
            Node::Program {
                statements: Vec::from([
                    Node::UnaryOp {
                        operator: UnaryOperator::Minus,
                        operand: Box::new(Node::Literal(Literal::Integer(1))),
                    },
                    Node::EOI
                ])
            },
            parse(source_code)
        );
    }

    #[test]
    fn test_test_unary_plus() {
        let source_code = r#"
           +1;
        "#;
        assert_eq!(
            Node::Program {
                statements: Vec::from([
                    Node::UnaryOp {
                        operator: UnaryOperator::Plus,
                        operand: Box::new(Node::Literal(Literal::Integer(1))),
                    },
                    Node::EOI
                ])
            },
            parse(source_code)
        );
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
                        value: Some(Box::new(Node::Literal(Literal::Integer(0))))
                    },
                    Node::EOI
                ])
            },
            parse(source_code)
        )
    }

    #[test]
    fn test_var_dec() {
        let source_code = r#"
        a:int;
        "#;
        assert_eq!(
            Node::Program {
                statements: Vec::from([
                    Node::VariableDeclaration {
                        name: "a".to_string(),
                        var_type: Type::Int,
                        value: None
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
            left: Box::new(access_var("i")),
            operator: Operator::Add,
            right: Box::new(Node::Literal(Literal::Integer(1))),
        };

        assert_eq!(
            Node::Program {
                statements: Vec::from([
                    Node::Assignment {
                        var: Box::new(Node::Access {
                            nodes: [Node::Identifier("i".to_string())].to_vec()
                        }),
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
                        value: Some(Box::new(Node::BinaryOp {
                            left: Box::new(Node::Literal(Literal::Integer(2))),
                            operator: Operator::Add,
                            right: Box::new(Node::Literal(Literal::Integer(3)))
                        }))
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
                        value: Some(Box::new(Node::BinaryOp {
                            left: Box::new(Node::Literal(Literal::Integer(2))),
                            operator: Operator::Add,
                            right: Box::new(Node::BinaryOp {
                                left: Box::new(Node::Literal(Literal::Integer(3))),
                                operator: Operator::Divide,
                                right: Box::new(Node::Literal(Literal::Integer(4))),
                            })
                        }))
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
                        value: Some(Box::new(Node::BinaryOp {
                            left: Box::new(Node::BinaryOp {
                                left: Box::new(Node::Literal(Literal::Integer(2))),
                                operator: Operator::Add,
                                right: Box::new(Node::Literal(Literal::Integer(3)))
                            }),
                            operator: Operator::Divide,
                            right: Box::new(Node::Literal(Literal::Integer(4)))
                        }))
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
                            left: Box::new(access_var("a")),
                            operator: Operator::Add,
                            right: Box::new(access_var("b"))
                        }))]),
                    },
                    Node::FunctionDeclaration {
                        name: "main".to_string(),
                        parameters: Vec::new(),
                        return_type: Type::Void,
                        body: Vec::from([Node::VariableDeclaration {
                            name: "res".to_string(),
                            var_type: Type::Int,
                            value: Some(Box::new(Node::FunctionCall {
                                name: "sum".to_string(),
                                arguments: Vec::from([
                                    Node::Literal(Literal::Integer(1)),
                                    Node::Literal(Literal::Integer(2))
                                ])
                            }))
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
                            ("i".to_string(), Type::Int),
                            ("s".to_string(), Type::String),
                            ("b".to_string(), Type::Boolean),
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
            left: Box::new(access_var("a")),
            operator: Operator::And,
            right: Box::new(access_var("b")),
        };
        let var_decl = Node::VariableDeclaration {
            var_type: Type::Boolean,
            name: "c".to_string(),
            value: Some(Box::new(and)),
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
            left: Box::new(access_var("a")),
            operator: Operator::Or,
            right: Box::new(access_var("b")),
        };
        let var_declr = Node::VariableDeclaration {
            var_type: Type::Boolean,
            name: "c".to_string(),
            value: Some(Box::new(and)),
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
            left: Box::new(access_var("b")),
            operator: Operator::And,
            right: Box::new(access_var("c")),
        };
        let or = Node::BinaryOp {
            left: Box::new(access_var("a")),
            operator: Operator::Or,
            right: Box::new(and),
        };
        let var_decl = Node::VariableDeclaration {
            var_type: Type::Boolean,
            name: "d".to_string(),
            value: Some(Box::new(or)),
        };

        assert_eq!(
            Node::Program {
                statements: Vec::from([var_decl, Node::EOI])
            },
            parse(source_code)
        )
    }

    #[test]
    fn test_not() {
        let source_code = r#"
            a = !a;
        "#;
        assert_eq!(
            Node::Program {
                statements: Vec::from([
                    Node::Assignment {
                        var: Box::new(access_var("a")),
                        value: Box::new(Node::UnaryOp {
                            operator: UnaryOperator::Negate,
                            operand: Box::new(access_var("a"))
                        })
                    },
                    Node::EOI
                ])
            },
            parse(source_code)
        );
    }

    #[test]
    fn test_not_equals() {
        let source_code = r#"
            1!=2;
        "#;

        assert_eq!(
            Node::Program {
                statements: Vec::from([
                    Node::BinaryOp {
                        left: Box::new(Node::Literal(Literal::Integer(1))),
                        operator: Operator::NotEquals,
                        right: Box::new(Node::Literal(Literal::Integer(2)))
                    },
                    Node::EOI
                ])
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
            left: Box::new(access_var("a")),
            operator: Operator::LessThan,
            right: Box::new(access_var("b")),
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
        let var_a = access_var("a");
        let var_b = access_var("b");
        let lt = Node::BinaryOp {
            left: Box::new(var_a.clone()),
            operator: Operator::LessThan,
            right: Box::new(var_b.clone()),
        };
        let gt = Node::BinaryOp {
            left: Box::new(var_a.clone()),
            operator: Operator::GreaterThan,
            right: Box::new(var_b.clone()),
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
        let var_a = access_var("a");
        let var_b = access_var("b");
        let lt = Node::BinaryOp {
            left: Box::new(var_a.clone()),
            operator: Operator::LessThan,
            right: Box::new(var_b.clone()),
        };
        let gt = Node::BinaryOp {
            left: Box::new(var_a.clone()),
            operator: Operator::GreaterThan,
            right: Box::new(var_b.clone()),
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
                i = i + 1;
            }
        "#;
        assert_eq!(
            Node::Program {
                statements: Vec::from([
                    Node::For {
                        init: Some(Box::new(int_var_decl("i", 0))),
                        condition: Some(Box::new(var_less_than_int("i", 10))),
                        update: None,
                        body: Vec::from([print_var("i"), inc_int_var("i")])
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
            f: Foo = Foo{};
            f.a = 1;
        "#;
        assert_eq!(
            Node::Program {
                statements: Vec::from([
                    Node::VariableDeclaration {
                        var_type: Type::Custom("Foo".to_string()),
                        name: "f".to_string(),
                        value: Some(Box::new(Node::StructInitialization {
                            name: "Foo".to_string(),
                            fields: Vec::from([])
                        }))
                    },
                    Node::Assignment {
                        var: Box::new(field_access("f", "a")),
                        value: Box::new(Node::Literal(Literal::Integer(1)))
                    },
                    Node::EOI
                ])
            },
            parse(source_code)
        );
    }

    #[test]
    fn test_instance_creation() {
        let source_code = r#"
            p:Point = Point {
                x: 0,
                y: 1
            };
        "#;
        assert_eq!(
            Node::Program {
                statements: Vec::from([
                    Node::VariableDeclaration {
                        var_type: Type::Custom("Point".to_string()),
                        name: "p".to_string(),
                        value: Some(Box::new(Node::StructInitialization {
                            name: "Point".to_string(),
                            fields: Vec::from([
                                ("x".to_string(), Node::Literal(Literal::Integer(0))),
                                ("y".to_string(), Node::Literal(Literal::Integer(1)))
                            ])
                        }))
                    },
                    Node::EOI
                ])
            },
            parse(source_code)
        )
    }

    #[test]
    fn test_int_array() {
        let source_code = r#"
            arr: int[1] = int[1]::new(1);
        "#;
        println!("{:?}", parse(source_code));

        assert_eq!(
            Node::Program {
                statements: Vec::from([
                    Node::VariableDeclaration {
                        var_type: Type::Array {
                            elem_type: Box::new(Type::Int),
                            dimensions: Vec::from([1])
                        },
                        name: "arr".to_string(),
                        value: Some(Box::new(Node::StaticFunctionCall {
                            _type: Type::Array {
                                elem_type: Box::new(Type::Int),
                                dimensions: Vec::from([1])
                            },
                            name: "new".to_string(),
                            arguments: Vec::from([Node::Literal(Literal::Integer(1))])
                        }))
                    },
                    Node::EOI
                ])
            },
            parse(source_code)
        )
    }

    #[test]
    fn array_1d_access() {
        let source_code = r#"
            arr[1];
        "#;

        assert_eq!(
            Node::Program {
                statements: vec![
                    Node::Access {
                        nodes: vec![
                            Node::Identifier("arr".to_string()),
                            Node::ArrayAccess {
                                coordinates: vec![Node::Literal(Literal::Integer(1))],
                            },
                        ]
                    },
                    Node::EOI,
                ]
            },
            parse(source_code)
        );
    }

    #[test]
    fn array_2d_access() {
        let source_code = r#"
            arr[1][2];
        "#;

        assert_eq!(
            Node::Program {
                statements: Vec::from([
                    Access {
                        nodes: [
                            Node::Identifier("arr".to_string()),
                            Node::ArrayAccess {
                                coordinates: [
                                    Node::Literal(Literal::Integer(1)),
                                    Node::Literal(Literal::Integer(2))
                                ]
                                .to_vec()
                            }
                        ]
                        .to_vec()
                    },
                    Node::EOI
                ])
            },
            parse(source_code)
        );
    }

    #[test]
    fn test_array_1d_assignment() {
        let source_code = r#"
        arr[0] = 1;
    "#;
        assert_eq!(
            Node::Program {
                statements: Vec::from([
                    Node::Assignment {
                        var: Box::new(Node::Access {
                            nodes: vec![
                                Node::Identifier("arr".to_string()),
                                Node::ArrayAccess {
                                    coordinates: vec![Node::Literal(Literal::Integer(0))],
                                }
                            ]
                        }),
                        value: Box::new(Node::Literal(Literal::Integer(1))),
                    },
                    Node::EOI,
                ])
            },
            parse(source_code)
        );
    }

    #[test]
    fn test_array_member_assignment() {
        let source_code = r#"
        arr[0].a = 1;
    "#;
        assert_eq!(
            Node::Program {
                statements: Vec::from([
                    Node::Assignment {
                        var: Box::new(Node::Access {
                            nodes: vec![
                                Node::Identifier("arr".to_string()),
                                Node::ArrayAccess {
                                    coordinates: vec![Node::Literal(Literal::Integer(0))],
                                },
                                MemberAccess {
                                    member: Box::new(Node::Identifier("a".to_string()))
                                },
                            ]
                        }),
                        value: Box::new(Node::Literal(Literal::Integer(1))),
                    },
                    Node::EOI,
                ])
            },
            parse(source_code)
        );
    }

    #[test]
    fn test_nested_array_member_assignment() {
        let source_code = r#"
        a[0].b[1].c = 1;
    "#;
        assert_eq!(
            Node::Program {
                statements: Vec::from([
                    Node::Assignment {
                        var: Box::new(Node::Access {
                            nodes: vec![
                                Node::Identifier("a".to_string()),
                                Node::ArrayAccess {
                                    coordinates: vec![Node::Literal(Literal::Integer(0))],
                                },
                                Node::MemberAccess {
                                    member: Box::new(Node::Identifier("b".to_string()))
                                },
                                Node::ArrayAccess {
                                    coordinates: vec![Node::Literal(Literal::Integer(1))],
                                },
                                Node::MemberAccess {
                                    member: Box::new(Node::Identifier("c".to_string()))
                                },
                            ]
                        }),
                        value: Box::new(Node::Literal(Literal::Integer(1))),
                    },
                    Node::EOI,
                ])
            },
            parse(source_code)
        );
    }

    #[test]
    fn member_function_call() {
        let source_code = r#"
            a.f();
        "#;
        println!("{:?}", parse(source_code));
        assert_eq!(
            Node::Program {
                statements: [
                    Node::Access {
                        nodes: [
                            Node::Identifier("a".to_string()),
                            Node::MemberAccess {
                                member: Box::new(Node::FunctionCall {
                                    name: "f".to_string(),
                                    arguments: [].to_vec()
                                })
                            }
                        ]
                        .to_vec()
                    },
                    Node::EOI
                ]
                .to_vec()
            },
            parse(source_code)
        )
    }

    #[test]
    fn test_mod_eq() {
        let source_code = r#"
            i % 2 == 0;
        "#;
        assert_eq!(
            Node::Program {
                statements: [
                    Node::BinaryOp {
                        left: Box::new(Node::BinaryOp {
                            left: Box::new(Access {
                                nodes: [Identifier("i".to_string())].to_vec()
                            }),
                            operator: Operator::Mod,
                            right: Box::new(Node::Literal(Literal::Integer(2)))
                        }),
                        operator: Operator::Equals,
                        right: Box::new(Node::Literal(Literal::Integer(0)))
                    },
                    Node::EOI
                ]
                .to_vec()
            },
            parse(source_code)
        );
    }

    #[test]
    fn test_unary_minus_mod_equals() {
        let source_code = r#"
        -i % 2 == 0;
    "#;
        assert_eq!(
            Node::Program {
                statements: vec![
                    Node::BinaryOp {
                        left: Box::new(Node::BinaryOp {
                            left: Box::new(Node::UnaryOp {
                                operator: UnaryOperator::Minus,
                                operand: Box::new(Node::Access {
                                    nodes: vec![Node::Identifier("i".to_string())],
                                }),
                            }),
                            operator: Operator::Mod,
                            right: Box::new(Node::Literal(Literal::Integer(2))),
                        }),
                        operator: Operator::Equals,
                        right: Box::new(Node::Literal(Literal::Integer(0))),
                    },
                    Node::EOI,
                ]
            },
            parse(source_code)
        );
    }

    #[test]
    fn test_add_mod_eq() {
        let source_code = r#"
        i + 1 % 2 == 0;
    "#;

        assert_eq!(
            Node::Program {
                statements: vec![
                    Node::BinaryOp {
                        left: Box::new(Node::BinaryOp {
                            left: Box::new(Node::Access {
                                nodes: vec![Node::Identifier("i".to_string())],
                            }),
                            operator: Operator::Add,
                            right: Box::new(Node::BinaryOp {
                                left: Box::new(Node::Literal(Literal::Integer(1))),
                                operator: Operator::Mod,
                                right: Box::new(Node::Literal(Literal::Integer(2))),
                            }),
                        }),
                        operator: Operator::Equals,
                        right: Box::new(Node::Literal(Literal::Integer(0))),
                    },
                    Node::EOI,
                ]
            },
            parse(source_code)
        );
    }

    #[test]
    fn test_power_multiplication_mod_eq() {
        let source_code = r#"
            i ^ 3 * 2 % 5 == 1;
        "#;
        assert_eq!(
            Node::Program {
                statements: vec![
                    Node::BinaryOp {
                        left: Box::new(Node::BinaryOp {
                            left: Box::new(Node::BinaryOp {
                                left: Box::new(Node::BinaryOp {
                                    left: Box::new(Access {
                                        nodes: vec![Identifier("i".to_string())]
                                    }),
                                    operator: Operator::Power,
                                    right: Box::new(Node::Literal(Literal::Integer(3))),
                                }),
                                operator: Operator::Multiply,
                                right: Box::new(Node::Literal(Literal::Integer(2))),
                            }),
                            operator: Operator::Mod,
                            right: Box::new(Node::Literal(Literal::Integer(5))),
                        }),
                        operator: Operator::Equals,
                        right: Box::new(Node::Literal(Literal::Integer(1))),
                    },
                    Node::EOI,
                ]
            },
            parse(source_code)
        );
    }

    #[test]
    fn test_complex_combination_with_not_and_or() {
        let source_code = r#"
            !(i % 2 == 0) || (i + 3 > 5);
             "#;
        assert_eq!(
            Node::Program {
                statements: vec![
                    Node::BinaryOp {
                        left: Box::new(Node::UnaryOp {
                            operator: UnaryOperator::Negate,
                            operand: Box::new(Node::BinaryOp {
                                left: Box::new(Node::BinaryOp {
                                    left: Box::new(Access {
                                        nodes: vec![Identifier("i".to_string())]
                                    }),
                                    operator: Operator::Mod,
                                    right: Box::new(Node::Literal(Literal::Integer(2))),
                                }),
                                operator: Operator::Equals,
                                right: Box::new(Node::Literal(Literal::Integer(0))),
                            }),
                        }),
                        operator: Operator::Or,
                        right: Box::new(Node::BinaryOp {
                            left: Box::new(Node::BinaryOp {
                                left: Box::new(Access {
                                    nodes: vec![Identifier("i".to_string())]
                                }),
                                operator: Operator::Add,
                                right: Box::new(Node::Literal(Literal::Integer(3))),
                            }),
                            operator: Operator::GreaterThan,
                            right: Box::new(Node::Literal(Literal::Integer(5))),
                        }),
                    },
                    Node::EOI,
                ]
            },
            parse(source_code)
        );
    }

    // #[test] todo --i not supported and it's not unary operator
    // fn test_nested_unary_operations_with_arithmetic() {
    //     let source_code = r#"
    //     --i + 2;
    // "#;
    //     assert_eq!(
    //         Node::Program {
    //             statements: vec![
    //                 Node::BinaryOp {
    //                     left: Box::new(Node::UnaryOp {
    //                         operator: UnaryOperator::Negate,
    //                         operand: Box::new(Node::UnaryOp {
    //                             operator: UnaryOperator::Negate,
    //                             operand: Box::new(Access {
    //                                 nodes: vec![Identifier("i".to_string())]
    //                             }),
    //                         }),
    //                     }),
    //                     operator: Operator::Add,
    //                     right: Box::new(Node::Literal(Literal::Integer(2))),
    //                 },
    //                 Node::EOI,
    //             ]
    //         },
    //         parse(source_code)
    //     );
    // }

    #[test]
    fn test_add_subtract_multiply_divide() {
        let source_code = r#"
        (i + 1) * (j - 2) / 3;
    "#;

        assert_eq!(
            Node::Program {
                statements: vec![
                    Node::BinaryOp {
                        left: Box::new(Node::BinaryOp {
                            left: Box::new(Node::BinaryOp {
                                left: Box::new(Node::Access {
                                    nodes: vec![Node::Identifier("i".to_string())],
                                }),
                                operator: Operator::Add,
                                right: Box::new(Node::Literal(Literal::Integer(1))),
                            }),
                            operator: Operator::Multiply,
                            right: Box::new(Node::BinaryOp {
                                left: Box::new(Node::Access {
                                    nodes: vec![Node::Identifier("j".to_string())],
                                }),
                                operator: Operator::Subtract,
                                right: Box::new(Node::Literal(Literal::Integer(2))),
                            }),
                        }),
                        operator: Operator::Divide,
                        right: Box::new(Node::Literal(Literal::Integer(3))),
                    },
                    Node::EOI,
                ]
            },
            parse(source_code)
        );
    }
}
