use crate::ast::Node::{ArrayInit, Identifier, MemberAccess, StructInitialization, EMPTY};
use crate::parser::{Rule, SkunkParser};
use pest::iterators::Pair;
use pest::pratt_parser::{Assoc, Op, PrattParser};
use pest::Parser;
use std::fmt;

#[derive(Debug, PartialEq, Clone)]
pub enum Node {
    Program {
        statements: Vec<Node>,
    },
    Module {
        name: String,
    },
    Block {
        statements: Vec<Node>,
        // metadata: Metadata,
    },
    // Statements
    StructDeclaration {
        name: String,
        fields: Vec<(String, Type)>,
        functions: Vec<Node>,
    },
    VariableDeclaration {
        var_type: Type,
        name: String,
        value: Option<Box<Node>>,
        metadata: Metadata,
    },
    FunctionDeclaration {
        name: String,
        parameters: Vec<(String, Type)>,
        return_type: Type,
        body: Vec<Node>, // The function body is a list of nodes (statements or expressions)
        lambda: bool,
    },
    Assignment {
        var: Box<Node>,
        value: Box<Node>,
        metadata: Metadata,
    },
    ArrayInit {
        elements: Vec<Node>,
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
        metadata: Metadata,
    },
    ArrayAccess {
        coordinates: Vec<Node>,
    },
    MemberAccess {
        member: Box<Node>, // field, function
        metadata: Metadata,
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
        metadata: Metadata,
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
        dimensions: Vec<Node>,
    },
    Slice {
        elem_type: Box<Type>,
    },
    Custom(String), // Custom types like structs
    Function {
        parameters: Vec<Type>,
        return_type: Box<Type>,
    },
    SkSelf, // special type for member functions
}

#[derive(Debug, PartialEq, Clone)]
pub struct Span {
    pub start: usize,  // start col
    pub end: usize,    // end col
    pub line: usize,   // line number
    pub input: String, // input being parsed
}

#[derive(Debug, PartialEq, Clone)]
pub struct Metadata {
    pub span: Span,
}

impl Metadata {
    pub const EMPTY: Metadata = Metadata {
        span: Span {
            start: 0,
            end: 0,
            line: 0,
            input: String::new(),
        },
    };
}

struct PestImpl {
    metadata_creator: fn(&Pair<Rule>) -> Metadata,
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

impl PestImpl {
    fn new() -> Self {
        PestImpl {
            metadata_creator: |p| Self::create_metadata(p),
        }
    }

    pub fn parse(&self, code: &str) -> Result<Node, String> {
        match SkunkParser::parse(Rule::program, code) {
            Ok(pairs) => {
                Self::pretty_print(pairs.clone().next().unwrap(), 0);
                Ok(self.create_ast(pairs.clone().next().unwrap()))
            }
            Err(e) => Err(format!("parser failed: {}", e)),
        }
    }

    fn create_ast(&self, pair: Pair<Rule>) -> Node {
        let r = pair.as_rule();
        match r {
            Rule::program => {
                let mut statements: Vec<Node> = Vec::new();
                for inner_pair in pair.into_inner() {
                    statements.push(self.create_ast(inner_pair));
                }
                Node::Program { statements }
            }
            Rule::module => self.create_module(pair),
            Rule::statement => {
                let mut pairs = pair.into_inner();
                let inner = pairs.next().unwrap();
                self.create_ast(inner)
            }
            Rule::block => self.create_block(pair),
            Rule::expression => self.create_expression(pair),
            Rule::assignment => self.create_assignment(pair),
            Rule::struct_decl => self.create_struct_decl(pair),
            Rule::var_decl => self.create_var_decl(pair),
            Rule::var_decl_stmt => self.create_var_decl(pair),
            Rule::func_decl => self.create_func_decl(pair),
            Rule::lambda_expr => self.create_func_decl(pair),
            Rule::literal => self.create_literal(pair),
            Rule::size => self.create_literal(pair),
            Rule::primary => self.create_primary(pair),
            Rule::IDENTIFIER => Node::Identifier(pair.as_str().to_string()),
            Rule::access => self.create_access(pair),
            Rule::func_call => self.create_function_call(pair),
            Rule::static_func_call => self.create_static_func_call(pair),
            Rule::struct_init => self.create_struct_init(pair),
            Rule::inline_array_init => self.create_inline_array_init(pair),
            Rule::sk_return => {
                let mut pairs = pair.into_inner();
                Node::Return(Box::new(self.create_ast(pairs.next().unwrap())))
            }
            Rule::io => {
                let p = pair.into_inner().next().unwrap();
                match p.as_rule() {
                    Rule::print => {
                        Node::Print(Box::new(self.create_ast(p.into_inner().next().unwrap())))
                    }
                    Rule::input => Node::Input,
                    _ => panic!("unsupported IO rule"),
                }
            }
            Rule::control_flow => {
                let p = pair.into_inner().next().unwrap();
                match p.as_rule() {
                    Rule::if_expr => self.create_if_expr(p),
                    Rule::for_expr => self.create_for_classic(p),
                    _ => panic!("unsupported control flow"),
                }
            }
            Rule::EOI => Node::EOI,
            _ => {
                panic!("Unexpected Node: {:?}", pair);
            }
        }
    }

    fn create_block(&self, pair: Pair<Rule>) -> Node {
        assert_eq!(pair.as_rule(), Rule::block);
        let span = pair.as_span();
        let statements = pair.into_inner().map(|p| self.create_ast(p)).collect();
        Node::Block {
            statements,
            //    metadata: Metadata{ span: Span { start: span.start(), end: span.end() }}
        }
    }

    fn create_primary(&self, pair: Pair<Rule>) -> Node {
        assert_eq!(pair.as_rule(), Rule::primary);
        let mut primary_inner_pairs = pair.into_inner();
        assert_eq!(
            primary_inner_pairs.len(),
            1,
            "primary should have exactly one child"
        );
        let primary_child = primary_inner_pairs.next().unwrap();
        match primary_child.as_rule() {
            Rule::unary_op => {
                let mut unary_pairs = primary_child.into_inner();
                assert_eq!(
                    unary_pairs.len(),
                    2,
                    "unary pair should have exactly two children"
                );
                let unary_op_pair = unary_pairs.next().unwrap();
                let unary_operand_pair = unary_pairs.next().unwrap();
                match unary_op_pair.as_rule() {
                    Rule::unary_plus => Node::UnaryOp {
                        operator: UnaryOperator::Plus,
                        operand: Box::new(self.create_ast(unary_operand_pair)),
                    },
                    Rule::unary_minus => Node::UnaryOp {
                        operator: UnaryOperator::Minus,
                        operand: Box::new(self.create_ast(unary_operand_pair)),
                    },
                    Rule::negate => Node::UnaryOp {
                        operator: UnaryOperator::Negate,
                        operand: Box::new(self.create_ast(unary_operand_pair)),
                    },
                    _ => panic!("unsupported unary operator {:?}", unary_op_pair),
                }
            }
            _ => self.create_ast(primary_child),
        }
    }

    fn create_module(&self, pair: Pair<Rule>) -> Node {
        let mut inner_pairs = pair.into_inner();
        if let Identifier(name) = self.create_identifier(inner_pairs.next().unwrap()) {
            Node::Module {
                name: name.to_string(),
            }
        } else {
            unreachable!()
        }
    }

    fn create_literal(&self, pair: Pair<Rule>) -> Node {
        let literal = pair.into_inner().next().unwrap();
        match literal.as_rule() {
            Rule::STRING_LITERAL => {
                Node::Literal(Literal::StringLiteral(literal.as_str().to_string()))
            }
            Rule::INTEGER => {
                Node::Literal(Literal::Integer(literal.as_str().parse::<i64>().unwrap()))
            }
            Rule::BOOLEAN_LITERAL => {
                Node::Literal(Literal::Boolean(literal.as_str().parse::<bool>().unwrap()))
            }
            _ => panic!("unsupported rule {:?}", literal),
        }
    }

    fn create_array_access(&self, pair: Pair<Rule>) -> Node {
        assert_eq!(Rule::array_access, pair.as_rule());
        let mut pairs = pair.into_inner();
        let mut coordinates: Vec<Node> = Vec::new();
        while let Some(dim_expr) = pairs.next() {
            coordinates.push(self.create_ast(dim_expr));
        }
        Node::ArrayAccess { coordinates }
    }

    fn create_inline_array_init(&self, pair: Pair<Rule>) -> Node {
        let mut inner_pairs = pair.into_inner();
        let elements = inner_pairs.map(|p| self.create_ast(p)).collect();
        ArrayInit { elements }
    }

    fn create_member_access(&self, pair: Pair<Rule>) -> Node {
        assert_eq!(Rule::member_access, pair.as_rule());
        let metadata = (self.metadata_creator)(&pair);
        let mut pairs = pair.into_inner();
        if let Some(inner) = pairs.next() {
            MemberAccess {
                member: Box::new(match inner.as_rule() {
                    Rule::func_call => self.create_function_call(inner),
                    Rule::IDENTIFIER => Identifier(inner.as_str().to_string()),
                    _ => panic!("unsupported member access rule: {:?}", inner),
                }),
                metadata,
            }
        } else {
            panic!("member access tree is empty")
        }
    }

    fn create_chained_access(&self, pair: Pair<Rule>) -> Vec<Node> {
        let mut nodes: Vec<Node> = Vec::new();
        let mut inner_pairs = pair.into_inner();
        while let Some(inner_pair) = inner_pairs.next() {
            match inner_pair.as_rule() {
                Rule::IDENTIFIER => nodes.push(self.create_identifier(inner_pair)),
                Rule::member_access => nodes.push(self.create_member_access(inner_pair)),
                Rule::array_access => nodes.push(self.create_array_access(inner_pair)),
                _ => panic!("unsupported chained access node: {:?}", inner_pair),
            }
        }
        nodes
    }

    fn create_access(&self, pair: Pair<Rule>) -> Node {
        let mut nodes: Vec<Node> = Vec::new();
        let mut inner_pairs = pair.into_inner();
        while let Some(inner_pair) = inner_pairs.next() {
            match inner_pair.as_rule() {
                Rule::chained_access => nodes.extend(self.create_chained_access(inner_pair)),
                Rule::IDENTIFIER => nodes.push(self.create_identifier(inner_pair)),
                _ => panic!("unsupported rule {:?}", inner_pair),
            }
        }
        Node::Access { nodes }
    }

    fn create_function_call(&self, pair: Pair<Rule>) -> Node {
        let metadata = (self.metadata_creator)(&pair);
        let mut inner_pairs = pair.into_inner();
        let name = inner_pairs.next().unwrap().as_str().to_string();
        let mut arguments = Vec::new();
        while let Some(arg_pair) = inner_pairs.next() {
            arguments.push(self.create_ast(arg_pair));
        }
        Node::FunctionCall {
            name,
            arguments,
            metadata,
        }
    }

    fn create_static_func_call(&self, pair: Pair<Rule>) -> Node {
        let metadata = (self.metadata_creator)(&pair);
        let mut inner_pairs = pair.into_inner();
        let _type = self.create_type(inner_pairs.next().unwrap());
        let name = inner_pairs.next().unwrap().as_str().to_string();
        let mut arguments = Vec::new();
        while let Some(arg_pair) = inner_pairs.next() {
            arguments.push(self.create_ast(arg_pair));
        }

        Node::StaticFunctionCall {
            _type,
            name,
            arguments,
            metadata,
        }
    }

    fn create_struct_init(&self, pair: Pair<Rule>) -> Node {
        println!("{:?}", pair);
        let mut inner_pairs = pair.into_inner();
        let name_pair = inner_pairs.next().unwrap();
        let name = self.create_identifier(name_pair);
        let mut fields: Vec<(String, Node)> = Vec::new();
        if let Some(mut init_field_list) = inner_pairs.next().map(|p| p.into_inner()) {
            while let Some(p) = init_field_list.next() {
                match p.as_rule() {
                    Rule::init_field => {
                        let mut init_field_pairs = p.into_inner();
                        let field_name = init_field_pairs.next().unwrap().as_str().to_string();
                        let body = self.create_ast(init_field_pairs.next().unwrap());
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

    fn create_assignment(&self, pair: Pair<Rule>) -> Node {
        assert_eq!(Rule::assignment, pair.as_rule());
        let metadata = (self.metadata_creator)(&pair);
        let mut inner_pairs = pair.into_inner();
        let var = Box::new(self.create_access(inner_pairs.next().unwrap()));
        let value = Box::new(self.create_ast(inner_pairs.next().unwrap()));
        Node::Assignment {
            var,
            value,
            metadata,
        }
    }

    fn create_metadata(p: &Pair<Rule>) -> Metadata {
        let span = p.as_span();
        let line_pos = span.start_pos().line_col();
        let line_num = line_pos.0;
        let col = line_pos.1;
        let input = span.as_str().to_string();
        println!("{:?}", span);
        Metadata {
            span: Span {
                start: col,
                end: span.end(),
                line: line_num,
                input,
            },
        }
    }

    fn create_identifier(&self, pair: Pair<Rule>) -> Node {
        match pair.as_rule() {
            Rule::IDENTIFIER => Identifier(pair.as_str().to_string()),
            _ => panic!("not identifier rule {}", pair),
        }
    }

    fn create_func_decl(&self, pair: Pair<Rule>) -> Node {
        let mut inner_pairs = pair.into_inner();
        let mut lambda: bool = false;
        let name = match inner_pairs.peek().unwrap().as_rule() {
            Rule::IDENTIFIER => self.create_identifier(inner_pairs.next().unwrap()),
            _ => {
                lambda = true;
                Node::Identifier("anonymous".to_string())
            }
        };
        let parameters = self.create_param_list(inner_pairs.next().unwrap());
        let return_type = match inner_pairs.peek() {
            Some(p) => {
                if p.as_rule() == Rule::return_type {
                    self.create_type(inner_pairs.next().unwrap().into_inner().next().unwrap())
                } else {
                    Type::Void
                }
            }
            _ => Type::Void,
        };
        let mut body: Vec<Node> = Vec::new();
        while let Some(statement) = inner_pairs.next() {
            body.push(self.create_ast(statement))
        }
        if let Identifier(s) = name {
            // todo
            Node::FunctionDeclaration {
                name: s,
                parameters,
                return_type,
                body,
                lambda,
            }
        } else {
            unreachable!()
        }
    }

    fn create_type(&self, pair: Pair<Rule>) -> Type {
        match pair.as_rule() {
            Rule::base_type => create_base_type_from_str(pair.as_str()),
            Rule::function_type => {
                let mut params: Vec<Type> = Vec::new();
                let mut inner_pairs = pair.into_inner();
                match inner_pairs.peek().unwrap().as_rule() {
                    Rule::param_type_list => {
                        let param_type_list = inner_pairs.next().unwrap();
                        params.extend(
                            param_type_list
                                .into_inner()
                                .map(|p| self.create_type(p))
                                .collect::<Vec<_>>(),
                        )
                    }
                    _ => (),
                }
                if let Some(r) = inner_pairs.next() {
                    assert_eq!(r.as_rule(), Rule::_type);
                    Type::Function {
                        parameters: params,
                        return_type: Box::new(self.create_type(r)),
                    }
                } else {
                    unreachable!()
                }
            }
            Rule::slice_type => {
                let mut inner_pairs = pair.into_inner();
                Type::Slice {
                    elem_type: Box::new(self.create_type(inner_pairs.next().unwrap())),
                }
            }
            Rule::array_type => {
                let mut inner_pairs = pair.into_inner();
                let elem_type = self.create_type(
                    inner_pairs
                        .next()
                        .filter(|x| matches!(x.as_rule(), Rule::base_type)) // only arrays of primitives ?
                        .unwrap_or_else(|| panic!("array type is missing")),
                );
                let mut dimensions: Vec<Node> = Vec::new();
                while let Some(dim_pair) = inner_pairs.next() {
                    let dim = self.create_ast(dim_pair.into_inner().next().unwrap());
                    dimensions.push(dim);
                    // match dim {
                    //     Node::Literal(Literal::Integer(v)) => dimensions.push(v),
                    //     //todo support Access nodes, e.g. identifier
                    //     _ => panic!("incorrect array size literal: {:?}", dim),
                    // }
                }
                Type::Array {
                    elem_type: Box::new(elem_type),
                    dimensions,
                }
            }
            Rule::_type => self.create_type(pair.into_inner().next().unwrap()),
            _ => panic!("unexpected pair {:?}", pair),
        }
    }

    fn pretty_print(pair: Pair<Rule>, depth: usize) {
        println!(
            "{:indent$}{:?}: `{}`",
            "",
            pair.as_rule(),
            pair.as_str(),
            indent = depth * 2
        );
        for inner_pair in pair.into_inner() {
            Self::pretty_print(inner_pair, depth + 1);
        }
    }

    fn create_if_expr(&self, pair: Pair<Rule>) -> Node {
        let mut inner_pairs = pair.into_inner();
        let condition = self.create_ast(inner_pairs.next().unwrap());
        let mut body = vec![];
        while let Some(p) = inner_pairs.peek() {
            if p.as_rule() == Rule::statement {
                inner_pairs.next();
                body.push(self.create_ast(p));
            } else {
                break;
            }
        }

        // Parse optional `else if` blocks
        let mut else_if_blocks = Vec::new();
        while let Some(else_if_pair) = inner_pairs.peek() {
            if else_if_pair.as_rule() == Rule::else_if_expr {
                inner_pairs.next(); // Consume the peeked pair
                let mut elif_pairs = else_if_pair.into_inner(); // condition + body
                let else_if_condition = self.create_ast(elif_pairs.next().unwrap());
                let else_if_body = self.create_body(&mut elif_pairs);
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
            Some(self.create_body(&mut else_pair.into_inner()))
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

    fn create_body(&self, pairs: &mut pest::iterators::Pairs<Rule>) -> Vec<Node> {
        pairs.by_ref().map(|p| self.create_ast(p)).collect()
    }

    fn create_for_classic(&self, pair: Pair<Rule>) -> Node {
        let mut inner_pairs = pair.into_inner();
        assert_eq!(Rule::for_classic, inner_pairs.peek().unwrap().as_rule());
        let mut for_decl = inner_pairs.next().unwrap().into_inner();
        let init_pair = for_decl.next().unwrap();
        let init: Option<Node> = if !init_pair.as_str().is_empty() {
            let init_kind = init_pair.into_inner().next().unwrap();
            match init_kind.as_rule() {
                Rule::assignment => Some(self.create_assignment(init_kind)),
                Rule::var_decl => Some(self.create_var_decl(init_kind)),
                _ => panic!("unsupported 'for' init rule {:?}", init_kind),
            }
        } else {
            None
        };

        let condition_pair = for_decl.next().unwrap();
        let condition: Option<Node> = if !condition_pair.as_str().is_empty() {
            // todo add cond rule. create_ast is too generic
            Some(self.create_ast(condition_pair.into_inner().next().unwrap()))
        } else {
            None
        };
        let update_pair = for_decl.next().unwrap();
        let update: Option<Node> = if !update_pair.as_str().is_empty() {
            Some(self.create_assignment(update_pair.into_inner().next().unwrap()))
        } else {
            None
        };

        let body = self.create_body(&mut inner_pairs);
        Node::For {
            init: init.map(Box::new),
            condition: condition.map(Box::new),
            update: update.map(Box::new),
            body,
        }
    }

    fn create_struct_decl(&self, pair: Pair<Rule>) -> Node {
        let mut inner_pairs = pair.into_inner();
        let name = inner_pairs.next().unwrap().as_str().to_string();
        let mut fields: Vec<(String, Type)> = Vec::new();
        let mut functions = Vec::new();
        while let Some(p) = inner_pairs.next() {
            match p.as_rule() {
                Rule::struct_field_decl => fields.push(self.create_struct_field_dec(p)),
                Rule::func_decl => functions.push(self.create_func_decl(p)),
                _ => panic!("unsupported rule {}", p),
            }
        }
        Node::StructDeclaration {
            name,
            fields,
            functions,
        }
    }

    fn create_struct_field_dec(&self, pair: Pair<Rule>) -> (String, Type) {
        let mut inner_pairs = pair.into_inner();
        let field_name = inner_pairs.next().unwrap().as_str().to_string();
        let field_type = self.create_type(inner_pairs.next().unwrap());
        (field_name, field_type)
    }

    fn create_var_decl(&self, pair: Pair<Rule>) -> Node {
        let metadata = (self.metadata_creator)(&pair);
        let mut inner_pairs = pair.into_inner();
        let var_name = inner_pairs.next().unwrap().as_str().to_string();
        let var_type = self.create_type(inner_pairs.next().unwrap());

        let body = if let Some(body_pair) = inner_pairs.next() {
            Some(Box::new(self.create_ast(body_pair)))
        } else {
            None
        };

        Node::VariableDeclaration {
            var_type,
            name: var_name,
            value: body,
            metadata,
        }
    }

    fn _create_param_list(&self, pair: Pair<Rule>) -> Vec<(String, Type)> {
        match pair.as_rule() {
            Rule::_self => Vec::from([("self".to_string(), Type::SkSelf)]),
            Rule::empty_params => Vec::new(),
            Rule::param_list => {
                let mut result: Vec<(String, Type)> = Vec::new();
                let pairs: Vec<Pair<Rule>> = pair.into_inner().collect();
                assert_eq!(pairs.len() % 2, 0);
                pairs.chunks(2).for_each(|chunk| {
                    result.push((
                        chunk[0].as_str().to_string(),
                        self.create_type(chunk[1].clone()),
                    ));
                });
                result
            }
            _ => panic!("unexpected  rule {}", pair),
        }
    }

    fn create_param_list(&self, pair: Pair<Rule>) -> Vec<(String, Type)> {
        let mut res = Vec::<(String, Type)>::new();
        match pair.as_rule() {
            Rule::member_func_params => {
                let inner_pairs = pair.into_inner();
                for p in inner_pairs {
                    res.extend(self._create_param_list(p));
                }
                res
            }
            Rule::static_func_params => {
                let mut inner_pairs = pair.into_inner();
                for p in inner_pairs {
                    res.extend(self._create_param_list(p));
                }
                res
            }
            _ => panic!("unexpected  rule {}", pair),
        }
    }

    fn create_expression(&self, pair: Pair<Rule>) -> Node {
        PRATT_PARSER
            .map_primary(|primary| self.create_primary(primary))
            .map_infix(|lhs, op, rhs| {
                let operator = match op.as_rule() {
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
}

pub fn parse(input: &str) -> Node {
    PestImpl::new().parse(input).unwrap()
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
            metadata,
        } => ((*name).clone(), (*var_type).clone()),
        _ => panic!("expected VariableDeclaration node but: {:?}", field),
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

mod tests {
    use super::*;
    use crate::ast::Node::Access;

    fn parse(code: &str) -> Node {
        PestImpl {
            metadata_creator: |_| Metadata::EMPTY,
        }
        .parse(code)
        .unwrap()
    }

    fn int_var_decl(name: &str, value: i64) -> Node {
        Node::VariableDeclaration {
            name: name.to_string(),
            var_type: Type::Int,
            value: Some(Box::new(Node::Literal(Literal::Integer(value)))),
            metadata: Metadata::EMPTY,
        }
    }

    fn int_var_assign(name: &str, value: i64) -> Node {
        Node::Assignment {
            var: Box::new(access_var(name)),
            value: Box::new(Node::Literal(Literal::Integer(value))),
            metadata: Metadata::EMPTY,
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
            metadata: Metadata::EMPTY,
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
                    metadata: Metadata::EMPTY,
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
                        lambda: false,
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
                        lambda: false,
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
                        lambda: false,
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
                        value: Some(Box::new(Node::Literal(Literal::Integer(0)))),
                        metadata: Metadata::EMPTY
                    },
                    Node::EOI
                ])
            },
            parse(source_code)
        )
    }

    #[test]
    fn test_var_dec_function_call() {
        let source_code = r#"
        a:int = f();
        "#;
        assert_eq!(
            Node::Program {
                statements: Vec::from([
                    Node::VariableDeclaration {
                        name: "a".to_string(),
                        var_type: Type::Int,
                        value: Some(Box::new(Node::FunctionCall {
                            name: "f".to_string(),
                            arguments: [].to_vec(),
                            metadata: Metadata::EMPTY
                        })),
                        metadata: Metadata::EMPTY
                    },
                    Node::EOI
                ])
            },
            parse(source_code)
        )
    }

    //#[test] todo should fail
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
                        value: None,
                        metadata: Metadata::EMPTY
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
                        value: Box::new(add),
                        metadata: Metadata::EMPTY
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
                        })),
                        metadata: Metadata::EMPTY
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
                        })),
                        metadata: Metadata::EMPTY
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
                        })),
                        metadata: Metadata::EMPTY
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
                        lambda: false,
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
                                ]),
                                metadata: Metadata::EMPTY
                            })),
                            metadata: Metadata::EMPTY
                        }]),
                        lambda: false,
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
                        fields: Vec::from([
                            ("i".to_string(), Type::Int),
                            ("s".to_string(), Type::String),
                            ("b".to_string(), Type::Boolean),
                        ]),
                        functions: [].to_vec()
                    },
                    Node::EOI
                ])
            },
            parse(source_code)
        )
    }

    #[test]
    fn test_struct_decl_with_functions() {
        let source_code = r#"
        struct Point {
            x: int;
            y: int;

            function distance(): int {
                return x - y;
            }
        }
        "#;
        assert_eq!(
            Node::Program {
                statements: Vec::from([
                    Node::StructDeclaration {
                        name: "Point".to_string(),
                        fields: Vec::from([
                            ("x".to_string(), Type::Int),
                            ("y".to_string(), Type::Int),
                        ]),
                        functions: [Node::FunctionDeclaration {
                            name: "distance".to_string(),
                            parameters: [].to_vec(),
                            return_type: Type::Int,
                            body: Vec::from([Node::Return(Box::new(Node::BinaryOp {
                                left: Box::new(access_var("x")),
                                operator: Operator::Subtract,
                                right: Box::new(access_var("y"))
                            }))]),
                            lambda: false,
                        }]
                        .to_vec()
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
            metadata: Metadata::EMPTY,
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
            metadata: Metadata::EMPTY,
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
            metadata: Metadata::EMPTY,
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
                        }),
                        metadata: Metadata::EMPTY
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
    fn test_if_multiline() {
        let source_code = r#"
        if(a) {
           print(1);
           print(2);
           print(3);
        } else if(b) {
           print(1);
           print(2);
           print(3);
        } else {
           print(1);
           print(2);
           print(3);
        }
    "#;

        let var_a = access_var("a");
        let var_b = access_var("b");

        let if_body = vec![
            Node::Print(Box::new(Node::Literal(Literal::Integer(1)))),
            Node::Print(Box::new(Node::Literal(Literal::Integer(2)))),
            Node::Print(Box::new(Node::Literal(Literal::Integer(3)))),
        ];

        let else_if_body = vec![
            Node::Print(Box::new(Node::Literal(Literal::Integer(1)))),
            Node::Print(Box::new(Node::Literal(Literal::Integer(2)))),
            Node::Print(Box::new(Node::Literal(Literal::Integer(3)))),
        ];

        let else_body = vec![
            Node::Print(Box::new(Node::Literal(Literal::Integer(1)))),
            Node::Print(Box::new(Node::Literal(Literal::Integer(2)))),
            Node::Print(Box::new(Node::Literal(Literal::Integer(3)))),
        ];

        let if_statement = Node::If {
            condition: Box::new(var_a),
            body: if_body,
            else_if_blocks: vec![Node::If {
                condition: Box::new(var_b),
                body: else_if_body,
                else_if_blocks: Vec::new(),
                else_block: None,
            }],
            else_block: Some(else_body),
        };

        assert_eq!(
            Node::Program {
                statements: vec![if_statement, Node::EOI]
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
    fn test_return_in_nested_for_and_if() {
        let source_code = r#"
        function test(): int {
            for (i = 0; i < 5; i = i + 1) {
                if (i % 2 == 0) {
                    return i;
                }
            }
            return -1;
        }
        test();
        "#;
        assert_eq!(
            Node::Program {
                statements: vec![
                    Node::FunctionDeclaration {
                        name: "test".to_string(),
                        parameters: vec![],
                        return_type: Type::Int,
                        body: vec![
                            Node::For {
                                init: Some(Box::new(Node::Assignment {
                                    var: Box::new(Node::Access {
                                        nodes: vec![Node::Identifier("i".to_string())],
                                    }),
                                    value: Box::new(Node::Literal(Literal::Integer(0))),
                                    metadata: Metadata::EMPTY
                                })),
                                condition: Some(Box::new(Node::BinaryOp {
                                    left: Box::new(Node::Access {
                                        nodes: vec![Node::Identifier("i".to_string())],
                                    }),
                                    operator: Operator::LessThan,
                                    right: Box::new(Node::Literal(Literal::Integer(5))),
                                })),
                                update: Some(Box::new(Node::Assignment {
                                    var: Box::new(Node::Access {
                                        nodes: vec![Node::Identifier("i".to_string())],
                                    }),
                                    value: Box::new(Node::BinaryOp {
                                        left: Box::new(Node::Access {
                                            nodes: vec![Node::Identifier("i".to_string())],
                                        }),
                                        operator: Operator::Add,
                                        right: Box::new(Node::Literal(Literal::Integer(1))),
                                    }),
                                    metadata: Metadata::EMPTY
                                })),
                                body: vec![Node::If {
                                    condition: Box::new(Node::BinaryOp {
                                        left: Box::new(Node::BinaryOp {
                                            left: Box::new(Node::Access {
                                                nodes: vec![Node::Identifier("i".to_string())],
                                            }),
                                            operator: Operator::Mod,
                                            right: Box::new(Node::Literal(Literal::Integer(2))),
                                        }),
                                        operator: Operator::Equals,
                                        right: Box::new(Node::Literal(Literal::Integer(0))),
                                    }),
                                    body: vec![Node::Return(Box::new(Node::Access {
                                        nodes: vec![Node::Identifier("i".to_string())],
                                    })),],
                                    else_if_blocks: vec![],
                                    else_block: None,
                                },],
                            },
                            Node::Return(Box::new(Node::UnaryOp {
                                operator: UnaryOperator::Minus,
                                operand: Box::new(Node::Literal(Literal::Integer(1))),
                            })),
                        ],
                        lambda: false,
                    },
                    Node::FunctionCall {
                        name: "test".to_string(),
                        arguments: vec![],
                        metadata: Metadata::EMPTY
                    },
                    Node::EOI
                ]
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
                        })),
                        metadata: Metadata::EMPTY
                    },
                    Node::Assignment {
                        var: Box::new(field_access("f", "a")),
                        value: Box::new(Node::Literal(Literal::Integer(1))),
                        metadata: Metadata::EMPTY
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
                        })),
                        metadata: Metadata::EMPTY
                    },
                    Node::EOI
                ])
            },
            parse(source_code)
        )
    }

    #[test]
    fn test_struct_member_function_empty_params() {
        let source_code = r#"
            struct Point {
                function f(self) {
                }
            }
        "#;
        println!("{:?}", parse(source_code))
    }

    #[test]
    fn test_struct_member_function_with_params() {
        let source_code = r#"
            struct Point {
                function f(self, i:int) {
                }
            }
        "#;
        println!("{:?}", parse(source_code))
    }

    #[test]
    fn test_struct_static_function() {
        let source_code = r#"
            struct Point {
                function f() {
                }
            }
        "#;
        println!("{:?}", parse(source_code))
    }

    #[test]
    fn test_struct_static_function_with_params() {
        let source_code = r#"
            struct Point {
                function f(i:int) {
                }
            }
        "#;
        println!("{:?}", parse(source_code))
    }

    #[test]
    fn test_struct_static_function_call() {
        let source_code = r#"
            struct Point {
                function new():Point {
                    return Point{};
                }
            }

           Point::new();
        "#;

        println!("{:?}", parse(source_code))
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
                            dimensions: Vec::from([Node::Literal(Literal::Integer(1))])
                        },
                        name: "arr".to_string(),
                        value: Some(Box::new(Node::StaticFunctionCall {
                            _type: Type::Array {
                                elem_type: Box::new(Type::Int),
                                dimensions: Vec::from([Node::Literal(Literal::Integer(1))])
                            },
                            name: "new".to_string(),
                            arguments: Vec::from([Node::Literal(Literal::Integer(1))]),
                            metadata: Metadata::EMPTY
                        })),
                        metadata: Metadata::EMPTY
                    },
                    Node::EOI
                ])
            },
            parse(source_code)
        )
    }

    #[test]
    fn test_array_init_inline() {
        let source_code = r#"
        arr: int[2] = [1, 2];
        "#;

        let expected_ast = Node::Program {
            statements: vec![
                Node::VariableDeclaration {
                    name: "arr".to_string(),
                    var_type: Type::Array {
                        elem_type: Box::new(Type::Int),
                        dimensions: vec![Node::Literal(Literal::Integer(2))],
                    },
                    value: Some(Box::new(Node::ArrayInit {
                        elements: vec![
                            Node::Literal(Literal::Integer(1)),
                            Node::Literal(Literal::Integer(2)),
                        ],
                    })),
                    metadata: Metadata::EMPTY,
                },
                Node::EOI,
            ],
        };

        let program = parse(source_code);
        assert_eq!(expected_ast, program);

        println!("{:?}", program);
    }

    #[test]
    fn test_2d_array_init_inline() {
        let source_code = r#"
        arr: int[2][2] = [
            [1, 2],
            [3, 4]
        ];
        "#;
        println!("{:?}", parse(source_code));
    }

    #[test]
    fn test_3d_array_init_inline() {
        let source_code = r#"
        arr: int[2][2][1] = [
            [[1], [2]],
            [[3], [4]]
        ];
        "#;
        println!("{:?}", parse(source_code));
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
    fn test_slice_type() {
        let source_code = r#"
            slice: int[] = [1, 2, 3];
        "#;

        println!("{:?}", parse(source_code));
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
                        metadata: Metadata::EMPTY
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
                                    member: Box::new(Node::Identifier("a".to_string())),
                                    metadata: Metadata::EMPTY
                                },
                            ]
                        }),
                        value: Box::new(Node::Literal(Literal::Integer(1))),
                        metadata: Metadata::EMPTY
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
                                    member: Box::new(Node::Identifier("b".to_string())),
                                    metadata: Metadata::EMPTY
                                },
                                Node::ArrayAccess {
                                    coordinates: vec![Node::Literal(Literal::Integer(1))],
                                },
                                Node::MemberAccess {
                                    member: Box::new(Node::Identifier("c".to_string())),
                                    metadata: Metadata::EMPTY
                                },
                            ]
                        }),
                        value: Box::new(Node::Literal(Literal::Integer(1))),
                        metadata: Metadata::EMPTY
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
                                    arguments: [].to_vec(),
                                    metadata: Metadata::EMPTY
                                }),
                                metadata: Metadata::EMPTY
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

    #[test]
    fn test_nested_unary_operations() {
        let source_code = r#"
        !!(a && !b);
    "#;
        assert_eq!(
            Node::Program {
                statements: vec![
                    Node::UnaryOp {
                        operator: UnaryOperator::Negate,
                        operand: Box::new(Node::UnaryOp {
                            operator: UnaryOperator::Negate,
                            operand: Box::new(Node::BinaryOp {
                                left: Box::new(Node::Access {
                                    nodes: vec![Node::Identifier("a".to_string())],
                                }),
                                operator: Operator::And,
                                right: Box::new(Node::UnaryOp {
                                    operator: UnaryOperator::Negate,
                                    operand: Box::new(Node::Access {
                                        nodes: vec![Node::Identifier("b".to_string())],
                                    })
                                })
                            })
                        })
                    },
                    Node::EOI
                ]
            },
            parse(source_code)
        );
    }

    #[test]
    fn test_complex_unary_operations() {
        let source_code = r#"
        -a * +b / !(c - d);
    "#;
        assert_eq!(
            Node::Program {
                statements: vec![
                    Node::BinaryOp {
                        left: Box::new(Node::BinaryOp {
                            left: Box::new(Node::UnaryOp {
                                operator: UnaryOperator::Minus,
                                operand: Box::new(Node::Access {
                                    nodes: vec![Node::Identifier("a".to_string())],
                                })
                            }),
                            operator: Operator::Multiply,
                            right: Box::new(Node::UnaryOp {
                                operator: UnaryOperator::Plus,
                                operand: Box::new(Node::Access {
                                    nodes: vec![Node::Identifier("b".to_string())],
                                })
                            })
                        }),
                        operator: Operator::Divide,
                        right: Box::new(Node::UnaryOp {
                            operator: UnaryOperator::Negate,
                            operand: Box::new(Node::BinaryOp {
                                left: Box::new(Node::Access {
                                    nodes: vec![Node::Identifier("c".to_string())],
                                }),
                                operator: Operator::Subtract,
                                right: Box::new(Node::Access {
                                    nodes: vec![Node::Identifier("d".to_string())],
                                })
                            })
                        })
                    },
                    Node::EOI
                ]
            },
            parse(source_code)
        );
    }

    #[test]
    fn test_nested_block() {
        let source_code = r#"
        function f() {
            i: int = 1;
            {
                i: int = 2;
            }
        }
        "#;

        assert_eq!(
            Node::Program {
                statements: vec![
                    Node::FunctionDeclaration {
                        name: "f".to_string(),
                        parameters: vec![],
                        return_type: Type::Void,
                        body: vec![
                            Node::VariableDeclaration {
                                var_type: Type::Int,
                                name: "i".to_string(),
                                value: Some(Box::new(Node::Literal(Literal::Integer(1)))),
                                metadata: Metadata::EMPTY
                            },
                            Node::Block {
                                statements: vec![Node::VariableDeclaration {
                                    var_type: Type::Int,
                                    name: "i".to_string(),
                                    value: Some(Box::new(Node::Literal(Literal::Integer(2)))),
                                    metadata: Metadata::EMPTY
                                }],
                            },
                        ],
                        lambda: false,
                    },
                    Node::EOI
                ]
            },
            parse(source_code)
        );
    }

    #[test]
    fn test_module() {
        let source_code = r#"
            module A;
        "#;
        assert_eq!(
            Node::Program {
                statements: vec![
                    Node::Module {
                        name: "A".to_string()
                    },
                    Node::EOI
                ]
            },
            parse(source_code)
        );
    }

    #[test]
    fn test_assign_function_to_var() {
        let source_code = r#"
        sum: (int, int) -> int = function (a:int, b:int): int {
            return a+b;
        }
    "#;

        assert_eq!(
            Node::Program {
                statements: Vec::from([
                    Node::VariableDeclaration {
                        name: "sum".to_string(),
                        var_type: Type::Function {
                            parameters: vec![Type::Int, Type::Int],
                            return_type: Box::new(Type::Int),
                        },
                        value: Some(Box::new(Node::FunctionDeclaration {
                            name: "anonymous".to_string(),
                            parameters: vec![
                                ("a".to_string(), Type::Int),
                                ("b".to_string(), Type::Int),
                            ],
                            return_type: Type::Int,
                            body: vec![Node::Return(Box::new(Node::BinaryOp {
                                left: Box::new(Node::Access {
                                    nodes: vec![Node::Identifier("a".to_string())],
                                }),
                                operator: Operator::Add,
                                right: Box::new(Node::Access {
                                    nodes: vec![Node::Identifier("b".to_string())],
                                }),
                            }))],
                            lambda: true,
                        })),
                        metadata: Metadata::EMPTY
                    },
                    Node::EOI
                ])
            },
            parse(source_code)
        )
    }

    #[test]
    fn test_function_type_empty_params() {
        let source_code = r#"
        no_args: () -> int = function (): int {
            return 42;
        }
    "#;

        assert_eq!(
            Node::Program {
                statements: Vec::from([
                    Node::VariableDeclaration {
                        name: "no_args".to_string(),
                        var_type: Type::Function {
                            parameters: vec![], // Empty parameter list
                            return_type: Box::new(Type::Int),
                        },
                        value: Some(Box::new(Node::FunctionDeclaration {
                            name: "anonymous".to_string(),
                            parameters: vec![], // No parameters in the declaration
                            return_type: Type::Int,
                            body: vec![Node::Return(Box::new(Node::Literal(Literal::Integer(42))))],
                            lambda: true,
                        })),
                        metadata: Metadata::EMPTY
                    },
                    Node::EOI
                ])
            },
            parse(source_code)
        )
    }

    #[test]
    fn anonymous_function() {
        let source_code = r#"
        f(function ():int {
            return 47;
        });
        "#;

        println!("{:?}", parse(source_code));
    }

    #[test]
    fn test_return_function() {
        let source_code = r#"
        function f(): (int) -> int {
            return function(a:int):int {
                return a;
            }
        }
        g: (int) -> int = f();
        g(47);
        "#;
        println!("{:?}", parse(source_code));
    }
}
