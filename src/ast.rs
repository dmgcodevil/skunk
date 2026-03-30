use crate::ast::Node::{ArrayInit, Identifier, MemberAccess, StructInitialization, EMPTY};
use crate::parser::{Rule, SkunkParser};
use pest::iterators::Pair;
use pest::pratt_parser::{Assoc, Op, PrattParser};
use pest::Parser;
use std::collections::HashMap;
use std::fmt;

#[derive(Debug, PartialEq, Clone)]
pub enum Node {
    Program {
        statements: Vec<Node>,
    },
    Module {
        name: String,
    },
    Import {
        name: String,
    },
    Export {
        declaration: Box<Node>,
    },
    Block {
        statements: Vec<Node>,
        // metadata: Metadata,
    },
    UnsafeBlock {
        statements: Vec<Node>,
    },
    // Statements
    StructDeclaration {
        name: String,
        fields: Vec<(String, Type)>,
        functions: Vec<Node>,
    },
    TraitDeclaration {
        name: String,
        methods: Vec<TraitMethodSignature>,
    },
    ImplDeclaration {
        generic_params: Vec<String>,
        generic_bounds: HashMap<String, Vec<String>>,
        trait_names: Vec<String>,
        target_type: Type,
    },
    EnumDeclaration {
        name: String,
        variants: Vec<EnumVariant>,
    },
    GenericStructDeclaration {
        name: String,
        generic_params: Vec<String>,
        generic_bounds: HashMap<String, Vec<String>>,
        fields: Vec<(String, Type)>,
        functions: Vec<Node>,
    },
    GenericEnumDeclaration {
        name: String,
        generic_params: Vec<String>,
        generic_bounds: HashMap<String, Vec<String>>,
        variants: Vec<EnumVariant>,
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
    GenericFunctionDeclaration {
        name: String,
        generic_params: Vec<String>,
        generic_bounds: HashMap<String, Vec<String>>,
        parameters: Vec<(String, Type)>,
        return_type: Type,
        body: Vec<Node>,
        lambda: bool,
    },
    Assignment {
        var: Box<Node>,
        value: Box<Node>,
        metadata: Metadata,
    },
    StructDestructure {
        struct_type: Type,
        fields: Vec<StructPatternField>,
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
    Match {
        value: Box<Node>,
        cases: Vec<MatchCase>,
    },
    For {
        init: Option<Box<Node>>,      // Initialization is a statement node
        condition: Option<Box<Node>>, // The condition is an expression node
        update: Option<Box<Node>>,    // Update is a statement node
        body: Vec<Node>,              // The body is a list of nodes
    },
    Return(Option<Box<Node>>), // The return value is an expression node
    Print(Box<Node>),          // The print expression is an expression node
    Input,                     // Read data from keyboard

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
        name: String,              // The function name
        type_arguments: Vec<Type>,
        arguments: Vec<Vec<Node>>, // The arguments are a list of expression nodes
        metadata: Metadata,
    },
    ArrayAccess {
        coordinates: Vec<Node>,
    },
    SliceAccess {
        start: Option<Box<Node>>,
        end: Option<Box<Node>>,
    },
    MemberAccess {
        member: Box<Node>, // field, function
        metadata: Metadata,
    },
    Dereference {
        metadata: Metadata,
    },
    Access {
        nodes: Vec<Node>,
    },
    StructInitialization {
        _type: Type,
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
pub struct EnumVariant {
    pub name: String,
    pub payload_types: Vec<Type>,
}

#[derive(Debug, PartialEq, Clone)]
pub struct TraitMethodSignature {
    pub name: String,
    pub parameters: Vec<(String, Type)>,
    pub return_type: Type,
}

#[derive(Debug, PartialEq, Clone)]
pub struct MatchCase {
    pub pattern: MatchPattern,
    pub body: Vec<Node>,
}

#[derive(Debug, PartialEq, Clone)]
pub struct StructPatternField {
    pub field_name: String,
    pub binding: String,
}

#[derive(Debug, PartialEq, Clone)]
pub enum MatchPattern {
    EnumVariant {
        enum_type: Option<Type>,
        variant: String,
        bindings: Vec<String>,
    },
    Struct {
        struct_type: Type,
        fields: Vec<StructPatternField>,
    },
}

#[derive(Debug, PartialEq, Clone)]
pub enum Literal {
    Integer(i64),
    Long(i64),
    Float(f32),
    Double(f64),
    StringLiteral(String),
    Boolean(bool),
    Char(char),
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
    AddressOf,
}

#[derive(Debug, PartialEq, Clone)]
pub enum Type {
    Void,
    Byte,
    Short,
    Int,
    Long,
    Float,
    Double,
    String,
    Boolean,
    Char,
    Const {
        inner: Box<Type>,
    },
    BindingConst {
        inner: Box<Type>,
    },
    Array {
        elem_type: Box<Type>,
        dimensions: Vec<Node>,
    },
    Pointer {
        target_type: Box<Type>,
    },
    Slice {
        elem_type: Box<Type>,
    },
    Allocator,
    Arena,
    GenericInstance {
        base: String,
        type_arguments: Vec<Type>,
    },
    Custom(String), // Custom types like structs
    Function {
        parameters: Vec<Type>,
        return_type: Box<Type>,
    },
    MutSelf, // special type for mut member functions
    SkSelf,  // special type for member functions
}

enum TypePrefix {
    Pointer,
    Slice,
    Array(Node),
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
            Ok(pairs) => Ok(self.create_ast(pairs.clone().next().unwrap())),
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
            Rule::import => self.create_import(pair),
            Rule::export_decl => self.create_export(pair),
            Rule::statement => {
                let mut pairs = pair.into_inner();
                let inner = pairs.next().unwrap();
                self.create_ast(inner)
            }
            Rule::block => self.create_block(pair),
            Rule::unsafe_block => self.create_unsafe_block(pair),
            Rule::expression => self.create_expression(pair),
            Rule::assignment => self.create_assignment(pair),
            Rule::struct_decl => self.create_struct_decl(pair),
            Rule::enum_decl => self.create_enum_decl(pair),
            Rule::trait_decl => self.create_trait_decl(pair),
            Rule::impl_decl => self.create_impl_decl(pair),
            Rule::var_decl => self.create_var_decl(pair),
            Rule::var_decl_stmt => self.create_var_decl(pair.into_inner().next().unwrap()),
            Rule::struct_destructure => self.create_struct_destructure(pair),
            Rule::struct_destructure_stmt => {
                self.create_struct_destructure(pair.into_inner().next().unwrap())
            }
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
                if pairs.len() != 0 {
                    Node::Return(Some(Box::new(self.create_ast(pairs.next().unwrap()))))
                } else {
                    Node::Return(None)
                }
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
                    Rule::match_expr => self.create_match_expr(p),
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

    fn create_unsafe_block(&self, pair: Pair<Rule>) -> Node {
        assert_eq!(pair.as_rule(), Rule::unsafe_block);
        let statements = pair.into_inner().map(|p| self.create_ast(p)).collect();
        Node::UnsafeBlock { statements }
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
                    Rule::address_of => Node::UnaryOp {
                        operator: UnaryOperator::AddressOf,
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
        Node::Module {
            name: self.create_qualified_name(inner_pairs.next().unwrap()),
        }
    }

    fn create_import(&self, pair: Pair<Rule>) -> Node {
        let mut inner_pairs = pair.into_inner();
        Node::Import {
            name: self.create_qualified_name(inner_pairs.next().unwrap()),
        }
    }

    fn create_export(&self, pair: Pair<Rule>) -> Node {
        let mut inner_pairs = pair.into_inner();
        Node::Export {
            declaration: Box::new(self.create_ast(inner_pairs.next().unwrap())),
        }
    }

    fn create_literal(&self, pair: Pair<Rule>) -> Node {
        let literal = pair.into_inner().next().unwrap();
        match literal.as_rule() {
            Rule::STRING_LITERAL => {
                Node::Literal(Literal::StringLiteral(literal.as_str().to_string()))
            }
            Rule::LONG_LITERAL => Node::Literal(Literal::Long(
                literal.as_str()[..literal.as_str().len() - 1]
                    .parse::<i64>()
                    .unwrap(),
            )),
            Rule::FLOAT_LITERAL => Node::Literal(Literal::Float(
                literal.as_str()[..literal.as_str().len() - 1]
                    .parse::<f32>()
                    .unwrap(),
            )),
            Rule::DOUBLE_LITERAL => {
                Node::Literal(Literal::Double(literal.as_str().parse::<f64>().unwrap()))
            }
            Rule::INTEGER => {
                Node::Literal(Literal::Integer(literal.as_str().parse::<i64>().unwrap()))
            }
            Rule::BOOLEAN_LITERAL => {
                Node::Literal(Literal::Boolean(literal.as_str().parse::<bool>().unwrap()))
            }
            Rule::CHAR_LITERAL => {
                Node::Literal(Literal::Char(parse_char_literal(literal.as_str()).unwrap()))
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

    fn create_slice_access(&self, pair: Pair<Rule>) -> Node {
        assert_eq!(Rule::slice_access, pair.as_rule());
        let mut pairs = pair.into_inner();
        let start = pairs
            .next()
            .and_then(|p| p.into_inner().next())
            .map(|p| Box::new(self.create_ast(p)));
        let end = pairs
            .next()
            .and_then(|p| p.into_inner().next())
            .map(|p| Box::new(self.create_ast(p)));
        Node::SliceAccess { start, end }
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

    fn create_dereference_access(&self, pair: Pair<Rule>) -> Node {
        assert_eq!(Rule::deref_access, pair.as_rule());
        Node::Dereference {
            metadata: (self.metadata_creator)(&pair),
        }
    }

    fn create_chained_access(&self, pair: Pair<Rule>) -> Vec<Node> {
        let mut nodes: Vec<Node> = Vec::new();
        let mut inner_pairs = pair.into_inner();
        while let Some(inner_pair) = inner_pairs.next() {
            let mut step_pair = inner_pair;
            if step_pair.as_rule() == Rule::access_step {
                step_pair = step_pair.into_inner().next().unwrap();
            }
            match step_pair.as_rule() {
                Rule::IDENTIFIER => nodes.push(self.create_identifier(step_pair)),
                Rule::member_access => nodes.push(self.create_member_access(step_pair)),
                Rule::deref_access => nodes.push(self.create_dereference_access(step_pair)),
                Rule::array_access => nodes.push(self.create_array_access(step_pair)),
                Rule::slice_access => nodes.push(self.create_slice_access(step_pair)),
                _ => panic!("unsupported chained access node: {:?}", step_pair),
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

    fn create_arg_list(&self, pair: Pair<Rule>) -> Vec<Node> {
        assert_eq!(pair.as_rule(), Rule::arg_list);
        let mut pairs = pair.into_inner();
        let mut args: Vec<Node> = Vec::new();
        while let Some(inner_pair) = pairs.next() {
            args.push(self.create_ast(inner_pair));
        }
        args
    }

    fn create_function_call(&self, pair: Pair<Rule>) -> Node {
        let metadata = (self.metadata_creator)(&pair);
        let mut inner_pairs = pair.into_inner().peekable();
        let name = inner_pairs.next().unwrap().as_str().to_string();
        let type_arguments = if inner_pairs
            .peek()
            .is_some_and(|pair| pair.as_rule() == Rule::type_arg_list)
        {
            inner_pairs
                .next()
                .unwrap()
                .into_inner()
                .map(|pair| self.create_type(pair))
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        let mut arguments = Vec::new();
        while let Some(arg_list) = inner_pairs.next() {
            arguments.push(self.create_arg_list(arg_list));
        }
        Node::FunctionCall {
            name,
            type_arguments,
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
        let mut inner_pairs = pair.into_inner();
        let struct_type = self.create_type(inner_pairs.next().unwrap());
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

        StructInitialization {
            _type: struct_type,
            fields,
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
        let mut generic_params = Vec::<String>::new();
        let mut generic_bounds = HashMap::<String, Vec<String>>::new();
        let name = match inner_pairs.peek().unwrap().as_rule() {
            Rule::IDENTIFIER => self.create_identifier(inner_pairs.next().unwrap()),
            _ => {
                lambda = true;
                Node::Identifier("anonymous".to_string())
            }
        };
        if !lambda {
            if let Some(peeked) = inner_pairs.peek() {
                if peeked.as_rule() == Rule::generic_params {
                    (generic_params, generic_bounds) =
                        self.create_generic_params(inner_pairs.next().unwrap());
                }
            }
        }
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
            if generic_params.is_empty() {
                Node::FunctionDeclaration {
                    name: s,
                    parameters,
                    return_type,
                    body,
                    lambda,
                }
            } else {
                Node::GenericFunctionDeclaration {
                    name: s,
                    generic_params,
                    generic_bounds,
                    parameters,
                    return_type,
                    body,
                    lambda,
                }
            }
        } else {
            unreachable!()
        }
    }

    fn create_type_prefix(&self, pair: Pair<Rule>) -> TypePrefix {
        assert_eq!(pair.as_rule(), Rule::type_prefix);
        let mut inner = pair.into_inner();
        if let Some(inner) = inner.next() {
            match inner.as_rule() {
                Rule::pointer_prefix => TypePrefix::Pointer,
                Rule::expression => TypePrefix::Array(self.create_ast(inner)),
                _ => TypePrefix::Slice,
            }
        } else {
            TypePrefix::Slice
        }
    }

    fn apply_type_prefixes(&self, base_type: Type, prefixes: Vec<TypePrefix>) -> Type {
        let mut current = base_type;
        for prefix in prefixes.into_iter().rev() {
            match prefix {
                TypePrefix::Array(dimension) => match current {
                    Type::Array {
                        elem_type,
                        mut dimensions,
                    } => {
                        dimensions.insert(0, dimension);
                        current = Type::Array {
                            elem_type,
                            dimensions,
                        };
                    }
                    other => {
                        current = Type::Array {
                            elem_type: Box::new(other),
                            dimensions: vec![dimension],
                        };
                    }
                },
                TypePrefix::Slice => {
                    current = Type::Slice {
                        elem_type: Box::new(current),
                    };
                }
                TypePrefix::Pointer => {
                    current = Type::Pointer {
                        target_type: Box::new(current),
                    };
                }
            }
        }
        current
    }

    fn create_type(&self, pair: Pair<Rule>) -> Type {
        match pair.as_rule() {
            Rule::base_type => self.create_type(pair.into_inner().next().unwrap()),
            Rule::builtin_type => create_base_type_from_str(pair.as_str()),
            Rule::nominal_type => self.create_nominal_type(pair),
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
            Rule::prefixed_type => {
                let mut inner_pairs: Vec<Pair<Rule>> = pair.into_inner().collect();
                let base_pair = inner_pairs.pop().unwrap();
                let mut base_type = self.create_type(base_pair);
                if inner_pairs
                    .last()
                    .is_some_and(|p| p.as_rule() == Rule::const_kw)
                {
                    inner_pairs.pop();
                    base_type = Type::Const {
                        inner: Box::new(base_type),
                    };
                }
                let prefixes = inner_pairs
                    .into_iter()
                    .map(|p| self.create_type_prefix(p))
                    .collect();
                self.apply_type_prefixes(base_type, prefixes)
            }
            Rule::legacy_slice_type => {
                let mut inner_pairs = pair.into_inner();
                Type::Slice {
                    elem_type: Box::new(self.create_type(inner_pairs.next().unwrap())),
                }
            }
            Rule::legacy_array_type => {
                let mut inner_pairs = pair.into_inner();
                let elem_type = self.create_type(
                    inner_pairs
                        .next()
                        .filter(|x| matches!(x.as_rule(), Rule::base_type))
                        .unwrap_or_else(|| panic!("array type is missing")),
                );
                let mut dimensions: Vec<Node> = Vec::new();
                while let Some(dim_pair) = inner_pairs.next() {
                    let dim = self.create_ast(dim_pair.into_inner().next().unwrap());
                    dimensions.push(dim);
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

    fn create_match_expr(&self, pair: Pair<Rule>) -> Node {
        assert_eq!(pair.as_rule(), Rule::match_expr);
        let mut inner_pairs = pair.into_inner();
        let value = self.create_ast(inner_pairs.next().unwrap());
        let cases = inner_pairs
            .map(|case_pair| self.create_match_case(case_pair))
            .collect();
        Node::Match {
            value: Box::new(value),
            cases,
        }
    }

    fn create_match_case(&self, pair: Pair<Rule>) -> MatchCase {
        assert_eq!(pair.as_rule(), Rule::match_case);
        let mut inner_pairs = pair.into_inner();
        let pattern = self.create_match_pattern(inner_pairs.next().unwrap());
        let body = inner_pairs.map(|p| self.create_ast(p)).collect();
        MatchCase { pattern, body }
    }

    fn create_match_pattern(&self, pair: Pair<Rule>) -> MatchPattern {
        assert_eq!(pair.as_rule(), Rule::match_pattern);
        let inner = pair.into_inner().next().unwrap();
        match inner.as_rule() {
            Rule::enum_match_pattern => self.create_enum_match_pattern(inner),
            Rule::struct_match_pattern => self.create_struct_match_pattern(inner),
            other => panic!("unexpected match pattern rule {:?}", other),
        }
    }

    fn create_enum_match_pattern(&self, pair: Pair<Rule>) -> MatchPattern {
        assert_eq!(pair.as_rule(), Rule::enum_match_pattern);
        let mut inner_pairs = pair.into_inner().peekable();
        let enum_type = if inner_pairs
            .peek()
            .is_some_and(|p| p.as_rule() == Rule::nominal_type)
        {
            Some(self.create_nominal_type(inner_pairs.next().unwrap()))
        } else {
            None
        };
        let variant = inner_pairs.next().unwrap().as_str().to_string();
        let bindings = inner_pairs.map(|p| p.as_str().to_string()).collect::<Vec<_>>();
        MatchPattern::EnumVariant {
            enum_type,
            variant,
            bindings,
        }
    }

    fn create_struct_pattern_fields(&self, pair: Pair<Rule>) -> Vec<StructPatternField> {
        assert_eq!(pair.as_rule(), Rule::struct_pattern_fields);
        pair.into_inner()
            .map(|field| {
                let mut inner = field.into_inner();
                let field_name = inner.next().unwrap().as_str().to_string();
                let binding = inner
                    .next()
                    .map(|pair| pair.as_str().to_string())
                    .unwrap_or_else(|| field_name.clone());
                StructPatternField {
                    field_name,
                    binding,
                }
            })
            .collect()
    }

    fn create_struct_match_pattern(&self, pair: Pair<Rule>) -> MatchPattern {
        assert_eq!(pair.as_rule(), Rule::struct_match_pattern);
        let mut inner = pair.into_inner();
        let struct_type = self.create_nominal_type(inner.next().unwrap());
        let fields = self.create_struct_pattern_fields(inner.next().unwrap());
        MatchPattern::Struct {
            struct_type,
            fields,
        }
    }

    fn create_struct_destructure(&self, pair: Pair<Rule>) -> Node {
        assert_eq!(pair.as_rule(), Rule::struct_destructure);
        let metadata = (self.metadata_creator)(&pair);
        let mut inner = pair.into_inner();
        let struct_type = self.create_nominal_type(inner.next().unwrap());
        let fields = self.create_struct_pattern_fields(inner.next().unwrap());
        let value = Box::new(self.create_ast(inner.next().unwrap()));
        Node::StructDestructure {
            struct_type,
            fields,
            value,
            metadata,
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
        let mut generic_params = Vec::<String>::new();
        let mut generic_bounds = HashMap::<String, Vec<String>>::new();
        if let Some(peeked) = inner_pairs.peek() {
            if peeked.as_rule() == Rule::generic_params {
                (generic_params, generic_bounds) =
                    self.create_generic_params(inner_pairs.next().unwrap());
            }
        }
        let mut fields: Vec<(String, Type)> = Vec::new();
        let mut functions = Vec::new();
        while let Some(p) = inner_pairs.next() {
            match p.as_rule() {
                Rule::struct_field_decl => fields.push(self.create_struct_field_dec(p)),
                Rule::func_decl => functions.push(self.create_func_decl(p)),
                _ => panic!("unsupported rule {}", p),
            }
        }
        if generic_params.is_empty() {
            Node::StructDeclaration {
                name,
                fields,
                functions,
            }
        } else {
            Node::GenericStructDeclaration {
                name,
                generic_params,
                generic_bounds,
                fields,
                functions,
            }
        }
    }

    fn create_enum_decl(&self, pair: Pair<Rule>) -> Node {
        assert_eq!(pair.as_rule(), Rule::enum_decl);
        let mut inner_pairs = pair.into_inner();
        let name = inner_pairs.next().unwrap().as_str().to_string();
        let mut generic_params = Vec::<String>::new();
        let mut generic_bounds = HashMap::<String, Vec<String>>::new();
        if let Some(peeked) = inner_pairs.peek() {
            if peeked.as_rule() == Rule::generic_params {
                (generic_params, generic_bounds) =
                    self.create_generic_params(inner_pairs.next().unwrap());
            }
        }
        let variants = inner_pairs
            .map(|p| self.create_enum_variant_decl(p))
            .collect::<Vec<_>>();
        if generic_params.is_empty() {
            Node::EnumDeclaration { name, variants }
        } else {
            Node::GenericEnumDeclaration {
                name,
                generic_params,
                generic_bounds,
                variants,
            }
        }
    }

    fn create_trait_decl(&self, pair: Pair<Rule>) -> Node {
        assert_eq!(pair.as_rule(), Rule::trait_decl);
        let mut inner_pairs = pair.into_inner();
        let name = inner_pairs.next().unwrap().as_str().to_string();
        let methods = inner_pairs
            .map(|p| self.create_trait_method_decl(p))
            .collect::<Vec<_>>();
        Node::TraitDeclaration { name, methods }
    }

    fn create_trait_method_decl(&self, pair: Pair<Rule>) -> TraitMethodSignature {
        assert_eq!(pair.as_rule(), Rule::trait_method_decl);
        let mut inner_pairs = pair.into_inner();
        let name = inner_pairs.next().unwrap().as_str().to_string();
        let parameters = self.create_param_list(inner_pairs.next().unwrap());
        let return_type = match inner_pairs.next() {
            Some(pair) => self.create_type(pair.into_inner().next().unwrap()),
            None => Type::Void,
        };
        TraitMethodSignature {
            name,
            parameters,
            return_type,
        }
    }

    fn create_impl_decl(&self, pair: Pair<Rule>) -> Node {
        assert_eq!(pair.as_rule(), Rule::impl_decl);
        let mut inner_pairs = pair.into_inner().peekable();
        let (generic_params, generic_bounds) = if inner_pairs
            .peek()
            .is_some_and(|pair| pair.as_rule() == Rule::generic_params)
        {
            self.create_generic_params(inner_pairs.next().unwrap())
        } else {
            (Vec::new(), HashMap::new())
        };
        let mut rest = inner_pairs.collect::<Vec<_>>();
        let target_type = self.create_type(rest.pop().unwrap());
        let trait_names = rest
            .into_iter()
            .map(|p| p.as_str().to_string())
            .collect();
        Node::ImplDeclaration {
            generic_params,
            generic_bounds,
            trait_names,
            target_type,
        }
    }

    fn create_enum_variant_decl(&self, pair: Pair<Rule>) -> EnumVariant {
        assert_eq!(pair.as_rule(), Rule::enum_variant_decl);
        let mut inner_pairs = pair.into_inner();
        let name = inner_pairs.next().unwrap().as_str().to_string();
        let payload_types = inner_pairs.map(|p| self.create_type(p)).collect::<Vec<_>>();
        EnumVariant {
            name,
            payload_types,
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
        let mut is_const = false;
        if inner_pairs
            .peek()
            .is_some_and(|pair| pair.as_rule() == Rule::const_kw)
        {
            inner_pairs.next();
            is_const = true;
        }
        let var_name = inner_pairs.next().unwrap().as_str().to_string();
        let mut var_type = self.create_type(inner_pairs.next().unwrap());
        if is_const {
            var_type = Type::BindingConst {
                inner: Box::new(var_type),
            };
        }

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
            Rule::_self => {
                let mut inner_pairs = pair.into_inner();
                let mut self_type = Type::SkSelf;
                if let Some(qualifier) = inner_pairs.next() {
                    self_type = match qualifier.as_rule() {
                        Rule::mut_kw => Type::MutSelf,
                        Rule::const_kw => Type::BindingConst {
                            inner: Box::new(self_type),
                        },
                        other => panic!("unexpected self qualifier rule {:?}", other),
                    };
                }
                Vec::from([("self".to_string(), self_type)])
            }
            Rule::empty_params => Vec::new(),
            Rule::param_list => {
                let mut result: Vec<(String, Type)> = Vec::new();
                for param in pair.into_inner() {
                    let mut inner_pairs = param.into_inner();
                    let mut is_const = false;
                    if inner_pairs
                        .peek()
                        .is_some_and(|pair| pair.as_rule() == Rule::const_kw)
                    {
                        inner_pairs.next();
                        is_const = true;
                    }
                    let name = inner_pairs.next().unwrap().as_str().to_string();
                    let mut param_type = self.create_type(inner_pairs.next().unwrap());
                    if is_const {
                        param_type = Type::BindingConst {
                            inner: Box::new(param_type),
                        };
                    }
                    result.push((name, param_type));
                }
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

    fn create_generic_params(
        &self,
        pair: Pair<Rule>,
    ) -> (Vec<String>, HashMap<String, Vec<String>>) {
        assert_eq!(pair.as_rule(), Rule::generic_params);
        let mut params = Vec::new();
        let mut bounds = HashMap::new();
        for param in pair.into_inner() {
            let mut inner_pairs = param.into_inner();
            let name = inner_pairs.next().unwrap().as_str().to_string();
            let trait_bounds = inner_pairs
                .map(|p| p.as_str().to_string())
                .collect::<Vec<_>>();
            if !trait_bounds.is_empty() {
                bounds.insert(name.clone(), trait_bounds);
            }
            params.push(name);
        }
        (params, bounds)
    }

    fn create_qualified_name(&self, pair: Pair<Rule>) -> String {
        assert_eq!(pair.as_rule(), Rule::qualified_name);
        pair.into_inner()
            .map(|p| p.as_str().to_string())
            .collect::<Vec<_>>()
            .join(".")
    }

    fn create_nominal_type(&self, pair: Pair<Rule>) -> Type {
        assert_eq!(pair.as_rule(), Rule::nominal_type);
        let mut inner_pairs = pair.into_inner();
        let base = inner_pairs.next().unwrap().as_str().to_string();
        let type_arguments = inner_pairs.map(|p| self.create_type(p)).collect::<Vec<_>>();
        if type_arguments.is_empty() {
            Type::Custom(base)
        } else {
            Type::GenericInstance {
                base,
                type_arguments,
            }
        }
    }
}

pub fn parse(input: &str) -> Node {
    PestImpl::new().parse(input).unwrap()
}

pub fn type_to_string(t: &Type) -> String {
    match t {
        Type::Void => "void".to_string(),
        Type::Byte => "byte".to_string(),
        Type::Short => "short".to_string(),
        Type::Int => "int".to_string(),
        Type::Long => "long".to_string(),
        Type::Float => "float".to_string(),
        Type::Double => "double".to_string(),
        Type::String => "string".to_string(),
        Type::Boolean => "boolean".to_string(),
        Type::Char => "char".to_string(),
        Type::Const { inner } => format!("const {}", type_to_string(inner)),
        Type::BindingConst { inner } => format!("const {}", type_to_string(inner)),
        Type::Array {
            elem_type,
            dimensions,
        } => {
            let prefix = dimensions
                .iter()
                .map(|dim| format!("[{}]", type_expr_to_string(dim)))
                .collect::<String>();
            format!("{}{}", prefix, type_to_string(elem_type))
        }
        Type::Pointer { target_type } => format!("*{}", type_to_string(target_type)),
        Type::Slice { elem_type } => format!("[]{}", type_to_string(elem_type)),
        Type::Allocator => "Allocator".to_string(),
        Type::Arena => "Arena".to_string(),
        Type::GenericInstance {
            base,
            type_arguments,
        } => format!(
            "{}[{}]",
            base,
            type_arguments
                .iter()
                .map(type_to_string)
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Type::Function {
            parameters,
            return_type,
        } => format!(
            "({}) -> {}",
            parameters
                .iter()
                .map(type_to_string)
                .collect::<Vec<_>>()
                .join(", "),
            type_to_string(return_type)
        ),
        Type::MutSelf => "mut self".to_string(),
        Type::SkSelf => "self".to_string(),
        Type::Custom(v) => v.to_string(),
    }
}

fn type_expr_to_string(node: &Node) -> String {
    match node {
        Node::Literal(Literal::Integer(value)) => value.to_string(),
        Node::Literal(Literal::Long(value)) => format!("{}L", value),
        Node::Literal(Literal::Float(value)) => format!("{}f", value),
        Node::Literal(Literal::Double(value)) => value.to_string(),
        Node::Literal(Literal::StringLiteral(value)) => format!("{:?}", value),
        Node::Literal(Literal::Boolean(value)) => value.to_string(),
        Node::Literal(Literal::Char(value)) => format!("{:?}", value),
        Node::Identifier(name) => name.clone(),
        other => format!("{:?}", other),
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
        "byte" => Type::Byte,
        "short" => Type::Short,
        "int" => Type::Int,
        "long" => Type::Long,
        "float" => Type::Float,
        "double" => Type::Double,
        "string" => Type::String,
        "boolean" | "bool" => Type::Boolean,
        "char" => Type::Char,
        "Allocator" => Type::Allocator,
        "Arena" => Type::Arena,
        "void" => Type::Void,
        _ => Type::Custom(s.to_string()),
    }
}

pub fn is_integral_type(t: &Type) -> bool {
    let t = unwrap_binding_const(unwrap_const_view(t));
    matches!(t, Type::Byte | Type::Short | Type::Int | Type::Long)
}

pub fn is_floating_type(t: &Type) -> bool {
    let t = unwrap_binding_const(unwrap_const_view(t));
    matches!(t, Type::Float | Type::Double)
}

pub fn is_numeric_type(t: &Type) -> bool {
    is_integral_type(t) || is_floating_type(t)
}

pub fn is_scalar_type(t: &Type) -> bool {
    is_numeric_type(t) || matches!(t, Type::Boolean | Type::Char | Type::String)
}

pub fn numeric_rank(t: &Type) -> Option<u8> {
    let t = unwrap_binding_const(unwrap_const_view(t));
    match t {
        Type::Byte => Some(0),
        Type::Short => Some(1),
        Type::Int => Some(2),
        Type::Long => Some(3),
        Type::Float => Some(4),
        Type::Double => Some(5),
        _ => None,
    }
}

pub fn is_numeric_assignable(expected: &Type, actual: &Type) -> bool {
    match (numeric_rank(expected), numeric_rank(actual)) {
        (Some(expected_rank), Some(actual_rank)) => actual_rank <= expected_rank,
        _ => false,
    }
}

pub fn promoted_numeric_type(left: &Type, right: &Type) -> Option<Type> {
    if !is_numeric_type(left) || !is_numeric_type(right) {
        return None;
    }

    if matches!(left, Type::Double) || matches!(right, Type::Double) {
        Some(Type::Double)
    } else if matches!(left, Type::Float) || matches!(right, Type::Float) {
        Some(Type::Float)
    } else if matches!(left, Type::Long) || matches!(right, Type::Long) {
        Some(Type::Long)
    } else {
        Some(Type::Int)
    }
}

pub fn fits_integer_type(value: i64, target: &Type) -> bool {
    let target = unwrap_binding_const(unwrap_const_view(target));
    match target {
        Type::Byte => i8::try_from(value).is_ok(),
        Type::Short => i16::try_from(value).is_ok(),
        Type::Int => i32::try_from(value).is_ok(),
        Type::Long => true,
        _ => false,
    }
}

pub fn is_binding_const(t: &Type) -> bool {
    matches!(t, Type::BindingConst { .. })
}

pub fn is_const_view(t: &Type) -> bool {
    matches!(t, Type::Const { .. })
}

pub fn is_self_type(t: &Type) -> bool {
    matches!(unwrap_binding_const(t), Type::SkSelf | Type::MutSelf)
}

pub fn is_mut_self_type(t: &Type) -> bool {
    matches!(unwrap_binding_const(t), Type::MutSelf)
}

pub fn unwrap_binding_const(t: &Type) -> &Type {
    match t {
        Type::BindingConst { inner } => inner,
        other => other,
    }
}

pub fn unwrap_const_view(t: &Type) -> &Type {
    match t {
        Type::Const { inner } => inner,
        other => other,
    }
}

pub fn strip_binding_const(t: &Type) -> Type {
    match t {
        Type::BindingConst { inner } => inner.as_ref().clone(),
        other => other.clone(),
    }
}

pub fn strip_const_view(t: &Type) -> Type {
    match t {
        Type::Const { inner } => inner.as_ref().clone(),
        other => other.clone(),
    }
}

pub fn parse_string_literal(literal: &str) -> Result<String, String> {
    let inner = literal
        .strip_prefix('"')
        .and_then(|s| s.strip_suffix('"'))
        .ok_or_else(|| format!("invalid string literal `{}`", literal))?;

    let mut output = String::new();
    let mut chars = inner.chars();
    while let Some(ch) = chars.next() {
        if ch != '\\' {
            output.push(ch);
            continue;
        }

        let escaped = chars
            .next()
            .ok_or_else(|| format!("unterminated escape sequence in `{}`", literal))?;
        match escaped {
            'n' => output.push('\n'),
            'r' => output.push('\r'),
            't' => output.push('\t'),
            '0' => output.push('\0'),
            '"' => output.push('"'),
            '\\' => output.push('\\'),
            other => {
                return Err(format!(
                    "unsupported escape sequence `\\{}` in `{}`",
                    other, literal
                ))
            }
        }
    }

    Ok(output)
}

fn parse_char_literal(literal: &str) -> Result<char, String> {
    let inner = literal
        .strip_prefix('\'')
        .and_then(|s| s.strip_suffix('\''))
        .ok_or_else(|| format!("invalid char literal `{}`", literal))?;

    let ch = match inner {
        "\\n" => '\n',
        "\\r" => '\r',
        "\\t" => '\t',
        "\\0" => '\0',
        "\\'" => '\'',
        "\\\\" => '\\',
        _ => {
            let mut chars = inner.chars();
            let ch = chars
                .next()
                .ok_or_else(|| format!("invalid char literal `{}`", literal))?;
            if chars.next().is_some() {
                return Err(format!(
                    "char literal must contain exactly one character: `{}`",
                    literal
                ));
            }
            ch
        }
    };

    if (ch as u32) > u16::MAX as u32 {
        return Err(format!(
            "char literal `{}` is outside the supported 16-bit range",
            literal
        ));
    }

    Ok(ch)
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
                            type_arguments: vec![],
                            arguments: [[].to_vec()].to_vec(),
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
                        body: Vec::from([Node::Return(Some(Box::new(Node::BinaryOp {
                            left: Box::new(access_var("a")),
                            operator: Operator::Add,
                            right: Box::new(access_var("b"))
                        })))]),
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
                                type_arguments: vec![],
                                arguments: Vec::from([Vec::from([
                                    Node::Literal(Literal::Integer(1)),
                                    Node::Literal(Literal::Integer(2))
                                ])]),
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
                            body: Vec::from([Node::Return(Some(Box::new(Node::BinaryOp {
                                left: Box::new(access_var("x")),
                                operator: Operator::Subtract,
                                right: Box::new(access_var("y"))
                            })))]),
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
                                    body: vec![Node::Return(Some(Box::new(Node::Access {
                                        nodes: vec![Node::Identifier("i".to_string())],
                                    })))],
                                    else_if_blocks: vec![],
                                    else_block: None,
                                },],
                            },
                            Node::Return(Some(Box::new(Node::UnaryOp {
                                operator: UnaryOperator::Minus,
                                operand: Box::new(Node::Literal(Literal::Integer(1))),
                            }))),
                        ],
                        lambda: false,
                    },
                    Node::FunctionCall {
                        name: "test".to_string(),
                        type_arguments: vec![],
                        arguments: vec![vec![]],
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
                            _type: Type::Custom("Foo".to_string()),
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
                            _type: Type::Custom("Point".to_string()),
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
    fn test_struct_member_function_with_mut_self() {
        let source_code = r#"
            struct Point {
                function f(mut self, i:int) {
                }
            }
        "#;

        assert_eq!(
            Node::Program {
                statements: vec![
                    Node::StructDeclaration {
                        name: "Point".to_string(),
                        fields: vec![],
                        functions: vec![Node::FunctionDeclaration {
                            name: "f".to_string(),
                            parameters: vec![
                                ("self".to_string(), Type::MutSelf),
                                ("i".to_string(), Type::Int),
                            ],
                            return_type: Type::Void,
                            body: vec![],
                            lambda: false,
                        }],
                    },
                    Node::EOI,
                ],
            },
            parse(source_code)
        );
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
    fn test_prefix_int_array_fill() {
        let source_code = r#"
            arr: [1]int = [1]int::fill(1);
        "#;

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
                            name: "fill".to_string(),
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
    fn test_prefix_array_without_initializer() {
        let source_code = r#"
            arr: [3]int;
        "#;

        assert_eq!(
            Node::Program {
                statements: Vec::from([
                    Node::VariableDeclaration {
                        var_type: Type::Array {
                            elem_type: Box::new(Type::Int),
                            dimensions: Vec::from([Node::Literal(Literal::Integer(3))])
                        },
                        name: "arr".to_string(),
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
    fn test_prefix_slice_type() {
        let source_code = r#"
            slice: []int = [1, 2, 3];
        "#;

        let expected_ast = Node::Program {
            statements: vec![
                Node::VariableDeclaration {
                    name: "slice".to_string(),
                    var_type: Type::Slice {
                        elem_type: Box::new(Type::Int),
                    },
                    value: Some(Box::new(Node::ArrayInit {
                        elements: vec![
                            Node::Literal(Literal::Integer(1)),
                            Node::Literal(Literal::Integer(2)),
                            Node::Literal(Literal::Integer(3)),
                        ],
                    })),
                    metadata: Metadata::EMPTY,
                },
                Node::EOI,
            ],
        };

        assert_eq!(expected_ast, parse(source_code));
    }

    #[test]
    fn test_pointer_type() {
        let source_code = r#"
            point_ptr: *Point;
        "#;

        let expected_ast = Node::Program {
            statements: vec![
                Node::VariableDeclaration {
                    name: "point_ptr".to_string(),
                    var_type: Type::Pointer {
                        target_type: Box::new(Type::Custom("Point".to_string())),
                    },
                    value: None,
                    metadata: Metadata::EMPTY,
                },
                Node::EOI,
            ],
        };

        assert_eq!(expected_ast, parse(source_code));
    }

    #[test]
    fn test_const_variable_declaration() {
        let source_code = r#"
            const answer: int = 42;
        "#;

        let expected_ast = Node::Program {
            statements: vec![
                Node::VariableDeclaration {
                    name: "answer".to_string(),
                    var_type: Type::BindingConst {
                        inner: Box::new(Type::Int),
                    },
                    value: Some(Box::new(Node::Literal(Literal::Integer(42)))),
                    metadata: Metadata::EMPTY,
                },
                Node::EOI,
            ],
        };

        assert_eq!(expected_ast, parse(source_code));
    }

    #[test]
    fn test_const_parameter_and_const_view_types() {
        let source_code = r#"
            function copy_into(const dst: []int, src: []const int, ptr: *const Point): void {}
        "#;

        let expected_ast = Node::Program {
            statements: vec![
                Node::FunctionDeclaration {
                    name: "copy_into".to_string(),
                    parameters: vec![
                        (
                            "dst".to_string(),
                            Type::BindingConst {
                                inner: Box::new(Type::Slice {
                                    elem_type: Box::new(Type::Int),
                                }),
                            },
                        ),
                        (
                            "src".to_string(),
                            Type::Slice {
                                elem_type: Box::new(Type::Const {
                                    inner: Box::new(Type::Int),
                                }),
                            },
                        ),
                        (
                            "ptr".to_string(),
                            Type::Pointer {
                                target_type: Box::new(Type::Const {
                                    inner: Box::new(Type::Custom("Point".to_string())),
                                }),
                            },
                        ),
                    ],
                    return_type: Type::Void,
                    body: vec![],
                    lambda: false,
                },
                Node::EOI,
            ],
        };

        assert_eq!(expected_ast, parse(source_code));
    }

    #[test]
    fn test_struct_destructure_statement() {
        let source_code = r#"
            Point { x, y: py } = point;
        "#;

        let expected_ast = Node::Program {
            statements: vec![
                Node::StructDestructure {
                    struct_type: Type::Custom("Point".to_string()),
                    fields: vec![
                        StructPatternField {
                            field_name: "x".to_string(),
                            binding: "x".to_string(),
                        },
                        StructPatternField {
                            field_name: "y".to_string(),
                            binding: "py".to_string(),
                        },
                    ],
                    value: Box::new(Node::Access {
                        nodes: vec![Node::Identifier("point".to_string())],
                    }),
                    metadata: Metadata::EMPTY,
                },
                Node::EOI,
            ],
        };

        assert_eq!(expected_ast, parse(source_code));
    }

    #[test]
    fn test_struct_match_pattern() {
        let source_code = r#"
            match (point) {
                case Point { x, y: py }: {
                    print(x);
                }
            }
        "#;

        let expected_ast = Node::Program {
            statements: vec![
                Node::Match {
                    value: Box::new(Node::Access {
                        nodes: vec![Node::Identifier("point".to_string())],
                    }),
                    cases: vec![MatchCase {
                        pattern: MatchPattern::Struct {
                            struct_type: Type::Custom("Point".to_string()),
                            fields: vec![
                                StructPatternField {
                                    field_name: "x".to_string(),
                                    binding: "x".to_string(),
                                },
                                StructPatternField {
                                    field_name: "y".to_string(),
                                    binding: "py".to_string(),
                                },
                            ],
                        },
                        body: vec![Node::Print(Box::new(Node::Access {
                            nodes: vec![Node::Identifier("x".to_string())],
                        }))],
                    }],
                },
                Node::EOI,
            ],
        };

        assert_eq!(expected_ast, parse(source_code));
    }

    #[test]
    fn test_generic_struct_declaration() {
        let source_code = r#"
            struct Box[T] {
                value: T;
            }
        "#;

        let expected_ast = Node::Program {
            statements: vec![
                Node::GenericStructDeclaration {
                    name: "Box".to_string(),
                    generic_params: vec!["T".to_string()],
                    generic_bounds: HashMap::new(),
                    fields: vec![("value".to_string(), Type::Custom("T".to_string()))],
                    functions: vec![],
                },
                Node::EOI,
            ],
        };

        assert_eq!(expected_ast, parse(source_code));
    }

    #[test]
    fn test_generic_function_declaration() {
        let source_code = r#"
            function id[T](value: T): T {
                return value;
            }
        "#;

        let expected_ast = Node::Program {
            statements: vec![
                Node::GenericFunctionDeclaration {
                    name: "id".to_string(),
                    generic_params: vec!["T".to_string()],
                    generic_bounds: HashMap::new(),
                    parameters: vec![("value".to_string(), Type::Custom("T".to_string()))],
                    return_type: Type::Custom("T".to_string()),
                    body: vec![Node::Return(Some(Box::new(Node::Access {
                        nodes: vec![Node::Identifier("value".to_string())],
                    })))],
                    lambda: false,
                },
                Node::EOI,
            ],
        };

        assert_eq!(expected_ast, parse(source_code));
    }

    #[test]
    fn test_trait_and_impl_declaration() {
        let source_code = r#"
            trait Writer {
                function write(self, value: int): int;
            }

            impl Writer for TextWriter {}
        "#;

        let expected_ast = Node::Program {
            statements: vec![
                Node::TraitDeclaration {
                    name: "Writer".to_string(),
                    methods: vec![TraitMethodSignature {
                        name: "write".to_string(),
                        parameters: vec![
                            ("self".to_string(), Type::SkSelf),
                            ("value".to_string(), Type::Int),
                        ],
                        return_type: Type::Int,
                    }],
                },
                Node::ImplDeclaration {
                    generic_params: vec![],
                    generic_bounds: HashMap::new(),
                    trait_names: vec!["Writer".to_string()],
                    target_type: Type::Custom("TextWriter".to_string()),
                },
                Node::EOI,
            ],
        };

        assert_eq!(expected_ast, parse(source_code));
    }

    #[test]
    fn test_generic_impl_declaration() {
        let source_code = r#"
            impl[T] Writer for Box[T] {}
        "#;

        let expected_ast = Node::Program {
            statements: vec![
                Node::ImplDeclaration {
                    generic_params: vec!["T".to_string()],
                    generic_bounds: HashMap::new(),
                    trait_names: vec!["Writer".to_string()],
                    target_type: Type::GenericInstance {
                        base: "Box".to_string(),
                        type_arguments: vec![Type::Custom("T".to_string())],
                    },
                },
                Node::EOI,
            ],
        };

        assert_eq!(expected_ast, parse(source_code));
    }

    #[test]
    fn test_trait_method_with_mut_self() {
        let source_code = r#"
            trait Writer {
                function write(mut self, value: int): int;
            }
        "#;

        let expected_ast = Node::Program {
            statements: vec![
                Node::TraitDeclaration {
                    name: "Writer".to_string(),
                    methods: vec![TraitMethodSignature {
                        name: "write".to_string(),
                        parameters: vec![
                            ("self".to_string(), Type::MutSelf),
                            ("value".to_string(), Type::Int),
                        ],
                        return_type: Type::Int,
                    }],
                },
                Node::EOI,
            ],
        };

        assert_eq!(expected_ast, parse(source_code));
    }

    #[test]
    fn test_generic_function_bounds() {
        let source_code = r#"
            function save[T: Writer + Flushable](value: T): T {
                return value;
            }
        "#;

        let expected_ast = Node::Program {
            statements: vec![
                Node::GenericFunctionDeclaration {
                    name: "save".to_string(),
                    generic_params: vec!["T".to_string()],
                    generic_bounds: HashMap::from([(
                        "T".to_string(),
                        vec!["Writer".to_string(), "Flushable".to_string()],
                    )]),
                    parameters: vec![("value".to_string(), Type::Custom("T".to_string()))],
                    return_type: Type::Custom("T".to_string()),
                    body: vec![Node::Return(Some(Box::new(Node::Access {
                        nodes: vec![Node::Identifier("value".to_string())],
                    })))],
                    lambda: false,
                },
                Node::EOI,
            ],
        };

        assert_eq!(expected_ast, parse(source_code));
    }

    #[test]
    fn test_explicit_generic_function_call() {
        let source_code = r#"
            value: int = id[int](7);
        "#;

        let expected_ast = Node::Program {
            statements: vec![
                Node::VariableDeclaration {
                    name: "value".to_string(),
                    var_type: Type::Int,
                    value: Some(Box::new(Node::FunctionCall {
                        name: "id".to_string(),
                        type_arguments: vec![Type::Int],
                        arguments: vec![vec![Node::Literal(Literal::Integer(7))]],
                        metadata: Metadata::EMPTY,
                    })),
                    metadata: Metadata::EMPTY,
                },
                Node::EOI,
            ],
        };

        assert_eq!(expected_ast, parse(source_code));
    }

    #[test]
    fn test_enum_multi_payload_declaration_and_match() {
        let source_code = r#"
            enum PairOrNone[A, B] {
                None;
                Pair(A, B);
            }

            match (value) {
                case Pair(a, b): {
                    print(a);
                }
            }
        "#;

        let expected_ast = Node::Program {
            statements: vec![
                Node::GenericEnumDeclaration {
                    name: "PairOrNone".to_string(),
                    generic_params: vec!["A".to_string(), "B".to_string()],
                    generic_bounds: HashMap::new(),
                    variants: vec![
                        EnumVariant {
                            name: "None".to_string(),
                            payload_types: vec![],
                        },
                        EnumVariant {
                            name: "Pair".to_string(),
                            payload_types: vec![
                                Type::Custom("A".to_string()),
                                Type::Custom("B".to_string()),
                            ],
                        },
                    ],
                },
                Node::Match {
                    value: Box::new(Node::Access {
                        nodes: vec![Node::Identifier("value".to_string())],
                    }),
                    cases: vec![MatchCase {
                        pattern: MatchPattern::EnumVariant {
                            enum_type: None,
                            variant: "Pair".to_string(),
                            bindings: vec!["a".to_string(), "b".to_string()],
                        },
                        body: vec![Node::Print(Box::new(Node::Access {
                            nodes: vec![Node::Identifier("a".to_string())],
                        }))],
                    }],
                },
                Node::EOI,
            ],
        };

        assert_eq!(expected_ast, parse(source_code));
    }

    #[test]
    fn test_generic_type_and_struct_init() {
        let source_code = r#"
            box: Box[int] = Box[int] { value: 7 };
        "#;

        let expected_type = Type::GenericInstance {
            base: "Box".to_string(),
            type_arguments: vec![Type::Int],
        };
        let expected_ast = Node::Program {
            statements: vec![
                Node::VariableDeclaration {
                    var_type: expected_type.clone(),
                    name: "box".to_string(),
                    value: Some(Box::new(Node::StructInitialization {
                        _type: expected_type,
                        fields: vec![("value".to_string(), Node::Literal(Literal::Integer(7)))],
                    })),
                    metadata: Metadata::EMPTY,
                },
                Node::EOI,
            ],
        };

        assert_eq!(expected_ast, parse(source_code));
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
                                    type_arguments: vec![],
                                    arguments: [[].to_vec()].to_vec(),
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
            module std.math;
        "#;
        assert_eq!(
            Node::Program {
                statements: vec![
                    Node::Module {
                        name: "std.math".to_string()
                    },
                    Node::EOI
                ]
            },
            parse(source_code)
        );
    }

    #[test]
    fn test_import() {
        let source_code = r#"
            import std.math;
        "#;
        assert_eq!(
            Node::Program {
                statements: vec![
                    Node::Import {
                        name: "std.math".to_string()
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
                            body: vec![Node::Return(Some(Box::new(Node::BinaryOp {
                                left: Box::new(Node::Access {
                                    nodes: vec![Node::Identifier("a".to_string())],
                                }),
                                operator: Operator::Add,
                                right: Box::new(Node::Access {
                                    nodes: vec![Node::Identifier("b".to_string())],
                                }),
                            })))],
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
                            body: vec![Node::Return(Some(Box::new(Node::Literal(
                                Literal::Integer(42)
                            ))))],
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

        assert_eq!(
            Node::Program {
                statements: Vec::from([
                    Node::FunctionCall {
                        name: "f".to_string(),
                        type_arguments: vec![],
                        arguments: vec![vec![Node::FunctionDeclaration {
                            name: "anonymous".to_string(),
                            parameters: vec![],
                            return_type: Type::Int,
                            body: vec![Node::Return(Some(Box::new(Node::Literal(
                                Literal::Integer(47)
                            ))))],
                            lambda: true,
                        }]],
                        metadata: Metadata::EMPTY
                    },
                    Node::EOI
                ])
            },
            parse(source_code)
        );
    }

    #[test]
    fn test_return_function() {
        let source_code = r#"
        function f(): (int) -> int {
            return function(a:int):int {
                return a;
            };
        }
        g: (int) -> int = f();
        g(47);
        "#;

        assert_eq!(
            Node::Program {
                statements: Vec::from([
                    Node::FunctionDeclaration {
                        name: "f".to_string(),
                        parameters: vec![], // Empty parameter list
                        return_type: Type::Function {
                            parameters: vec![Type::Int],
                            return_type: Box::new(Type::Int),
                        },
                        body: vec![Node::Return(Some(Box::new(Node::FunctionDeclaration {
                            name: "anonymous".to_string(),
                            parameters: vec![("a".to_string(), Type::Int)],
                            return_type: Type::Int,
                            body: vec![Node::Return(Some(Box::new(Node::Access {
                                nodes: vec![Node::Identifier("a".to_string())],
                            })))],
                            lambda: true,
                        })))],
                        lambda: false,
                    },
                    Node::VariableDeclaration {
                        name: "g".to_string(),
                        var_type: Type::Function {
                            parameters: vec![Type::Int],
                            return_type: Box::new(Type::Int),
                        },
                        value: Some(Box::new(Node::FunctionCall {
                            name: "f".to_string(),
                            type_arguments: vec![],
                            arguments: vec![vec![]],
                            metadata: Metadata::EMPTY
                        })),
                        metadata: Metadata::EMPTY,
                    },
                    Node::FunctionCall {
                        name: "g".to_string(),
                        type_arguments: vec![],
                        arguments: vec![vec![Node::Literal(Literal::Integer(47))]],
                        metadata: Metadata::EMPTY,
                    },
                    Node::EOI
                ])
            },
            parse(source_code)
        );
    }

    #[test]
    fn test_function_chain() {
        let source_code = r#"
            f()(1)(2,3);
        "#;

        assert_eq!(
            Node::Program {
                statements: Vec::from([
                    Node::FunctionCall {
                        name: "f".to_string(),
                        type_arguments: vec![],
                        arguments: vec![
                            vec![],
                            vec![Node::Literal(Literal::Integer(1))],
                            vec![
                                Node::Literal(Literal::Integer(2)),
                                Node::Literal(Literal::Integer(3))
                            ]
                        ],
                        metadata: Metadata::EMPTY
                    },
                    Node::EOI
                ])
            },
            parse(source_code)
        );
    }

    #[test]
    fn test_new_builtin_type_literals() {
        let source_code = r#"
            b: byte = 1;
            s: short = 2;
            l: long = 3L;
            f: float = 1.5f;
            d: double = 2.5;
            c: char = 'A';
            ok: bool = true;
        "#;

        assert_eq!(
            Node::Program {
                statements: Vec::from([
                    Node::VariableDeclaration {
                        name: "b".to_string(),
                        var_type: Type::Byte,
                        value: Some(Box::new(Node::Literal(Literal::Integer(1)))),
                        metadata: Metadata::EMPTY,
                    },
                    Node::VariableDeclaration {
                        name: "s".to_string(),
                        var_type: Type::Short,
                        value: Some(Box::new(Node::Literal(Literal::Integer(2)))),
                        metadata: Metadata::EMPTY,
                    },
                    Node::VariableDeclaration {
                        name: "l".to_string(),
                        var_type: Type::Long,
                        value: Some(Box::new(Node::Literal(Literal::Long(3)))),
                        metadata: Metadata::EMPTY,
                    },
                    Node::VariableDeclaration {
                        name: "f".to_string(),
                        var_type: Type::Float,
                        value: Some(Box::new(Node::Literal(Literal::Float(1.5)))),
                        metadata: Metadata::EMPTY,
                    },
                    Node::VariableDeclaration {
                        name: "d".to_string(),
                        var_type: Type::Double,
                        value: Some(Box::new(Node::Literal(Literal::Double(2.5)))),
                        metadata: Metadata::EMPTY,
                    },
                    Node::VariableDeclaration {
                        name: "c".to_string(),
                        var_type: Type::Char,
                        value: Some(Box::new(Node::Literal(Literal::Char('A')))),
                        metadata: Metadata::EMPTY,
                    },
                    Node::VariableDeclaration {
                        name: "ok".to_string(),
                        var_type: Type::Boolean,
                        value: Some(Box::new(Node::Literal(Literal::Boolean(true)))),
                        metadata: Metadata::EMPTY,
                    },
                    Node::EOI
                ])
            },
            parse(source_code)
        );
    }
}
