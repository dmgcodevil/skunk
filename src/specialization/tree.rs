//! Internal tree used while expanding generic declarations into concrete ones.
//!
//! This representation has no parser of its own and is not part of the public
//! compiler API. Source syntax enters through `specialization::convert`.

use std::collections::HashMap;

#[derive(Debug, PartialEq, Clone, Default)]
pub struct SubtypeBounds {
    pub lower: Option<Type>,
    pub upper: Option<Type>,
}

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
    TypeAliasDeclaration {
        name: String,
        generic_params: Vec<String>,
        generic_bounds: HashMap<String, Vec<String>>,
        subtype_bounds: HashMap<String, SubtypeBounds>,
        target_type: Type,
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
        generic_params: Vec<String>,
        generic_bounds: HashMap<String, Vec<String>>,
        subtype_bounds: HashMap<String, SubtypeBounds>,
        supertraits: Vec<String>,
        methods: Vec<TraitMethodSignature>,
    },
    ShapeDeclaration {
        name: String,
        methods: Vec<TraitMethodSignature>,
    },
    AttachDeclaration {
        generic_params: Vec<String>,
        generic_bounds: HashMap<String, Vec<String>>,
        subtype_bounds: HashMap<String, SubtypeBounds>,
        target_type: Type,
        functions: Vec<Node>,
    },
    ConformDeclaration {
        generic_params: Vec<String>,
        generic_bounds: HashMap<String, Vec<String>>,
        subtype_bounds: HashMap<String, SubtypeBounds>,
        trait_type: Type,
        target_type: Type,
        functions: Vec<Node>,
    },
    ImplDeclaration {
        generic_params: Vec<String>,
        generic_bounds: HashMap<String, Vec<String>>,
        subtype_bounds: HashMap<String, SubtypeBounds>,
        trait_types: Vec<Type>,
        target_type: Type,
    },
    EnumDeclaration {
        name: String,
        variants: Vec<EnumVariant>,
        functions: Vec<Node>,
    },
    GenericStructDeclaration {
        name: String,
        generic_params: Vec<String>,
        generic_bounds: HashMap<String, Vec<String>>,
        subtype_bounds: HashMap<String, SubtypeBounds>,
        fields: Vec<(String, Type)>,
        functions: Vec<Node>,
    },
    GenericEnumDeclaration {
        name: String,
        generic_params: Vec<String>,
        generic_bounds: HashMap<String, Vec<String>>,
        subtype_bounds: HashMap<String, SubtypeBounds>,
        variants: Vec<EnumVariant>,
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
    /// A body-less declaration of a native C function, e.g.
    /// `extern "C" function cos(value: double): double;`.
    /// The name is the exact, unmangled linker symbol.
    ExternFunctionDeclaration {
        name: String,
        parameters: Vec<(String, Type)>,
        return_type: Type,
    },
    /// A native test block: `test "name" { ... }`. Converted into a plain
    /// function plus a generated runner main by `skunk test`.
    TestDeclaration {
        name: String,
        body: Vec<Node>,
    },
    GenericFunctionDeclaration {
        name: String,
        generic_params: Vec<String>,
        generic_bounds: HashMap<String, Vec<String>>,
        subtype_bounds: HashMap<String, SubtypeBounds>,
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
    Defer(Box<Node>),          // The expression runs when its lexical scope exits
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
        name: String, // The function name
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
    End,
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
    pub default_body: Option<Vec<Node>>,
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
    String(String),
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
    AddressOfMut,
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
    Reference {
        target_type: Box<Type>,
        mutable: bool,
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
    Union(Vec<Type>),
    Intersection(Vec<Type>),
    Custom(String), // Custom types like structs
    Function {
        parameters: Vec<Type>,
        return_type: Box<Type>,
    },
    MutSelf, // special type for mut member functions
    SkSelf,  // special type for member functions
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

/// Parses source through the canonical syntax parser and converts it to the
/// transitional monomorphic tree used by specialization and LLVM lowering.
///
/// Test helper that enters through the canonical syntax parser.
#[cfg(test)]
pub fn parse(input: &str) -> Node {
    try_parse(input).unwrap()
}

/// Like [`parse`] but propagates parser errors instead of panicking.
#[cfg(test)]
pub fn try_parse(input: &str) -> Result<Node, String> {
    let mut sources = crate::source_map::SourceMap::default();
    let file = sources
        .add_file("<memory>", input)
        .map_err(|error| error.to_string())?;
    let module = crate::syntax::parser::parse_module(&sources, file).map_err(|diagnostics| {
        diagnostics
            .into_iter()
            .map(|diagnostic| diagnostic.render(&sources))
            .collect::<Vec<_>>()
            .join("\n")
    })?;
    let module = crate::syntax::normalize::normalize(module).map_err(|diagnostics| {
        diagnostics
            .into_iter()
            .map(|diagnostic| diagnostic.render(&sources))
            .collect::<Vec<_>>()
            .join("\n")
    })?;
    Ok(crate::specialization::convert::to_tree(&module))
}

/// Renders a type into the surface-language spelling used in diagnostics.
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
        Type::Reference {
            target_type,
            mutable,
        } => {
            if *mutable {
                format!("&mut {}", type_to_string(target_type))
            } else {
                format!("&{}", type_to_string(target_type))
            }
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
        Type::Union(members) => members
            .iter()
            .map(type_to_string)
            .collect::<Vec<_>>()
            .join(" | "),
        Type::Intersection(members) => members
            .iter()
            .map(|member| match member {
                Type::Union(_) => format!("({})", type_to_string(member)),
                _ => type_to_string(member),
            })
            .collect::<Vec<_>>()
            .join(" & "),
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
        Node::Literal(Literal::String(value)) => format!("{:?}", value),
        Node::Literal(Literal::Boolean(value)) => value.to_string(),
        Node::Literal(Literal::Char(value)) => format!("{:?}", value),
        Node::Identifier(name) => name.clone(),
        other => format!("{:?}", other),
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
