//! Typed high-level intermediate representation.
//!
//! HIR contains resolved identities and semantic `TypeId`s only. It never
//! embeds syntax AST nodes or unresolved source names.

pub mod lower;
pub mod validate;

use crate::ids::{DefId, FieldId, LocalId, NodeId, TypeId, VariantId};
use crate::source_map::Span;
use crate::syntax::ast::{BinaryOperator, Literal, UnaryOperator, Visibility};

#[derive(Debug)]
pub struct Module {
    pub items: Vec<Item>,
}

#[derive(Debug)]
pub struct Item {
    pub source: NodeId,
    pub span: Span,
    pub definition: Option<DefId>,
    pub visibility: Visibility,
    pub kind: ItemKind,
}

#[derive(Debug)]
pub enum ItemKind {
    Struct(Struct),
    Enum(Enum),
    Trait(Trait),
    Shape(Trait),
    Implementation { traits: Vec<TypeId>, target: TypeId },
    Function(Function),
    ExternFunction(FunctionSignature),
    Global(Global),
    Test(Block),
    Statement(Stmt),
}

#[derive(Debug)]
pub struct Struct {
    pub fields: Vec<Field>,
    pub methods: Vec<Function>,
}

#[derive(Debug)]
pub struct Field {
    pub id: FieldId,
    pub name: String,
    pub ty: TypeId,
    pub is_const: bool,
}

#[derive(Debug)]
pub struct Enum {
    pub variants: Vec<Variant>,
    pub methods: Vec<Function>,
}

#[derive(Debug)]
pub struct Variant {
    pub id: VariantId,
    pub name: String,
    pub payload: Vec<TypeId>,
}

#[derive(Debug)]
pub struct Trait {
    pub supertraits: Vec<DefId>,
    pub methods: Vec<TraitMethod>,
}

#[derive(Debug)]
pub struct TraitMethod {
    pub definition: DefId,
    pub name: String,
    pub receiver: Option<Receiver>,
    pub parameters: Vec<TypeId>,
    pub result: TypeId,
    pub default_body: Option<Block>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Receiver {
    pub mutable: bool,
    pub is_const: bool,
}

#[derive(Debug)]
pub struct FunctionSignature {
    pub parameters: Vec<TypeId>,
    pub result: TypeId,
}

#[derive(Debug)]
pub struct Function {
    pub definition: Option<DefId>,
    pub parameters: Vec<Parameter>,
    pub result: TypeId,
    pub body: Block,
}

#[derive(Debug)]
pub struct Parameter {
    pub local: LocalId,
    pub ty: TypeId,
    pub is_const: bool,
}

#[derive(Debug)]
pub struct Global {
    pub ty: TypeId,
    pub is_const: bool,
    pub initializer: Option<Expr>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Flow {
    FallsThrough,
    Returns,
    Diverges,
}

impl Flow {
    pub fn continues(self) -> bool {
        self == Self::FallsThrough
    }
}

#[derive(Debug)]
pub struct Block {
    pub source: NodeId,
    pub span: Span,
    pub statements: Vec<Stmt>,
    pub flow: Flow,
}

#[derive(Debug)]
pub struct Stmt {
    pub source: NodeId,
    pub span: Span,
    pub kind: StmtKind,
}

#[derive(Debug)]
pub enum StmtKind {
    Local {
        local: LocalId,
        ty: TypeId,
        is_const: bool,
        initializer: Option<Expr>,
    },
    Destructure {
        value: Expr,
        bindings: Vec<(FieldId, LocalId)>,
    },
    Assignment {
        target: Expr,
        value: Expr,
    },
    Expression(Expr),
    Return(Option<Expr>),
    Defer(Expr),
    Print(Expr),
    Input,
    Block(Block),
    Unsafe(Block),
    If(If),
    Match(Match),
    For(For),
}

#[derive(Debug)]
pub struct If {
    pub condition: Expr,
    pub then_block: Block,
    pub else_if: Vec<(Expr, Block)>,
    pub else_block: Option<Block>,
    pub flow: Flow,
}

#[derive(Debug)]
pub struct Match {
    pub value: Expr,
    pub cases: Vec<MatchCase>,
    pub flow: Flow,
}

#[derive(Debug)]
pub struct MatchCase {
    pub pattern: MatchPattern,
    pub bindings: Vec<LocalId>,
    pub body: Block,
}

#[derive(Debug)]
pub enum MatchPattern {
    EnumVariant {
        variant: VariantId,
    },
    Struct {
        definition: DefId,
        fields: Vec<FieldId>,
    },
}

#[derive(Debug)]
pub struct For {
    pub initializer: Option<Box<Stmt>>,
    pub condition: Option<Expr>,
    pub update: Option<Box<Stmt>>,
    pub body: Block,
    pub flow: Flow,
}

#[derive(Debug)]
pub struct Expr {
    pub source: NodeId,
    pub span: Span,
    pub ty: TypeId,
    pub kind: ExprKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Value {
    Definition(DefId),
    Local(LocalId),
}

#[derive(Debug)]
pub enum ExprKind {
    Literal(Literal),
    Value(Value),
    Unary {
        operator: UnaryOperator,
        operand: Box<Expr>,
    },
    Binary {
        left: Box<Expr>,
        operator: BinaryOperator,
        right: Box<Expr>,
    },
    Call {
        callee: Box<Expr>,
        argument_groups: Vec<Vec<Expr>>,
    },
    MethodCall {
        receiver: Box<Expr>,
        method: MethodTarget,
        argument_groups: Vec<Vec<Expr>>,
    },
    Field {
        receiver: Box<Expr>,
        field: FieldId,
    },
    Length {
        receiver: Box<Expr>,
    },
    Index {
        receiver: Box<Expr>,
        coordinates: Vec<Expr>,
    },
    Slice {
        receiver: Box<Expr>,
        start: Option<Box<Expr>>,
        end: Option<Box<Expr>>,
    },
    StructInit {
        definition: DefId,
        fields: Vec<(FieldId, Expr)>,
    },
    StaticCall {
        target: StaticTarget,
        arguments: Vec<Expr>,
    },
    Array(Vec<Expr>),
    Lambda(Function),
    Block(Block),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StaticTarget {
    Definition(DefId),
    Variant(VariantId),
    Intrinsic { owner: TypeId, name: String },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MethodTarget {
    Definition(DefId),
    Dynamic { owner: TypeId, method: DefId },
    Intrinsic { owner: TypeId, name: String },
}
