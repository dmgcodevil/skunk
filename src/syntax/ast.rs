use crate::ids::NodeId;
use crate::source_map::Span;

pub type Identifier = String;

#[derive(Clone, Debug, PartialEq)]
pub struct Module {
    pub id: NodeId,
    pub span: Span,
    pub name: Option<Path>,
    pub entries: Vec<TopLevel>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Path {
    pub segments: Vec<Identifier>,
}

impl Path {
    pub fn single(name: impl Into<String>) -> Self {
        Self {
            segments: vec![name.into()],
        }
    }

    pub fn from_qualified(name: &str) -> Self {
        Self {
            segments: name.split('.').map(str::to_owned).collect(),
        }
    }

    pub fn qualified_name(&self) -> String {
        self.segments.join(".")
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Visibility {
    Private,
    Public,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TopLevel {
    pub id: NodeId,
    pub span: Span,
    pub visibility: Visibility,
    pub kind: TopLevelKind,
}

#[derive(Clone, Debug, PartialEq)]
pub enum TopLevelKind {
    Import(ImportDecl),
    TypeAlias(TypeAliasDecl),
    Struct(StructDecl),
    Enum(EnumDecl),
    Trait(TraitDecl),
    Shape(ShapeDecl),
    Attach(AttachDecl),
    Conformance(ConformanceDecl),
    /// Temporary compatibility form for a conformance whose methods were
    /// already merged by the legacy parser. Direct syntax parsing produces a
    /// `Conformance` instead; this variant disappears with the legacy bridge.
    Implementation(ImplementationDecl),
    Function(FunctionDecl),
    ExternFunction(ExternFunctionDecl),
    Global(LocalDecl),
    Test(TestDecl),
    Statement(Stmt),
}

#[derive(Clone, Debug, PartialEq)]
pub struct ImportDecl {
    pub module: Path,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TypeAliasDecl {
    pub name: Identifier,
    pub generic_parameters: Vec<GenericParameter>,
    pub target: TypeSyntax,
}

#[derive(Clone, Debug, PartialEq)]
pub struct StructDecl {
    pub name: Identifier,
    pub generic_parameters: Vec<GenericParameter>,
    pub fields: Vec<StructField>,
    /// Methods may originate from source `attach` blocks after the temporary
    /// legacy compatibility conversion. The direct parser keeps `attach`
    /// declarations separate.
    pub methods: Vec<FunctionDecl>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct StructField {
    pub id: NodeId,
    pub span: Span,
    pub name: Identifier,
    pub is_const: bool,
    pub ty: TypeSyntax,
}

#[derive(Clone, Debug, PartialEq)]
pub struct EnumDecl {
    pub name: Identifier,
    pub generic_parameters: Vec<GenericParameter>,
    pub variants: Vec<EnumVariant>,
    pub methods: Vec<FunctionDecl>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct EnumVariant {
    pub id: NodeId,
    pub span: Span,
    pub name: Identifier,
    pub payload: Vec<TypeSyntax>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TraitDecl {
    pub name: Identifier,
    pub generic_parameters: Vec<GenericParameter>,
    pub supertraits: Vec<Path>,
    pub methods: Vec<TraitMethod>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TraitMethod {
    pub id: NodeId,
    pub span: Span,
    pub name: Identifier,
    pub parameters: Vec<Parameter>,
    pub return_type: TypeSyntax,
    pub default_body: Option<Block>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ShapeDecl {
    pub name: Identifier,
    pub methods: Vec<TraitMethod>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AttachDecl {
    pub generic_parameters: Vec<GenericParameter>,
    pub target: TypeSyntax,
    pub methods: Vec<FunctionDecl>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ConformanceDecl {
    pub generic_parameters: Vec<GenericParameter>,
    pub traits: Vec<TypeSyntax>,
    pub target: TypeSyntax,
    pub methods: Vec<FunctionDecl>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ImplementationDecl {
    pub generic_parameters: Vec<GenericParameter>,
    pub traits: Vec<TypeSyntax>,
    pub target: TypeSyntax,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FunctionDecl {
    pub name: Identifier,
    pub generic_parameters: Vec<GenericParameter>,
    pub parameters: Vec<Parameter>,
    pub return_type: TypeSyntax,
    pub body: Block,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExternFunctionDecl {
    pub abi: String,
    pub name: Identifier,
    pub parameters: Vec<Parameter>,
    pub return_type: TypeSyntax,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TestDecl {
    pub name: String,
    pub body: Block,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GenericParameter {
    pub id: NodeId,
    pub span: Span,
    pub name: Identifier,
    pub capabilities: Vec<Path>,
    pub lower_bound: Option<TypeSyntax>,
    pub upper_bound: Option<TypeSyntax>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Parameter {
    pub id: NodeId,
    pub span: Span,
    pub kind: ParameterKind,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ParameterKind {
    Named {
        name: Identifier,
        is_const: bool,
        ty: TypeSyntax,
    },
    Receiver {
        mutable: bool,
        is_const: bool,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct Block {
    pub id: NodeId,
    pub span: Span,
    pub statements: Vec<Stmt>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Stmt {
    pub id: NodeId,
    pub span: Span,
    pub kind: StmtKind,
}

#[derive(Clone, Debug, PartialEq)]
pub enum StmtKind {
    Local(LocalDecl),
    StructDestructure(StructDestructure),
    Assignment { target: Expr, value: Expr },
    Expression(Expr),
    Return(Option<Expr>),
    Defer(Expr),
    Print(Expr),
    Input,
    Declaration(Box<TopLevel>),
    Block(Block),
    Unsafe(Block),
    If(IfExpr),
    Match(MatchExpr),
    For(ForStmt),
}

#[derive(Clone, Debug, PartialEq)]
pub struct LocalDecl {
    pub name: Identifier,
    pub is_const: bool,
    pub ty: TypeSyntax,
    pub initializer: Option<Expr>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct StructDestructure {
    pub ty: TypeSyntax,
    pub fields: Vec<PatternField>,
    pub value: Expr,
}

#[derive(Clone, Debug, PartialEq)]
pub struct IfExpr {
    pub condition: Expr,
    pub then_block: Block,
    pub else_if: Vec<(Expr, Block)>,
    pub else_block: Option<Block>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MatchExpr {
    pub value: Expr,
    pub cases: Vec<MatchCase>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MatchCase {
    pub id: NodeId,
    pub span: Span,
    pub pattern: Pattern,
    pub body: Block,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Pattern {
    EnumVariant {
        enum_type: Option<TypeSyntax>,
        variant: Identifier,
        bindings: Vec<Identifier>,
    },
    Struct {
        ty: TypeSyntax,
        fields: Vec<PatternField>,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct PatternField {
    pub name: Identifier,
    pub binding: Identifier,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ForStmt {
    pub initializer: Option<Box<Stmt>>,
    pub condition: Option<Expr>,
    pub update: Option<Box<Stmt>>,
    pub body: Block,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Expr {
    pub id: NodeId,
    pub span: Span,
    pub kind: ExprKind,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ExprKind {
    Literal(Literal),
    Name(Path),
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
        type_arguments: Vec<TypeSyntax>,
        /// Skunk permits repeated argument groups (`make(1)(2)`). Keeping the
        /// groups explicit avoids encoding source currying as nested AST calls.
        argument_groups: Vec<Vec<Expr>>,
    },
    Field {
        receiver: Box<Expr>,
        name: Identifier,
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
        ty: TypeSyntax,
        fields: Vec<(Identifier, Expr)>,
    },
    StaticCall {
        ty: TypeSyntax,
        name: Identifier,
        arguments: Vec<Expr>,
    },
    Array(Vec<Expr>),
    Lambda(LambdaExpr),
    Block(Block),
}

#[derive(Clone, Debug, PartialEq)]
pub struct LambdaExpr {
    pub parameters: Vec<Parameter>,
    pub return_type: TypeSyntax,
    pub body: Block,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Literal {
    Integer(i64),
    Long(i64),
    Float(f32),
    Double(f64),
    String(String),
    Boolean(bool),
    Char(char),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinaryOperator {
    Add,
    Subtract,
    Multiply,
    Divide,
    Modulo,
    Power,
    Equals,
    NotEquals,
    LessThan,
    GreaterThan,
    LessThanOrEqual,
    GreaterThanOrEqual,
    And,
    Or,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnaryOperator {
    Plus,
    Minus,
    Not,
    AddressOf,
    AddressOfMut,
    Dereference,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TypeSyntax {
    pub id: NodeId,
    pub span: Span,
    pub kind: TypeSyntaxKind,
}

#[derive(Clone, Debug, PartialEq)]
pub enum TypeSyntaxKind {
    Builtin(BuiltinType),
    Named {
        path: Path,
        arguments: Vec<TypeSyntax>,
    },
    Const(Box<TypeSyntax>),
    Array {
        element: Box<TypeSyntax>,
        dimensions: Vec<Expr>,
    },
    Reference {
        target: Box<TypeSyntax>,
        mutable: bool,
    },
    Pointer(Box<TypeSyntax>),
    Slice(Box<TypeSyntax>),
    Union(Vec<TypeSyntax>),
    Intersection(Vec<TypeSyntax>),
    Function {
        parameters: Vec<TypeSyntax>,
        result: Box<TypeSyntax>,
    },
    SelfType {
        mutable: bool,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BuiltinType {
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
    Allocator,
    Arena,
}
