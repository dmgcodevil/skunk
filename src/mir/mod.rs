//! Typed, control-flow-oriented mid-level intermediate representation.
//!
//! HIR preserves structured language constructs. MIR makes evaluation order,
//! temporary values, and control-flow edges explicit before LLVM lowering.
//! It is deliberately non-SSA: the LLVM backend remains responsible for SSA
//! temporaries while MIR provides a small, typed contract independent of LLVM.

pub mod lower;
pub mod validate;

use crate::ids::{DefId, FieldId, LocalId, MirBlockId, MirLocalId, NodeId, TypeId, VariantId};
use crate::source_map::Span;
use crate::syntax::ast::{BinaryOperator, Literal, UnaryOperator};

#[derive(Clone, Debug, PartialEq)]
pub struct Module {
    /// Runtime and ABI declarations needed independently of function bodies.
    pub declarations: Vec<Declaration>,
    pub functions: Vec<Function>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Declaration {
    pub source: NodeId,
    pub span: Span,
    pub exported: bool,
    pub kind: DeclarationKind,
}

#[derive(Clone, Debug, PartialEq)]
pub enum DeclarationKind {
    Struct {
        definition: DefId,
        fields: Vec<FieldDeclaration>,
        methods: Vec<DefId>,
    },
    Enum {
        definition: DefId,
        variants: Vec<VariantDeclaration>,
        methods: Vec<DefId>,
    },
    Trait {
        definition: DefId,
        supertraits: Vec<DefId>,
        methods: Vec<TraitMethodDeclaration>,
    },
    /// Shapes participate in compile-time structural constraints but have no
    /// runtime layout. Keeping their signatures here makes MIR self-describing.
    Shape {
        definition: DefId,
        methods: Vec<TraitMethodDeclaration>,
    },
    Implementation {
        traits: Vec<TypeId>,
        target: TypeId,
    },
    ExternFunction {
        definition: DefId,
        parameters: Vec<TypeId>,
        result: TypeId,
    },
    Global {
        definition: DefId,
        ty: TypeId,
        mutable: bool,
        /// Source identity of the initializer's executable MIR body.
        initializer: Option<NodeId>,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FieldDeclaration {
    pub id: FieldId,
    pub name: String,
    pub ty: TypeId,
    pub mutable: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VariantDeclaration {
    pub id: VariantId,
    pub name: String,
    pub payload: Vec<TypeId>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TraitMethodDeclaration {
    pub definition: DefId,
    pub name: String,
    pub receiver: Option<Receiver>,
    pub parameters: Vec<TypeId>,
    pub result: TypeId,
    pub default_body: Option<NodeId>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Receiver {
    pub mutable: bool,
    pub is_const: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Function {
    pub source: NodeId,
    pub span: Span,
    pub origin: FunctionOrigin,
    /// Locals supplied by a closure environment, in environment field order.
    pub captures: Vec<MirLocalId>,
    pub parameters: Vec<MirLocalId>,
    pub result: TypeId,
    pub locals: Vec<Local>,
    pub entry: MirBlockId,
    pub blocks: Vec<BasicBlock>,
}

impl Function {
    pub fn definition(&self) -> Option<DefId> {
        match self.origin {
            FunctionOrigin::Definition { definition, .. } => Some(definition),
            FunctionOrigin::Closure | FunctionOrigin::GlobalInitializer { .. } => None,
        }
    }

    pub fn owner(&self) -> Option<DefId> {
        match self.origin {
            FunctionOrigin::Definition { owner, .. } => owner,
            FunctionOrigin::Closure | FunctionOrigin::GlobalInitializer { .. } => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FunctionOrigin {
    Definition {
        definition: DefId,
        /// Nominal type that owns a method; absent for free functions.
        owner: Option<DefId>,
    },
    Closure,
    GlobalInitializer {
        global: DefId,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LocalKind {
    Capture,
    Parameter,
    User,
    Temporary,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Local {
    pub id: MirLocalId,
    pub source: Option<LocalId>,
    pub ty: TypeId,
    pub mutable: bool,
    pub kind: LocalKind,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BasicBlock {
    pub id: MirBlockId,
    pub statements: Vec<Statement>,
    pub terminator: Terminator,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Statement {
    pub source: NodeId,
    pub span: Span,
    pub kind: StatementKind,
}

#[derive(Clone, Debug, PartialEq)]
pub enum StatementKind {
    Assign {
        destination: Place,
        value: Rvalue,
    },
    Call {
        destination: Option<Place>,
        target: CallTarget,
        argument_groups: Vec<Vec<Operand>>,
        result: TypeId,
    },
    Print(Operand),
    Input,
}

/// A resolved call site. Dispatch remains explicit so code generation never
/// has to repeat method lookup or distinguish enum constructors by name.
#[derive(Clone, Debug, PartialEq)]
pub enum CallTarget {
    Operand(Operand),
    Method {
        receiver: Operand,
        method: MethodCallee,
    },
    Static(StaticCallee),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MethodCallee {
    Definition(DefId),
    Dynamic { owner: TypeId, method: DefId },
    Intrinsic { owner: TypeId, name: String },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StaticCallee {
    Definition(DefId),
    Variant(VariantId),
    Intrinsic { owner: TypeId, name: String },
}

#[derive(Clone, Debug, PartialEq)]
pub struct Terminator {
    pub source: NodeId,
    pub span: Span,
    pub kind: TerminatorKind,
}

#[derive(Clone, Debug, PartialEq)]
pub enum TerminatorKind {
    Goto {
        target: MirBlockId,
    },
    If {
        condition: Operand,
        then_target: MirBlockId,
        else_target: MirBlockId,
    },
    SwitchEnum {
        discriminator: Operand,
        targets: Vec<(VariantId, MirBlockId)>,
        otherwise: MirBlockId,
    },
    Return(Option<Operand>),
    Unreachable,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Place {
    /// The storage location from which projections start.
    pub base: PlaceBase,
    /// Ordered operations that select the final addressable value.
    pub projections: Vec<Projection>,
}

impl Place {
    pub fn local(local: MirLocalId) -> Self {
        Self {
            base: PlaceBase::Local(local),
            projections: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PlaceBase {
    Local(MirLocalId),
    Definition(DefId),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Projection {
    /// The type of the place immediately after this projection is applied.
    pub ty: TypeId,
    pub kind: ProjectionKind,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ProjectionKind {
    Dereference,
    Field(FieldId),
    Index(Vec<Operand>),
    VariantField { variant: VariantId, index: u32 },
}

#[derive(Clone, Debug, PartialEq)]
pub struct Operand {
    pub ty: TypeId,
    pub kind: OperandKind,
}

#[derive(Clone, Debug, PartialEq)]
pub enum OperandKind {
    Copy(Place),
    Definition(DefId),
    Constant(Constant),
}

#[derive(Clone, Debug, PartialEq)]
pub enum Constant {
    Literal(Literal),
    Unit,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Rvalue {
    pub ty: TypeId,
    pub kind: RvalueKind,
}

#[derive(Clone, Debug, PartialEq)]
pub enum RvalueKind {
    Use(Operand),
    /// A semantic conversion approved by the front end. Keeping it explicit
    /// prevents code generation from guessing at assignment boundaries.
    Coerce(Operand),
    Unary {
        operator: UnaryOperator,
        operand: Operand,
    },
    Binary {
        left: Operand,
        operator: BinaryOperator,
        right: Operand,
    },
    Reference {
        mutable: bool,
        place: Place,
    },
    Length(Operand),
    Slice {
        receiver: Operand,
        start: Option<Operand>,
        end: Option<Operand>,
    },
    Aggregate(Aggregate),
    Closure {
        function: NodeId,
        captures: Vec<Place>,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub enum Aggregate {
    Struct {
        definition: DefId,
        fields: Vec<(FieldId, Operand)>,
    },
    Array(Vec<Operand>),
}

#[cfg(test)]
mod tests;
