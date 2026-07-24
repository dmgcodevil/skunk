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
    pub functions: Vec<Function>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Function {
    pub source: NodeId,
    pub span: Span,
    pub definition: Option<DefId>,
    /// Locals supplied by a closure environment, in environment field order.
    pub captures: Vec<MirLocalId>,
    pub parameters: Vec<MirLocalId>,
    pub result: TypeId,
    pub locals: Vec<Local>,
    pub entry: MirBlockId,
    pub blocks: Vec<BasicBlock>,
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
