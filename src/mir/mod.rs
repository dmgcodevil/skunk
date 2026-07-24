//! Typed, control-flow-oriented mid-level intermediate representation.
//!
//! HIR preserves structured language constructs. MIR makes evaluation order,
//! temporary values, and control-flow edges explicit before LLVM lowering.
//! It is deliberately non-SSA: the LLVM backend remains responsible for SSA
//! temporaries while MIR provides a small, typed contract independent of LLVM.

pub mod lower;
pub mod validate;

use crate::ids::{DefId, LocalId, MirBlockId, MirLocalId, NodeId, TypeId};
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
    pub parameters: Vec<MirLocalId>,
    pub result: TypeId,
    pub locals: Vec<Local>,
    pub entry: MirBlockId,
    pub blocks: Vec<BasicBlock>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LocalKind {
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
        callee: Operand,
        argument_groups: Vec<Vec<Operand>>,
        result: TypeId,
    },
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
    Return(Option<Operand>),
    Unreachable,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Place {
    pub local: MirLocalId,
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
}

#[cfg(test)]
mod tests;
