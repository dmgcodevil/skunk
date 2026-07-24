//! Structural and typing invariants for MIR.
//!
//! LLVM lowering may rely on these checks: IDs are in range, every block has
//! one terminator, control-flow targets exist, operands agree with local types,
//! conditions are boolean, and return values match their function signature.

use super::*;
use crate::analysis::model::SemanticModel;
use crate::analysis::types::TypeKind;
use crate::diagnostic::Diagnostic;
use crate::ids::{MirBlockId, TypeId};
use crate::source_map::Span;
use crate::syntax::ast::BuiltinType;
use std::collections::HashSet;

pub fn validate(module: &Module, model: &SemanticModel) -> Result<(), Vec<Diagnostic>> {
    let mut validator = Validator {
        model,
        diagnostics: Vec::new(),
    };
    for function in &module.functions {
        validator.function(function);
    }
    if validator.diagnostics.is_empty() {
        Ok(())
    } else {
        Err(validator.diagnostics)
    }
}

struct Validator<'a> {
    model: &'a SemanticModel,
    diagnostics: Vec<Diagnostic>,
}

impl Validator<'_> {
    fn function(&mut self, function: &Function) {
        self.span(function.span);
        self.ty(function.result, function.span);
        if let Some(definition) = function.definition {
            if definition.index() >= self.model.resolutions.definitions.len() {
                self.invalid_id("definition", definition.index(), function.span);
            }
        }

        let mut source_locals = HashSet::new();
        for (index, local) in function.locals.iter().enumerate() {
            if local.id.index() != index {
                self.diagnostics.push(
                    Diagnostic::error("MIR local ID does not match its table position")
                        .with_code("E5101")
                        .at(function.span),
                );
            }
            self.ty(local.ty, function.span);
            if let Some(source) = local.source {
                if source.index() >= self.model.resolutions.locals.len() {
                    self.invalid_id("source local", source.index(), function.span);
                }
                if !source_locals.insert(source) {
                    self.diagnostics.push(
                        Diagnostic::error("one HIR local maps to multiple MIR locals")
                            .with_code("E5102")
                            .at(function.span),
                    );
                }
            }
        }

        let mut parameters = HashSet::new();
        for parameter in &function.parameters {
            let Some(local) = function.locals.get(parameter.index()) else {
                self.invalid_id("MIR parameter local", parameter.index(), function.span);
                continue;
            };
            if local.kind != LocalKind::Parameter {
                self.diagnostics.push(
                    Diagnostic::error("MIR parameter list references a non-parameter local")
                        .with_code("E5103")
                        .at(function.span),
                );
            }
            if !parameters.insert(*parameter) {
                self.diagnostics.push(
                    Diagnostic::error("MIR parameter appears more than once")
                        .with_code("E5104")
                        .at(function.span),
                );
            }
        }

        if function.entry.index() >= function.blocks.len() {
            self.invalid_id("entry block", function.entry.index(), function.span);
        }
        for (index, block) in function.blocks.iter().enumerate() {
            if block.id.index() != index {
                self.diagnostics.push(
                    Diagnostic::error("MIR block ID does not match its table position")
                        .with_code("E5105")
                        .at(function.span),
                );
            }
            self.block(function, block);
        }
    }

    fn block(&mut self, function: &Function, block: &BasicBlock) {
        for statement in &block.statements {
            self.span(statement.span);
            match &statement.kind {
                StatementKind::Assign { destination, value } => {
                    let destination_ty = self.place(function, destination, statement.span);
                    self.rvalue(function, value, statement.span);
                    if destination_ty.is_some_and(|ty| ty != value.ty) {
                        self.diagnostics.push(
                            Diagnostic::error("MIR assignment changes the destination type")
                                .with_code("E5106")
                                .at(statement.span),
                        );
                    }
                }
                StatementKind::Call {
                    destination,
                    callee,
                    argument_groups,
                    result,
                } => {
                    self.ty(*result, statement.span);
                    self.operand(function, callee, statement.span);
                    for group in argument_groups {
                        for argument in group {
                            self.operand(function, argument, statement.span);
                        }
                    }
                    self.call_shape(callee.ty, argument_groups, *result, statement.span);
                    match destination {
                        Some(destination) => {
                            if self
                                .place(function, destination, statement.span)
                                .is_some_and(|ty| ty != *result)
                            {
                                self.diagnostics.push(
                                    Diagnostic::error(
                                        "MIR call destination does not match its result type",
                                    )
                                    .with_code("E5118")
                                    .at(statement.span),
                                );
                            }
                        }
                        None if !self.is_void(*result) => self.diagnostics.push(
                            Diagnostic::error("value-returning MIR call has no destination")
                                .with_code("E5119")
                                .at(statement.span),
                        ),
                        None => {}
                    }
                }
            }
        }
        self.span(block.terminator.span);
        match &block.terminator.kind {
            TerminatorKind::Goto { target } => {
                self.target(function, *target, block.terminator.span)
            }
            TerminatorKind::If {
                condition,
                then_target,
                else_target,
            } => {
                self.operand(function, condition, block.terminator.span);
                if !self.is_boolean(condition.ty) {
                    self.diagnostics.push(
                        Diagnostic::error("MIR branch condition is not boolean")
                            .with_code("E5107")
                            .at(block.terminator.span),
                    );
                }
                self.target(function, *then_target, block.terminator.span);
                self.target(function, *else_target, block.terminator.span);
            }
            TerminatorKind::Return(value) => match value {
                Some(value) => {
                    self.operand(function, value, block.terminator.span);
                    if value.ty != function.result {
                        self.diagnostics.push(
                            Diagnostic::error("MIR return value does not match function result")
                                .with_code("E5108")
                                .at(block.terminator.span),
                        );
                    }
                }
                None if !self.is_void(function.result) => self.diagnostics.push(
                    Diagnostic::error("non-void MIR function returns without a value")
                        .with_code("E5109")
                        .at(block.terminator.span),
                ),
                None => {}
            },
            TerminatorKind::Unreachable => {}
        }
    }

    fn rvalue(&mut self, function: &Function, value: &Rvalue, span: Span) {
        self.ty(value.ty, span);
        match &value.kind {
            RvalueKind::Use(operand) => {
                self.operand(function, operand, span);
                if value.ty != operand.ty {
                    self.diagnostics.push(
                        Diagnostic::error("MIR use rvalue changes its operand type")
                            .with_code("E5110")
                            .at(span),
                    );
                }
            }
            RvalueKind::Unary { operand, .. } => self.operand(function, operand, span),
            RvalueKind::Binary { left, right, .. } => {
                self.operand(function, left, span);
                self.operand(function, right, span);
            }
        }
    }

    fn call_shape(
        &mut self,
        mut callee_ty: TypeId,
        groups: &[Vec<Operand>],
        result: TypeId,
        span: Span,
    ) {
        for group in groups {
            if callee_ty.index() >= self.model.types.len() {
                return;
            }
            let TypeKind::Function { parameters, result } = self.model.types.kind(callee_ty) else {
                self.diagnostics.push(
                    Diagnostic::error("MIR call applies arguments to a non-function value")
                        .with_code("E5111")
                        .at(span),
                );
                return;
            };
            if parameters.len() != group.len() {
                self.diagnostics.push(
                    Diagnostic::error("MIR call argument count does not match its function type")
                        .with_code("E5112")
                        .at(span),
                );
            }
            callee_ty = *result;
        }
        if callee_ty != result {
            self.diagnostics.push(
                Diagnostic::error("MIR call result does not match the resulting function type")
                    .with_code("E5113")
                    .at(span),
            );
        }
    }

    fn operand(&mut self, function: &Function, operand: &Operand, span: Span) {
        self.ty(operand.ty, span);
        match &operand.kind {
            OperandKind::Copy(place) => {
                if self
                    .place(function, place, span)
                    .is_some_and(|ty| ty != operand.ty)
                {
                    self.diagnostics.push(
                        Diagnostic::error("MIR operand type differs from its local type")
                            .with_code("E5114")
                            .at(span),
                    );
                }
            }
            OperandKind::Definition(definition) => {
                if definition.index() >= self.model.resolutions.definitions.len() {
                    self.invalid_id("definition", definition.index(), span);
                } else if self
                    .model
                    .definition_types
                    .get(definition)
                    .is_some_and(|ty| *ty != operand.ty)
                {
                    self.diagnostics.push(
                        Diagnostic::error("MIR definition operand has the wrong semantic type")
                            .with_code("E5120")
                            .at(span),
                    );
                }
            }
            OperandKind::Constant(Constant::Literal(_)) => {}
            OperandKind::Constant(Constant::Unit) => {
                if !self.is_void(operand.ty) {
                    self.diagnostics.push(
                        Diagnostic::error("MIR unit constant does not have void type")
                            .with_code("E5115")
                            .at(span),
                    );
                }
            }
        }
    }

    fn place(&mut self, function: &Function, place: &Place, span: Span) -> Option<TypeId> {
        match function.locals.get(place.local.index()) {
            Some(local) => Some(local.ty),
            None => {
                self.invalid_id("MIR local", place.local.index(), span);
                None
            }
        }
    }

    fn target(&mut self, function: &Function, target: MirBlockId, span: Span) {
        if target.index() >= function.blocks.len() {
            self.invalid_id("MIR block", target.index(), span);
        }
    }

    fn ty(&mut self, ty: TypeId, span: Span) {
        if ty.index() >= self.model.types.len() {
            self.invalid_id("type", ty.index(), span);
        }
    }

    fn is_boolean(&self, ty: TypeId) -> bool {
        ty.index() < self.model.types.len()
            && matches!(
                self.model.types.kind(ty),
                TypeKind::Builtin(BuiltinType::Boolean)
            )
    }

    fn is_void(&self, ty: TypeId) -> bool {
        ty.index() < self.model.types.len()
            && matches!(
                self.model.types.kind(ty),
                TypeKind::Builtin(BuiltinType::Void)
            )
    }

    fn span(&mut self, span: Span) {
        if span.start > span.end {
            self.diagnostics.push(
                Diagnostic::error("MIR contains an invalid source span")
                    .with_code("E5116")
                    .at(span),
            );
        }
    }

    fn invalid_id(&mut self, kind: &str, index: usize, span: Span) {
        self.diagnostics.push(
            Diagnostic::error(format!("MIR references unknown {kind} ID {index}"))
                .with_code("E5117")
                .at(span),
        );
    }
}
