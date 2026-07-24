//! Structural and typing invariants for MIR.
//!
//! LLVM lowering may rely on these checks: IDs are in range, every block has
//! one terminator, control-flow targets exist, projected places are well typed,
//! aggregates and calls match their semantic shapes, conditions are boolean,
//! and return values match their function signature.

mod value;

use super::*;
use crate::analysis::model::SemanticModel;
use crate::analysis::resolver::DefinitionKind;
use crate::analysis::types::TypeKind;
use crate::diagnostic::Diagnostic;
use crate::ids::{MirBlockId, TypeId};
use crate::source_map::Span;
use crate::syntax::ast::BuiltinType;
use std::collections::{HashMap, HashSet};

pub fn validate(module: &Module, model: &SemanticModel) -> Result<(), Vec<Diagnostic>> {
    let functions = module
        .functions
        .iter()
        .map(|function| {
            let capture_types = function
                .captures
                .iter()
                .map(|local| function.locals.get(local.index()).map(|local| local.ty))
                .collect();
            let parameter_types = function
                .parameters
                .iter()
                .map(|local| function.locals.get(local.index()).map(|local| local.ty))
                .collect();
            (
                function.source,
                FunctionShape {
                    captures: capture_types,
                    parameters: parameter_types,
                    result: function.result,
                },
            )
        })
        .collect();
    let mut validator = Validator {
        model,
        functions,
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

#[derive(Clone)]
struct FunctionShape {
    captures: Vec<Option<TypeId>>,
    parameters: Vec<Option<TypeId>>,
    result: TypeId,
}

struct Validator<'a> {
    model: &'a SemanticModel,
    functions: HashMap<NodeId, FunctionShape>,
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

        let mut captures = HashSet::new();
        for capture in &function.captures {
            let Some(local) = function.locals.get(capture.index()) else {
                self.invalid_id("MIR capture local", capture.index(), function.span);
                continue;
            };
            if local.kind != LocalKind::Capture {
                self.diagnostics.push(
                    Diagnostic::error("MIR capture list references a non-capture local")
                        .with_code("E5148")
                        .at(function.span),
                );
            }
            if !captures.insert(*capture) {
                self.diagnostics.push(
                    Diagnostic::error("MIR capture appears more than once")
                        .with_code("E5149")
                        .at(function.span),
                );
            }
            if parameters.contains(capture) {
                self.diagnostics.push(
                    Diagnostic::error("one MIR local is both a capture and a parameter")
                        .with_code("E5150")
                        .at(function.span),
                );
            }
        }
        for local in &function.locals {
            if local.kind == LocalKind::Capture && !captures.contains(&local.id) {
                self.diagnostics.push(
                    Diagnostic::error("MIR capture local is absent from the capture list")
                        .with_code("E5151")
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
                    target,
                    argument_groups,
                    result,
                } => {
                    self.ty(*result, statement.span);
                    for group in argument_groups {
                        for argument in group {
                            self.operand(function, argument, statement.span);
                        }
                    }
                    self.call_target(function, target, argument_groups, *result, statement.span);
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
                StatementKind::Print(value) => self.operand(function, value, statement.span),
                StatementKind::Input => {}
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
            TerminatorKind::SwitchEnum {
                discriminator,
                targets,
                otherwise,
            } => {
                self.operand(function, discriminator, block.terminator.span);
                let owner = self.nominal_definition(discriminator.ty);
                if owner.is_none() {
                    self.diagnostics.push(
                        Diagnostic::error("MIR enum switch discriminator is not nominal")
                            .with_code("E5138")
                            .at(block.terminator.span),
                    );
                }
                let mut variants = HashSet::new();
                for (variant, target) in targets {
                    if !variants.insert(*variant) {
                        self.diagnostics.push(
                            Diagnostic::error("MIR enum switch contains a duplicate variant")
                                .with_code("E5139")
                                .at(block.terminator.span),
                        );
                    }
                    match self.model.variant_owners.get(variant) {
                        Some(variant_owner) if Some(*variant_owner) == owner => {}
                        Some(_) => self.diagnostics.push(
                            Diagnostic::error(
                                "MIR enum switch variant belongs to a different enum",
                            )
                            .with_code("E5140")
                            .at(block.terminator.span),
                        ),
                        None => self.invalid_id("variant", variant.index(), block.terminator.span),
                    }
                    self.target(function, *target, block.terminator.span);
                }
                if let Some(owner) = owner {
                    let expected = self
                        .model
                        .variant_owners
                        .values()
                        .filter(|candidate| **candidate == owner)
                        .count();
                    if expected != variants.len() {
                        self.diagnostics.push(
                            Diagnostic::error("MIR enum switch is not exhaustive")
                                .with_code("E5141")
                                .at(block.terminator.span),
                        );
                    }
                }
                self.target(function, *otherwise, block.terminator.span);
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

    fn definition(&mut self, definition: crate::ids::DefId, span: Span) {
        if definition.index() >= self.model.resolutions.definitions.len() {
            self.invalid_id("definition", definition.index(), span);
        }
    }

    fn definition_type(&mut self, definition: crate::ids::DefId, span: Span) -> Option<TypeId> {
        self.definition(definition, span);
        let ty = self.model.definition_types.get(&definition).copied();
        if ty.is_none() && definition.index() < self.model.resolutions.definitions.len() {
            self.diagnostics.push(
                Diagnostic::error("MIR call target has no semantic type")
                    .with_code("E5137")
                    .at(span),
            );
        }
        ty
    }

    fn valid_type_kind(&self, ty: TypeId) -> Option<&TypeKind> {
        (ty.index() < self.model.types.len()).then(|| self.model.types.kind(ty))
    }

    fn reference_matches(&self, result: TypeId, target: TypeId, mutable: bool) -> bool {
        match self.valid_type_kind(result) {
            Some(TypeKind::Reference {
                target: actual,
                mutable: actual_mutable,
            }) => *actual == target && *actual_mutable == mutable,
            Some(TypeKind::Pointer(actual)) => *actual == target,
            _ => false,
        }
    }

    fn dereference_matches(&self, source: TypeId, result: TypeId) -> bool {
        match self.valid_type_kind(source) {
            Some(TypeKind::Reference { target, .. }) | Some(TypeKind::Pointer(target)) => {
                *target == result
            }
            Some(TypeKind::Const(inner)) => self.dereference_matches(*inner, result),
            _ => false,
        }
    }

    fn index_matches(&self, source: TypeId, coordinate_count: usize, result: TypeId) -> bool {
        match self.valid_type_kind(source) {
            Some(TypeKind::Array {
                element,
                dimensions,
            }) => {
                if coordinate_count >= dimensions.len() {
                    *element == result
                } else {
                    matches!(
                        self.valid_type_kind(result),
                        Some(TypeKind::Array {
                            element: result_element,
                            dimensions: result_dimensions,
                        }) if result_element == element
                            && result_dimensions == &dimensions[coordinate_count..]
                    )
                }
            }
            Some(TypeKind::Slice(element)) | Some(TypeKind::Pointer(element)) => {
                coordinate_count == 1 && *element == result
            }
            Some(TypeKind::Const(target)) => self.index_matches(*target, coordinate_count, result),
            _ => false,
        }
    }

    fn slice_matches(&self, source: TypeId, result: TypeId) -> bool {
        match self.valid_type_kind(source) {
            Some(TypeKind::Array { element, .. }) | Some(TypeKind::Slice(element)) => matches!(
                self.valid_type_kind(result),
                Some(TypeKind::Slice(result_element)) if result_element == element
            ),
            Some(TypeKind::Reference { target, .. }) | Some(TypeKind::Const(target)) => {
                self.slice_matches(*target, result)
            }
            _ => false,
        }
    }

    fn is_sequence(&self, ty: TypeId) -> bool {
        match self.valid_type_kind(ty) {
            Some(TypeKind::Array { .. }) | Some(TypeKind::Slice(_)) => true,
            Some(TypeKind::Reference { target, .. }) | Some(TypeKind::Const(target)) => {
                self.is_sequence(*target)
            }
            _ => false,
        }
    }

    fn nominal_definition(&self, ty: TypeId) -> Option<crate::ids::DefId> {
        match self.valid_type_kind(ty) {
            Some(TypeKind::Nominal { definition, .. }) => Some(*definition),
            Some(TypeKind::Const(inner)) => self.nominal_definition(*inner),
            _ => None,
        }
    }

    fn contains_generic(&self, ty: TypeId) -> bool {
        match self.valid_type_kind(ty) {
            Some(TypeKind::GenericParameter(_)) => true,
            Some(TypeKind::Nominal { arguments, .. })
            | Some(TypeKind::Union(arguments))
            | Some(TypeKind::Intersection(arguments)) => arguments
                .iter()
                .any(|argument| self.contains_generic(*argument)),
            Some(TypeKind::Const(inner))
            | Some(TypeKind::Pointer(inner))
            | Some(TypeKind::Slice(inner))
            | Some(TypeKind::Reference { target: inner, .. }) => self.contains_generic(*inner),
            Some(TypeKind::Array { element, .. }) => self.contains_generic(*element),
            Some(TypeKind::Function { parameters, result }) => {
                parameters
                    .iter()
                    .any(|parameter| self.contains_generic(*parameter))
                    || self.contains_generic(*result)
            }
            _ => false,
        }
    }

    fn is_integral(&self, ty: TypeId) -> bool {
        matches!(
            self.valid_type_kind(ty),
            Some(TypeKind::Builtin(
                BuiltinType::Byte | BuiltinType::Short | BuiltinType::Int | BuiltinType::Long
            ))
        )
    }

    fn is_int(&self, ty: TypeId) -> bool {
        matches!(
            self.valid_type_kind(ty),
            Some(TypeKind::Builtin(BuiltinType::Int))
        )
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
