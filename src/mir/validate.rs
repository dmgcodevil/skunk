//! Structural and typing invariants for MIR.
//!
//! LLVM lowering may rely on these checks: IDs are in range, every block has
//! one terminator, control-flow targets exist, projected places are well typed,
//! aggregates and calls match their semantic shapes, conditions are boolean,
//! and return values match their function signature.

use super::*;
use crate::analysis::model::SemanticModel;
use crate::analysis::resolver::DefinitionKind;
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
            RvalueKind::Reference { mutable, place } => {
                let target = self.place(function, place, span);
                if target.is_some_and(|target| !self.reference_matches(value.ty, target, *mutable))
                {
                    self.diagnostics.push(
                        Diagnostic::error("MIR reference rvalue has an incompatible result type")
                            .with_code("E5121")
                            .at(span),
                    );
                }
            }
            RvalueKind::Length(receiver) => {
                self.operand(function, receiver, span);
                if !self.is_int(value.ty) || !self.is_sequence(receiver.ty) {
                    self.diagnostics.push(
                        Diagnostic::error(
                            "MIR length rvalue has incompatible operand or result types",
                        )
                        .with_code("E5122")
                        .at(span),
                    );
                }
            }
            RvalueKind::Slice {
                receiver,
                start,
                end,
            } => {
                self.operand(function, receiver, span);
                for bound in [start, end].into_iter().flatten() {
                    self.operand(function, bound, span);
                    if !self.is_integral(bound.ty) {
                        self.diagnostics.push(
                            Diagnostic::error("MIR slice bound is not an integer")
                                .with_code("E5123")
                                .at(span),
                        );
                    }
                }
                if !self.slice_matches(receiver.ty, value.ty) {
                    self.diagnostics.push(
                        Diagnostic::error("MIR slice result does not match its receiver")
                            .with_code("E5124")
                            .at(span),
                    );
                }
            }
            RvalueKind::Aggregate(aggregate) => self.aggregate(function, aggregate, value.ty, span),
        }
    }

    fn call_target(
        &mut self,
        function: &Function,
        target: &CallTarget,
        groups: &[Vec<Operand>],
        result: TypeId,
        span: Span,
    ) {
        match target {
            CallTarget::Operand(callee) => {
                self.operand(function, callee, span);
                self.call_shape(callee.ty, groups, result, span);
            }
            CallTarget::Method { receiver, method } => {
                self.operand(function, receiver, span);
                match method {
                    MethodCallee::Definition(definition) => {
                        if let Some(signature) = self.definition_type(*definition, span) {
                            self.method_call_shape(signature, groups, result, span);
                        }
                    }
                    MethodCallee::Dynamic { owner, method } => {
                        self.ty(*owner, span);
                        if let Some(signature) = self.definition_type(*method, span) {
                            self.method_call_shape(signature, groups, result, span);
                        }
                    }
                    MethodCallee::Intrinsic { owner, .. } => self.ty(*owner, span),
                }
            }
            CallTarget::Static(target) => match target {
                StaticCallee::Definition(definition) => {
                    if let Some(signature) = self.definition_type(*definition, span) {
                        self.call_shape(signature, groups, result, span);
                    }
                }
                StaticCallee::Variant(variant) => {
                    if let Some(payload) = self.model.variant_payloads.get(variant) {
                        if groups.len() != 1 || groups[0].len() != payload.len() {
                            self.diagnostics.push(
                                Diagnostic::error(
                                    "MIR enum constructor arguments do not match its payload",
                                )
                                .with_code("E5136")
                                .at(span),
                            );
                        }
                    } else {
                        self.invalid_id("variant", variant.index(), span);
                    }
                }
                StaticCallee::Intrinsic { owner, .. } => self.ty(*owner, span),
            },
        }
    }

    fn method_call_shape(
        &mut self,
        signature: TypeId,
        groups: &[Vec<Operand>],
        result: TypeId,
        span: Span,
    ) {
        let Some(TypeKind::Function {
            parameters,
            result: method_result,
        }) = self.valid_type_kind(signature)
        else {
            self.diagnostics.push(
                Diagnostic::error("MIR method target does not have a function type")
                    .with_code("E5111")
                    .at(span),
            );
            return;
        };
        let expected_argument_count = parameters.len().saturating_sub(1);
        let method_result = *method_result;
        let Some(first_group) = groups.first() else {
            self.diagnostics.push(
                Diagnostic::error("MIR method call has no argument group")
                    .with_code("E5112")
                    .at(span),
            );
            return;
        };
        if first_group.len() != expected_argument_count {
            self.diagnostics.push(
                Diagnostic::error("MIR method argument count does not match its function type")
                    .with_code("E5112")
                    .at(span),
            );
        }
        self.call_shape(method_result, &groups[1..], result, span);
    }

    fn aggregate(
        &mut self,
        function: &Function,
        aggregate: &Aggregate,
        result: TypeId,
        span: Span,
    ) {
        match aggregate {
            Aggregate::Struct { definition, fields } => {
                self.definition(*definition, span);
                if !matches!(
                    self.valid_type_kind(result),
                    Some(TypeKind::Nominal {
                        definition: result_definition,
                        ..
                    }) if result_definition == definition
                ) {
                    self.diagnostics.push(
                        Diagnostic::error("MIR struct aggregate has the wrong nominal result type")
                            .with_code("E5125")
                            .at(span),
                    );
                }
                let mut seen = HashSet::new();
                for (field, value) in fields {
                    if !self.model.field_types.contains_key(field) {
                        self.invalid_id("field", field.index(), span);
                    }
                    if !seen.insert(*field) {
                        self.diagnostics.push(
                            Diagnostic::error("MIR struct aggregate initializes a field twice")
                                .with_code("E5126")
                                .at(span),
                        );
                    }
                    self.operand(function, value, span);
                }
            }
            Aggregate::Array(elements) => {
                let expected_element = match self.valid_type_kind(result).cloned() {
                    Some(TypeKind::Array {
                        element,
                        dimensions,
                    }) => {
                        if dimensions
                            .first()
                            .is_some_and(|length| *length != elements.len() as u64)
                        {
                            self.diagnostics.push(
                                Diagnostic::error(
                                    "MIR array aggregate length does not match its type",
                                )
                                .with_code("E5127")
                                .at(span),
                            );
                        }
                        Some(element)
                    }
                    Some(TypeKind::Slice(element)) => Some(element),
                    _ => {
                        self.diagnostics.push(
                            Diagnostic::error(
                                "MIR array aggregate does not have an array-like type",
                            )
                            .with_code("E5128")
                            .at(span),
                        );
                        None
                    }
                };
                for element in elements {
                    self.operand(function, element, span);
                    if expected_element.is_some_and(|expected| expected != element.ty) {
                        self.diagnostics.push(
                            Diagnostic::error("MIR array element does not match its element type")
                                .with_code("E5129")
                                .at(span),
                        );
                    }
                }
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
        let mut current = match place.base {
            PlaceBase::Local(local) => match function.locals.get(local.index()) {
                Some(local) => local.ty,
                None => {
                    self.invalid_id("MIR local", local.index(), span);
                    return None;
                }
            },
            PlaceBase::Definition(definition) => {
                let Some(record) = self.model.resolutions.definitions.get(definition.index())
                else {
                    self.invalid_id("definition", definition.index(), span);
                    return None;
                };
                if record.kind != DefinitionKind::Global {
                    self.diagnostics.push(
                        Diagnostic::error("MIR place definition is not a global value")
                            .with_code("E5130")
                            .at(span),
                    );
                }
                let Some(ty) = self.model.definition_types.get(&definition).copied() else {
                    self.diagnostics.push(
                        Diagnostic::error("MIR global place has no semantic type")
                            .with_code("E5131")
                            .at(span),
                    );
                    return None;
                };
                ty
            }
        };

        for projection in &place.projections {
            self.ty(projection.ty, span);
            match &projection.kind {
                ProjectionKind::Dereference => {
                    if !self.dereference_matches(current, projection.ty) {
                        self.diagnostics.push(
                            Diagnostic::error(
                                "MIR dereference projection has an incompatible type",
                            )
                            .with_code("E5132")
                            .at(span),
                        );
                    }
                }
                ProjectionKind::Field(field) => {
                    if !self.model.field_types.contains_key(field) {
                        self.invalid_id("field", field.index(), span);
                    }
                    if !self.is_nominal(current) {
                        self.diagnostics.push(
                            Diagnostic::error("MIR field projection has a non-struct receiver")
                                .with_code("E5135")
                                .at(span),
                        );
                    }
                }
                ProjectionKind::Index(coordinates) => {
                    for coordinate in coordinates {
                        self.operand(function, coordinate, span);
                        if !self.is_integral(coordinate.ty) {
                            self.diagnostics.push(
                                Diagnostic::error("MIR index coordinate is not an integer")
                                    .with_code("E5133")
                                    .at(span),
                            );
                        }
                    }
                    if coordinates.is_empty()
                        || !self.index_matches(current, coordinates.len(), projection.ty)
                    {
                        self.diagnostics.push(
                            Diagnostic::error("MIR index projection has incompatible types")
                                .with_code("E5134")
                                .at(span),
                        );
                    }
                }
            }
            current = projection.ty;
        }
        Some(current)
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

    fn is_nominal(&self, ty: TypeId) -> bool {
        match self.valid_type_kind(ty) {
            Some(TypeKind::Nominal { .. }) => true,
            Some(TypeKind::Const(inner)) => self.is_nominal(*inner),
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
