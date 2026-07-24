//! Lowering from structured HIR into typed, control-flow-oriented MIR.
//!
//! Lowering makes evaluation order, addressable places, aggregate construction,
//! dispatch targets, and branching explicit. Constructs not modeled by MIR yet
//! produce diagnostics instead of leaking HIR or backend assumptions forward.

mod captures;
mod control_flow;
mod declaration;
mod expression;

use super::*;
use crate::analysis::model::SemanticModel;
use crate::analysis::resolver::DefinitionKind;
use crate::analysis::types::TypeKind;
use crate::diagnostic::Diagnostic;
use crate::hir;
use crate::ids::{LocalId, MirBlockId, MirLocalId, NodeId, TypeId};
use crate::source_map::Span;
use crate::syntax::ast::BuiltinType;
use std::collections::HashMap;

pub fn lower(module: &hir::Module, model: &SemanticModel) -> Result<Module, Vec<Diagnostic>> {
    let mut functions = Vec::new();
    let mut diagnostics = Vec::new();
    let declarations = declaration::lower_module(module, model, &mut functions, &mut diagnostics);

    if diagnostics.is_empty() {
        Ok(Module {
            declarations,
            functions,
        })
    } else {
        Err(diagnostics)
    }
}

fn collect_function(
    function: &hir::Function,
    source: NodeId,
    span: Span,
    owner: Option<crate::ids::DefId>,
    model: &SemanticModel,
    functions: &mut Vec<Function>,
    diagnostics: &mut Vec<Diagnostic>,
) {
    match FunctionLowerer::new(function, source, span, owner, model, Vec::new()).lower() {
        Ok(mut lowered) => functions.append(&mut lowered),
        Err(mut errors) => diagnostics.append(&mut errors),
    }
}

#[derive(Clone, Copy)]
struct CaptureBinding {
    source: LocalId,
    ty: TypeId,
    mutable: bool,
}

struct DraftBlock {
    id: MirBlockId,
    statements: Vec<Statement>,
    terminator: Option<Terminator>,
    reachable: bool,
}

#[derive(Clone, Copy)]
enum FunctionBody<'a> {
    Block(&'a hir::Block),
    Expression(&'a hir::Expr),
}

#[derive(Clone, Copy)]
struct FunctionInput<'a> {
    origin: FunctionOrigin,
    parameters: &'a [hir::Parameter],
    result: TypeId,
    body: FunctionBody<'a>,
    source: NodeId,
    span: Span,
}

struct FunctionLowerer<'a> {
    origin: FunctionOrigin,
    parameter_inputs: &'a [hir::Parameter],
    result: TypeId,
    body: FunctionBody<'a>,
    source: NodeId,
    span: Span,
    model: &'a SemanticModel,
    locals: Vec<Local>,
    source_locals: HashMap<LocalId, MirLocalId>,
    capture_bindings: Vec<CaptureBinding>,
    captures: Vec<MirLocalId>,
    parameters: Vec<MirLocalId>,
    nested_functions: Vec<Function>,
    blocks: Vec<DraftBlock>,
    current: MirBlockId,
    deferred_scopes: Vec<Vec<&'a hir::Expr>>,
    diagnostics: Vec<Diagnostic>,
}

impl<'a> FunctionLowerer<'a> {
    fn new(
        hir: &'a hir::Function,
        source: NodeId,
        span: Span,
        owner: Option<crate::ids::DefId>,
        model: &'a SemanticModel,
        capture_bindings: Vec<CaptureBinding>,
    ) -> Self {
        let origin = match hir.definition {
            Some(definition) => FunctionOrigin::Definition { definition, owner },
            None => FunctionOrigin::Closure,
        };
        Self::with_body(
            FunctionInput {
                origin,
                parameters: &hir.parameters,
                result: hir.result,
                body: FunctionBody::Block(&hir.body),
                source,
                span,
            },
            model,
            capture_bindings,
        )
    }

    fn new_initializer(
        expression: &'a hir::Expr,
        result: TypeId,
        global: crate::ids::DefId,
        source: NodeId,
        span: Span,
        model: &'a SemanticModel,
    ) -> Self {
        Self::with_body(
            FunctionInput {
                origin: FunctionOrigin::GlobalInitializer { global },
                parameters: &[],
                result,
                body: FunctionBody::Expression(expression),
                source,
                span,
            },
            model,
            Vec::new(),
        )
    }

    fn with_body(
        input: FunctionInput<'a>,
        model: &'a SemanticModel,
        capture_bindings: Vec<CaptureBinding>,
    ) -> Self {
        Self {
            origin: input.origin,
            parameter_inputs: input.parameters,
            result: input.result,
            body: input.body,
            source: input.source,
            span: input.span,
            model,
            locals: Vec::new(),
            source_locals: HashMap::new(),
            capture_bindings,
            captures: Vec::new(),
            parameters: Vec::new(),
            nested_functions: Vec::new(),
            blocks: vec![DraftBlock {
                id: MirBlockId::new(0),
                statements: Vec::new(),
                terminator: None,
                reachable: true,
            }],
            current: MirBlockId::new(0),
            deferred_scopes: Vec::new(),
            diagnostics: Vec::new(),
        }
    }

    fn lower(mut self) -> Result<Vec<Function>, Vec<Diagnostic>> {
        for capture in self.capture_bindings.clone() {
            if let Some(local) = self.register_source_local(
                capture.source,
                capture.ty,
                capture.mutable,
                LocalKind::Capture,
                self.span,
            ) {
                self.captures.push(local);
            }
        }
        for parameter in self.parameter_inputs {
            if let Some(local) = self.register_source_local(
                parameter.local,
                parameter.ty,
                !parameter.is_const,
                LocalKind::Parameter,
                self.span,
            ) {
                self.parameters.push(local);
            }
        }

        match self.body {
            FunctionBody::Block(body) => self.lower_block(body),
            FunctionBody::Expression(expression) => {
                if let Some(value) = self.lower_expression(expression).and_then(|value| {
                    self.coerce_operand(value, self.result, expression.source, expression.span)
                }) {
                    self.terminate(Terminator {
                        source: expression.source,
                        span: expression.span,
                        kind: TerminatorKind::Return(Some(value)),
                    });
                }
            }
        }
        self.finish_open_blocks();

        if !self.diagnostics.is_empty() {
            return Err(self.diagnostics);
        }

        let blocks = self
            .blocks
            .into_iter()
            .map(|block| {
                block
                    .terminator
                    .map(|terminator| BasicBlock {
                        id: block.id,
                        statements: block.statements,
                        terminator,
                    })
                    .ok_or_else(|| {
                        Diagnostic::error("internal MIR lowering error: unterminated block")
                            .with_code("E5099")
                            .at(self.span)
                    })
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(|diagnostic| vec![diagnostic])?;

        let function = Function {
            source: self.source,
            span: self.span,
            origin: self.origin,
            captures: self.captures,
            parameters: self.parameters,
            result: self.result,
            locals: self.locals,
            entry: MirBlockId::new(0),
            blocks,
        };
        let mut functions = vec![function];
        functions.append(&mut self.nested_functions);
        Ok(functions)
    }

    fn lower_block(&mut self, block: &'a hir::Block) {
        self.deferred_scopes.push(Vec::new());
        for statement in &block.statements {
            if !self.current_is_open_and_reachable() {
                break;
            }
            self.lower_statement(statement);
        }
        if self.current_is_open_and_reachable() {
            self.emit_current_scope_defers();
        }
        self.deferred_scopes.pop();
    }

    fn lower_statement(&mut self, statement: &'a hir::Stmt) {
        match &statement.kind {
            hir::StmtKind::Local {
                local,
                ty,
                is_const,
                initializer,
            } => {
                let destination = self.register_source_local(
                    *local,
                    *ty,
                    !*is_const,
                    LocalKind::User,
                    statement.span,
                );
                if let (Some(destination), Some(initializer)) = (destination, initializer.as_ref())
                {
                    if let Some(value) = self.lower_expression(initializer) {
                        self.assign(
                            Place::local(destination),
                            self.use_or_coerce(value, *ty),
                            statement.source,
                            statement.span,
                        );
                    }
                }
            }
            hir::StmtKind::Destructure { value, bindings } => {
                self.lower_destructure(value, bindings, statement.source, statement.span)
            }
            hir::StmtKind::Assignment { target, value } => {
                let destination = self.lower_place(target);
                let value = self.lower_expression(value);
                if let (Some(destination), Some(value)) = (destination, value) {
                    self.assign(
                        destination,
                        self.use_or_coerce(value, target.ty),
                        statement.source,
                        statement.span,
                    );
                }
            }
            hir::StmtKind::Expression(expression) => {
                let _ = self.lower_expression(expression);
            }
            hir::StmtKind::Return(value) => {
                let value = value
                    .as_ref()
                    .and_then(|expression| self.lower_expression(expression))
                    .and_then(|value| {
                        self.coerce_operand(value, self.result, statement.source, statement.span)
                    });
                self.emit_all_scope_defers();
                self.terminate(Terminator {
                    source: statement.source,
                    span: statement.span,
                    kind: TerminatorKind::Return(value),
                });
            }
            hir::StmtKind::Defer(expression) => {
                if let Some(scope) = self.deferred_scopes.last_mut() {
                    scope.push(expression);
                } else {
                    self.internal_error("defer appears outside a lexical scope", statement.span);
                }
            }
            hir::StmtKind::Print(expression) => {
                if let Some(value) = self.lower_expression(expression) {
                    self.emit(
                        Statement {
                            source: statement.source,
                            span: statement.span,
                            kind: StatementKind::Print(value),
                        },
                        statement.span,
                    );
                }
            }
            hir::StmtKind::Input => self.emit(
                Statement {
                    source: statement.source,
                    span: statement.span,
                    kind: StatementKind::Input,
                },
                statement.span,
            ),
            hir::StmtKind::Block(block) | hir::StmtKind::Unsafe(block) => self.lower_block(block),
            hir::StmtKind::If(branch) => self.lower_if(branch, statement.source, statement.span),
            hir::StmtKind::Match(branch) => {
                self.lower_match(branch, statement.source, statement.span)
            }
            hir::StmtKind::For(loop_statement) => {
                self.lower_for(loop_statement, statement.source, statement.span)
            }
        }
    }

    fn lower_if(&mut self, branch: &'a hir::If, source: NodeId, span: Span) {
        let Some(condition) = self.lower_expression(&branch.condition) else {
            return;
        };
        let Some(then_target) = self.new_block(span) else {
            return;
        };
        let Some(else_target) = self.new_block(span) else {
            return;
        };
        let Some(join_target) = self.new_block(span) else {
            return;
        };

        self.terminate(Terminator {
            source,
            span,
            kind: TerminatorKind::If {
                condition,
                then_target,
                else_target,
            },
        });

        self.current = then_target;
        self.lower_block(&branch.then_block);
        self.goto_if_open(join_target, source, span);

        self.current = else_target;
        for (condition, body) in &branch.else_if {
            if !self.current_is_open_and_reachable() {
                break;
            }
            let Some(condition) = self.lower_expression(condition) else {
                return;
            };
            let Some(body_target) = self.new_block(body.span) else {
                return;
            };
            let Some(next_target) = self.new_block(body.span) else {
                return;
            };
            self.terminate(Terminator {
                source: body.source,
                span: body.span,
                kind: TerminatorKind::If {
                    condition,
                    then_target: body_target,
                    else_target: next_target,
                },
            });
            self.current = body_target;
            self.lower_block(body);
            self.goto_if_open(join_target, body.source, body.span);
            self.current = next_target;
        }

        if self.current_is_open_and_reachable() {
            if let Some(else_block) = &branch.else_block {
                self.lower_block(else_block);
            }
            self.goto_if_open(join_target, source, span);
        }

        self.current = join_target;
    }

    fn emit_current_scope_defers(&mut self) {
        let deferred = self
            .deferred_scopes
            .last()
            .into_iter()
            .flat_map(|scope| scope.iter().rev())
            .copied()
            .collect();
        self.emit_deferred_expressions(deferred);
    }

    fn emit_all_scope_defers(&mut self) {
        let deferred = self
            .deferred_scopes
            .iter()
            .rev()
            .flat_map(|scope| scope.iter().rev())
            .copied()
            .collect();
        self.emit_deferred_expressions(deferred);
    }

    fn emit_deferred_expressions(&mut self, deferred: Vec<&'a hir::Expr>) {
        for expression in deferred {
            let _ = self.lower_expression(expression);
        }
    }

    fn register_source_local(
        &mut self,
        source: LocalId,
        ty: TypeId,
        mutable: bool,
        kind: LocalKind,
        span: Span,
    ) -> Option<MirLocalId> {
        if self.source_locals.contains_key(&source) {
            self.diagnostics.push(
                Diagnostic::error(format!(
                    "HIR local {} is introduced more than once",
                    source.index()
                ))
                .with_code("E5002")
                .at(span),
            );
            return None;
        }
        let local = self.allocate_local(Some(source), ty, mutable, kind, span)?;
        self.source_locals.insert(source, local);
        Some(local)
    }

    fn allocate_local(
        &mut self,
        source: Option<LocalId>,
        ty: TypeId,
        mutable: bool,
        kind: LocalKind,
        span: Span,
    ) -> Option<MirLocalId> {
        let raw = match u32::try_from(self.locals.len()) {
            Ok(raw) => raw,
            Err(_) => {
                self.diagnostics.push(
                    Diagnostic::error("function contains too many MIR locals")
                        .with_code("E5001")
                        .at(span),
                );
                return None;
            }
        };
        let id = MirLocalId::new(raw);
        self.locals.push(Local {
            id,
            source,
            ty,
            mutable,
            kind,
        });
        Some(id)
    }

    fn new_block(&mut self, span: Span) -> Option<MirBlockId> {
        let raw = match u32::try_from(self.blocks.len()) {
            Ok(raw) => raw,
            Err(_) => {
                self.diagnostics.push(
                    Diagnostic::error("function contains too many MIR basic blocks")
                        .with_code("E5001")
                        .at(span),
                );
                return None;
            }
        };
        let id = MirBlockId::new(raw);
        self.blocks.push(DraftBlock {
            id,
            statements: Vec::new(),
            terminator: None,
            reachable: false,
        });
        Some(id)
    }

    fn assign(&mut self, destination: Place, value: Rvalue, source: NodeId, span: Span) {
        let statement = Statement {
            source,
            span,
            kind: StatementKind::Assign { destination, value },
        };
        self.emit(statement, span);
    }

    fn emit(&mut self, statement: Statement, span: Span) {
        if let Some(block) = self.blocks.get_mut(self.current.index()) {
            block.statements.push(statement);
        } else {
            self.internal_error("current MIR block does not exist", span);
        }
    }

    fn terminate(&mut self, terminator: Terminator) {
        let targets = match &terminator.kind {
            TerminatorKind::Goto { target } => vec![*target],
            TerminatorKind::If {
                then_target,
                else_target,
                ..
            } => vec![*then_target, *else_target],
            TerminatorKind::SwitchEnum {
                targets, otherwise, ..
            } => targets
                .iter()
                .map(|(_, target)| *target)
                .chain(std::iter::once(*otherwise))
                .collect(),
            TerminatorKind::Return(_) | TerminatorKind::Unreachable => Vec::new(),
        };
        let span = terminator.span;
        let Some(block) = self.blocks.get_mut(self.current.index()) else {
            self.internal_error("current MIR block does not exist", span);
            return;
        };
        if block.terminator.is_some() {
            self.internal_error("MIR block was terminated more than once", span);
            return;
        }
        block.terminator = Some(terminator);
        for target in targets {
            if let Some(block) = self.blocks.get_mut(target.index()) {
                block.reachable = true;
            } else {
                self.internal_error("MIR terminator targets an unknown block", span);
            }
        }
    }

    fn goto_if_open(&mut self, target: MirBlockId, source: NodeId, span: Span) {
        if self.current_is_open_and_reachable() {
            self.terminate(Terminator {
                source,
                span,
                kind: TerminatorKind::Goto { target },
            });
        }
    }

    fn current_is_open_and_reachable(&self) -> bool {
        self.blocks
            .get(self.current.index())
            .is_some_and(|block| block.reachable && block.terminator.is_none())
    }

    fn finish_open_blocks(&mut self) {
        let result_is_void = self.type_is_void(self.result);
        for index in 0..self.blocks.len() {
            if self.blocks[index].terminator.is_some() {
                continue;
            }
            let kind = if !self.blocks[index].reachable {
                TerminatorKind::Unreachable
            } else if result_is_void {
                TerminatorKind::Return(None)
            } else {
                self.diagnostics.push(
                    Diagnostic::error("reachable MIR path falls through a non-void function")
                        .with_code("E5006")
                        .at(self.span),
                );
                TerminatorKind::Unreachable
            };
            self.blocks[index].terminator = Some(Terminator {
                source: self.source,
                span: self.span,
                kind,
            });
        }
    }

    fn type_is_void(&self, ty: TypeId) -> bool {
        ty.index() < self.model.types.len()
            && matches!(
                self.model.types.kind(ty),
                TypeKind::Builtin(BuiltinType::Void)
            )
    }

    fn definition_operand(&self, definition: crate::ids::DefId, ty: TypeId) -> Operand {
        let kind = if self.definition_is_global(definition) {
            OperandKind::Copy(Place {
                base: PlaceBase::Definition(definition),
                projections: Vec::new(),
            })
        } else {
            OperandKind::Definition(definition)
        };
        Operand { ty, kind }
    }

    fn use_or_coerce(&self, operand: Operand, expected: TypeId) -> Rvalue {
        let kind = if operand.ty == expected {
            RvalueKind::Use(operand)
        } else {
            RvalueKind::Coerce(operand)
        };
        Rvalue { ty: expected, kind }
    }

    fn coerce_operand(
        &mut self,
        operand: Operand,
        expected: TypeId,
        source: NodeId,
        span: Span,
    ) -> Option<Operand> {
        if operand.ty == expected {
            return Some(operand);
        }
        let local = self.allocate_local(None, expected, true, LocalKind::Temporary, span)?;
        self.assign(
            Place::local(local),
            Rvalue {
                ty: expected,
                kind: RvalueKind::Coerce(operand),
            },
            source,
            span,
        );
        Some(Operand {
            ty: expected,
            kind: OperandKind::Copy(Place::local(local)),
        })
    }

    fn definition_is_global(&self, definition: crate::ids::DefId) -> bool {
        self.model
            .resolutions
            .definitions
            .get(definition.index())
            .is_some_and(|record| record.kind == DefinitionKind::Global)
    }

    fn internal_error(&mut self, message: &str, span: Span) {
        self.diagnostics.push(
            Diagnostic::error(format!("internal MIR lowering error: {message}"))
                .with_code("E5099")
                .at(span),
        );
    }
}

fn lower_method_target(target: &hir::MethodTarget) -> MethodCallee {
    match target {
        hir::MethodTarget::Definition(definition) => MethodCallee::Definition(*definition),
        hir::MethodTarget::Dynamic { owner, method } => MethodCallee::Dynamic {
            owner: *owner,
            method: *method,
        },
        hir::MethodTarget::Intrinsic { owner, name } => MethodCallee::Intrinsic {
            owner: *owner,
            name: name.clone(),
        },
    }
}

fn lower_static_target(target: &hir::StaticTarget) -> StaticCallee {
    match target {
        hir::StaticTarget::Definition(definition) => StaticCallee::Definition(*definition),
        hir::StaticTarget::Variant(variant) => StaticCallee::Variant(*variant),
        hir::StaticTarget::Intrinsic { owner, name } => StaticCallee::Intrinsic {
            owner: *owner,
            name: name.clone(),
        },
    }
}
