//! Lowering from structured HIR into typed, control-flow-oriented MIR.
//!
//! This first lowering slice covers ordinary functions, parameters, locals,
//! assignments, literals, unary/binary operations, calls, nested blocks,
//! returns, and `if`/`else-if`/`else` control flow. Unsupported HIR constructs
//! produce diagnostics instead of being silently copied into MIR.

use super::*;
use crate::analysis::model::SemanticModel;
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

    for item in &module.items {
        match &item.kind {
            hir::ItemKind::Function(function) => collect_function(
                function,
                item.source,
                item.span,
                model,
                &mut functions,
                &mut diagnostics,
            ),
            hir::ItemKind::Struct(declaration) => {
                for function in &declaration.methods {
                    collect_function(
                        function,
                        function.body.source,
                        function.body.span,
                        model,
                        &mut functions,
                        &mut diagnostics,
                    );
                }
            }
            hir::ItemKind::Enum(declaration) => {
                for function in &declaration.methods {
                    collect_function(
                        function,
                        function.body.source,
                        function.body.span,
                        model,
                        &mut functions,
                        &mut diagnostics,
                    );
                }
            }
            hir::ItemKind::Trait(_)
            | hir::ItemKind::Shape(_)
            | hir::ItemKind::Implementation { .. }
            | hir::ItemKind::ExternFunction(_)
            | hir::ItemKind::Global(_)
            | hir::ItemKind::Test(_)
            | hir::ItemKind::Statement(_) => {}
        }
    }

    if diagnostics.is_empty() {
        Ok(Module { functions })
    } else {
        Err(diagnostics)
    }
}

fn collect_function(
    function: &hir::Function,
    source: NodeId,
    span: Span,
    model: &SemanticModel,
    functions: &mut Vec<Function>,
    diagnostics: &mut Vec<Diagnostic>,
) {
    match FunctionLowerer::new(function, source, span, model).lower() {
        Ok(function) => functions.push(function),
        Err(mut errors) => diagnostics.append(&mut errors),
    }
}

struct DraftBlock {
    id: MirBlockId,
    statements: Vec<Statement>,
    terminator: Option<Terminator>,
    reachable: bool,
}

struct FunctionLowerer<'a> {
    hir: &'a hir::Function,
    source: NodeId,
    span: Span,
    model: &'a SemanticModel,
    locals: Vec<Local>,
    source_locals: HashMap<LocalId, MirLocalId>,
    parameters: Vec<MirLocalId>,
    blocks: Vec<DraftBlock>,
    current: MirBlockId,
    diagnostics: Vec<Diagnostic>,
}

impl<'a> FunctionLowerer<'a> {
    fn new(hir: &'a hir::Function, source: NodeId, span: Span, model: &'a SemanticModel) -> Self {
        Self {
            hir,
            source,
            span,
            model,
            locals: Vec::new(),
            source_locals: HashMap::new(),
            parameters: Vec::new(),
            blocks: vec![DraftBlock {
                id: MirBlockId::new(0),
                statements: Vec::new(),
                terminator: None,
                reachable: true,
            }],
            current: MirBlockId::new(0),
            diagnostics: Vec::new(),
        }
    }

    fn lower(mut self) -> Result<Function, Vec<Diagnostic>> {
        for parameter in &self.hir.parameters {
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

        self.lower_block(&self.hir.body);
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

        Ok(Function {
            source: self.source,
            span: self.span,
            definition: self.hir.definition,
            parameters: self.parameters,
            result: self.hir.result,
            locals: self.locals,
            entry: MirBlockId::new(0),
            blocks,
        })
    }

    fn lower_block(&mut self, block: &hir::Block) {
        for statement in &block.statements {
            if !self.current_is_open_and_reachable() {
                break;
            }
            self.lower_statement(statement);
        }
    }

    fn lower_statement(&mut self, statement: &hir::Stmt) {
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
                            Place { local: destination },
                            Rvalue {
                                ty: value.ty,
                                kind: RvalueKind::Use(value),
                            },
                            statement.source,
                            statement.span,
                        );
                    }
                }
            }
            hir::StmtKind::Assignment { target, value } => {
                let destination = self.lower_place(target);
                let value = self.lower_expression(value);
                if let (Some(destination), Some(value)) = (destination, value) {
                    self.assign(
                        destination,
                        Rvalue {
                            ty: value.ty,
                            kind: RvalueKind::Use(value),
                        },
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
                    .and_then(|expression| self.lower_expression(expression));
                self.terminate(Terminator {
                    source: statement.source,
                    span: statement.span,
                    kind: TerminatorKind::Return(value),
                });
            }
            hir::StmtKind::Block(block) | hir::StmtKind::Unsafe(block) => self.lower_block(block),
            hir::StmtKind::If(branch) => self.lower_if(branch, statement.source, statement.span),
            hir::StmtKind::Destructure { .. }
            | hir::StmtKind::Defer(_)
            | hir::StmtKind::Print(_)
            | hir::StmtKind::Input
            | hir::StmtKind::Match(_)
            | hir::StmtKind::For(_) => self.unsupported_statement(statement),
        }
    }

    fn lower_if(&mut self, branch: &hir::If, source: NodeId, span: Span) {
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

    fn lower_expression(&mut self, expression: &hir::Expr) -> Option<Operand> {
        match &expression.kind {
            hir::ExprKind::Literal(literal) => Some(Operand {
                ty: expression.ty,
                kind: OperandKind::Constant(Constant::Literal(literal.clone())),
            }),
            hir::ExprKind::Value(hir::Value::Definition(definition)) => Some(Operand {
                ty: expression.ty,
                kind: OperandKind::Definition(*definition),
            }),
            hir::ExprKind::Value(hir::Value::Local(local)) => {
                let mir_local = self.source_locals.get(local).copied();
                match mir_local {
                    Some(local) => Some(Operand {
                        ty: expression.ty,
                        kind: OperandKind::Copy(Place { local }),
                    }),
                    None => {
                        self.diagnostics.push(
                            Diagnostic::error(format!(
                                "HIR local {} is used before it is introduced in MIR",
                                local.index()
                            ))
                            .with_code("E5003")
                            .at(expression.span),
                        );
                        None
                    }
                }
            }
            hir::ExprKind::Unary { operator, operand } => {
                let operand = self.lower_expression(operand)?;
                self.temporary(
                    expression,
                    RvalueKind::Unary {
                        operator: *operator,
                        operand,
                    },
                )
            }
            hir::ExprKind::Binary {
                left,
                operator,
                right,
            } => {
                let left = self.lower_expression(left)?;
                let right = self.lower_expression(right)?;
                self.temporary(
                    expression,
                    RvalueKind::Binary {
                        left,
                        operator: *operator,
                        right,
                    },
                )
            }
            hir::ExprKind::Call {
                callee,
                argument_groups,
            } => {
                let callee = self.lower_expression(callee)?;
                let argument_groups = argument_groups
                    .iter()
                    .map(|group| {
                        group
                            .iter()
                            .map(|argument| self.lower_expression(argument))
                            .collect::<Option<Vec<_>>>()
                    })
                    .collect::<Option<Vec<_>>>()?;
                self.call(expression, callee, argument_groups)
            }
            hir::ExprKind::Block(block) => {
                self.lower_block(block);
                Some(Operand {
                    ty: expression.ty,
                    kind: OperandKind::Constant(Constant::Unit),
                })
            }
            hir::ExprKind::MethodCall { .. }
            | hir::ExprKind::Field { .. }
            | hir::ExprKind::Length { .. }
            | hir::ExprKind::Index { .. }
            | hir::ExprKind::Slice { .. }
            | hir::ExprKind::StructInit { .. }
            | hir::ExprKind::StaticCall { .. }
            | hir::ExprKind::Array(_)
            | hir::ExprKind::Lambda(_) => {
                self.diagnostics.push(
                    Diagnostic::error(
                        "HIR expression is not supported by the initial MIR lowering",
                    )
                    .with_code("E5004")
                    .at(expression.span),
                );
                None
            }
        }
    }

    fn lower_place(&mut self, expression: &hir::Expr) -> Option<Place> {
        let hir::ExprKind::Value(hir::Value::Local(local)) = &expression.kind else {
            self.diagnostics.push(
                Diagnostic::error("initial MIR lowering supports assignment to locals only")
                    .with_code("E5005")
                    .at(expression.span),
            );
            return None;
        };
        self.source_locals
            .get(local)
            .copied()
            .map(|local| Place { local })
            .or_else(|| {
                self.diagnostics.push(
                    Diagnostic::error(format!(
                        "assignment target local {} is not declared in MIR",
                        local.index()
                    ))
                    .with_code("E5003")
                    .at(expression.span),
                );
                None
            })
    }

    fn temporary(&mut self, expression: &hir::Expr, kind: RvalueKind) -> Option<Operand> {
        let local = self.allocate_local(
            None,
            expression.ty,
            true,
            LocalKind::Temporary,
            expression.span,
        )?;
        self.assign(
            Place { local },
            Rvalue {
                ty: expression.ty,
                kind,
            },
            expression.source,
            expression.span,
        );
        Some(Operand {
            ty: expression.ty,
            kind: OperandKind::Copy(Place { local }),
        })
    }

    fn call(
        &mut self,
        expression: &hir::Expr,
        callee: Operand,
        argument_groups: Vec<Vec<Operand>>,
    ) -> Option<Operand> {
        let destination = if self.type_is_void(expression.ty) {
            None
        } else {
            Some(Place {
                local: self.allocate_local(
                    None,
                    expression.ty,
                    true,
                    LocalKind::Temporary,
                    expression.span,
                )?,
            })
        };
        let result = destination.as_ref().map(|place| Operand {
            ty: expression.ty,
            kind: OperandKind::Copy(place.clone()),
        });
        self.emit(
            Statement {
                source: expression.source,
                span: expression.span,
                kind: StatementKind::Call {
                    destination,
                    callee,
                    argument_groups,
                    result: expression.ty,
                },
            },
            expression.span,
        );
        if let Some(result) = result {
            Some(result)
        } else {
            Some(Operand {
                ty: expression.ty,
                kind: OperandKind::Constant(Constant::Unit),
            })
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
        let result_is_void = self.type_is_void(self.hir.result);
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

    fn unsupported_statement(&mut self, statement: &hir::Stmt) {
        self.diagnostics.push(
            Diagnostic::error("HIR statement is not supported by the initial MIR lowering")
                .with_code("E5007")
                .at(statement.span),
        );
    }

    fn internal_error(&mut self, message: &str, span: Span) {
        self.diagnostics.push(
            Diagnostic::error(format!("internal MIR lowering error: {message}"))
                .with_code("E5099")
                .at(span),
        );
    }
}
