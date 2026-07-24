use super::*;
use std::collections::BTreeSet;

impl<'a> FunctionLowerer<'a> {
    pub(super) fn lower_closure(
        &mut self,
        expression: &'a hir::Expr,
        function: &'a hir::Function,
    ) -> Option<Operand> {
        let mut bindings = Vec::new();
        let mut captures = Vec::new();
        for source in referenced_outer_locals(function, &self.source_locals) {
            let Some(mir_local) = self.source_locals.get(&source).copied() else {
                continue;
            };
            let Some(local) = self.locals.get(mir_local.index()) else {
                self.internal_error("closure capture local does not exist", expression.span);
                return None;
            };
            bindings.push(CaptureBinding {
                source,
                ty: local.ty,
                mutable: local.mutable,
            });
            captures.push(Place::local(mir_local));
        }

        match FunctionLowerer::new(
            function,
            expression.source,
            expression.span,
            None,
            self.model,
            bindings,
        )
        .lower()
        {
            Ok(mut functions) => self.nested_functions.append(&mut functions),
            Err(mut diagnostics) => {
                self.diagnostics.append(&mut diagnostics);
                return None;
            }
        }

        self.temporary(
            expression,
            RvalueKind::Closure {
                function: expression.source,
                captures,
            },
        )
    }
}

fn referenced_outer_locals(
    function: &hir::Function,
    outer: &HashMap<LocalId, MirLocalId>,
) -> Vec<LocalId> {
    let mut captures = BTreeSet::new();
    visit_block(&function.body, outer, &mut captures);
    captures.into_iter().collect()
}

fn visit_block(
    block: &hir::Block,
    outer: &HashMap<LocalId, MirLocalId>,
    captures: &mut BTreeSet<LocalId>,
) {
    for statement in &block.statements {
        visit_statement(statement, outer, captures);
    }
}

fn visit_statement(
    statement: &hir::Stmt,
    outer: &HashMap<LocalId, MirLocalId>,
    captures: &mut BTreeSet<LocalId>,
) {
    match &statement.kind {
        hir::StmtKind::Local { initializer, .. } => {
            if let Some(initializer) = initializer {
                visit_expression(initializer, outer, captures);
            }
        }
        hir::StmtKind::Destructure { value, .. } => visit_expression(value, outer, captures),
        hir::StmtKind::Assignment { target, value } => {
            visit_expression(target, outer, captures);
            visit_expression(value, outer, captures);
        }
        hir::StmtKind::Expression(expression)
        | hir::StmtKind::Defer(expression)
        | hir::StmtKind::Print(expression) => visit_expression(expression, outer, captures),
        hir::StmtKind::Return(value) => {
            if let Some(value) = value {
                visit_expression(value, outer, captures);
            }
        }
        hir::StmtKind::Block(block) | hir::StmtKind::Unsafe(block) => {
            visit_block(block, outer, captures)
        }
        hir::StmtKind::If(branch) => {
            visit_expression(&branch.condition, outer, captures);
            visit_block(&branch.then_block, outer, captures);
            for (condition, body) in &branch.else_if {
                visit_expression(condition, outer, captures);
                visit_block(body, outer, captures);
            }
            if let Some(body) = &branch.else_block {
                visit_block(body, outer, captures);
            }
        }
        hir::StmtKind::Match(branch) => {
            visit_expression(&branch.value, outer, captures);
            for case in &branch.cases {
                visit_block(&case.body, outer, captures);
            }
        }
        hir::StmtKind::For(loop_statement) => {
            if let Some(initializer) = &loop_statement.initializer {
                visit_statement(initializer, outer, captures);
            }
            if let Some(condition) = &loop_statement.condition {
                visit_expression(condition, outer, captures);
            }
            if let Some(update) = &loop_statement.update {
                visit_statement(update, outer, captures);
            }
            visit_block(&loop_statement.body, outer, captures);
        }
        hir::StmtKind::Input => {}
    }
}

fn visit_expression(
    expression: &hir::Expr,
    outer: &HashMap<LocalId, MirLocalId>,
    captures: &mut BTreeSet<LocalId>,
) {
    match &expression.kind {
        hir::ExprKind::Value(hir::Value::Local(local)) => {
            if outer.contains_key(local) {
                captures.insert(*local);
            }
        }
        hir::ExprKind::Value(hir::Value::Definition(_)) | hir::ExprKind::Literal(_) => {}
        hir::ExprKind::Unary { operand, .. } => visit_expression(operand, outer, captures),
        hir::ExprKind::Binary { left, right, .. } => {
            visit_expression(left, outer, captures);
            visit_expression(right, outer, captures);
        }
        hir::ExprKind::Call {
            callee,
            argument_groups,
        } => {
            visit_expression(callee, outer, captures);
            visit_argument_groups(argument_groups, outer, captures);
        }
        hir::ExprKind::MethodCall {
            receiver,
            argument_groups,
            ..
        } => {
            visit_expression(receiver, outer, captures);
            visit_argument_groups(argument_groups, outer, captures);
        }
        hir::ExprKind::Field { receiver, .. }
        | hir::ExprKind::Length { receiver }
        | hir::ExprKind::Index { receiver, .. }
        | hir::ExprKind::Slice { receiver, .. } => {
            visit_expression(receiver, outer, captures);
            match &expression.kind {
                hir::ExprKind::Index { coordinates, .. } => {
                    for coordinate in coordinates {
                        visit_expression(coordinate, outer, captures);
                    }
                }
                hir::ExprKind::Slice { start, end, .. } => {
                    for bound in [start, end].into_iter().flatten() {
                        visit_expression(bound, outer, captures);
                    }
                }
                _ => {}
            }
        }
        hir::ExprKind::StructInit { fields, .. } => {
            for (_, value) in fields {
                visit_expression(value, outer, captures);
            }
        }
        hir::ExprKind::StaticCall { arguments, .. } | hir::ExprKind::Array(arguments) => {
            for argument in arguments {
                visit_expression(argument, outer, captures);
            }
        }
        hir::ExprKind::Lambda(function) => visit_block(&function.body, outer, captures),
        hir::ExprKind::Block(block) => visit_block(block, outer, captures),
    }
}

fn visit_argument_groups(
    groups: &[Vec<hir::Expr>],
    outer: &HashMap<LocalId, MirLocalId>,
    captures: &mut BTreeSet<LocalId>,
) {
    for group in groups {
        for argument in group {
            visit_expression(argument, outer, captures);
        }
    }
}
