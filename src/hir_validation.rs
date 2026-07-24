//! Structural invariants for the typed HIR boundary.
//!
//! HIR is the last source-oriented representation. Code generation may rely on
//! these invariants instead of repeating bounds checks or accepting sentinel
//! IDs left behind by an incomplete lowering rule.

use crate::diagnostic::Diagnostic;
use crate::hir;
use crate::ids::{DefId, FieldId, LocalId, TypeId, VariantId};
use crate::semantic_types::TypeKind;
use crate::semantics::SemanticModel;
use crate::source_map::Span;
use std::collections::HashSet;

pub fn validate(module: &hir::Module, model: &SemanticModel) -> Result<(), Vec<Diagnostic>> {
    let mut validator = Validator {
        model,
        diagnostics: Vec::new(),
        checked_types: HashSet::new(),
    };
    for item in &module.items {
        validator.item(item);
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
    checked_types: HashSet<TypeId>,
}

impl Validator<'_> {
    fn item(&mut self, item: &hir::Item) {
        self.span(item.span);
        let _ = item.source;
        let _ = item.visibility;
        if let Some(definition) = item.definition {
            self.definition(definition, item.span);
        }
        match &item.kind {
            hir::ItemKind::Struct(declaration) => {
                for field in &declaration.fields {
                    self.field(field.id, item.span);
                    self.ty(field.ty, item.span);
                    let _ = (&field.name, field.is_const);
                }
                for method in &declaration.methods {
                    self.function(method, item.span);
                }
            }
            hir::ItemKind::Enum(declaration) => {
                for variant in &declaration.variants {
                    self.variant(variant.id, item.span);
                    let _ = &variant.name;
                    for ty in &variant.payload {
                        self.ty(*ty, item.span);
                    }
                }
                for method in &declaration.methods {
                    self.function(method, item.span);
                }
            }
            hir::ItemKind::Trait(declaration) | hir::ItemKind::Shape(declaration) => {
                for supertrait in &declaration.supertraits {
                    self.definition(*supertrait, item.span);
                }
                for method in &declaration.methods {
                    self.definition(method.definition, item.span);
                    let _ = (&method.name, method.receiver);
                    for parameter in &method.parameters {
                        self.ty(*parameter, item.span);
                    }
                    self.ty(method.result, item.span);
                    if let Some(body) = &method.default_body {
                        self.block(body);
                    }
                }
            }
            hir::ItemKind::Implementation { traits, target } => {
                for ty in traits {
                    self.ty(*ty, item.span);
                }
                self.ty(*target, item.span);
            }
            hir::ItemKind::Function(function) => self.function(function, item.span),
            hir::ItemKind::ExternFunction(signature) => self.signature(signature, item.span),
            hir::ItemKind::Global(global) => {
                self.ty(global.ty, item.span);
                let _ = global.is_const;
                if let Some(initializer) = &global.initializer {
                    self.expression(initializer);
                }
            }
            hir::ItemKind::Test(block) => self.block(block),
            hir::ItemKind::Statement(statement) => self.statement(statement),
        }
    }

    fn signature(&mut self, signature: &hir::FunctionSignature, span: Span) {
        for parameter in &signature.parameters {
            self.ty(*parameter, span);
        }
        self.ty(signature.result, span);
    }

    fn function(&mut self, function: &hir::Function, span: Span) {
        if let Some(definition) = function.definition {
            self.definition(definition, span);
        }
        for parameter in &function.parameters {
            self.local(parameter.local, span);
            self.ty(parameter.ty, span);
            let _ = parameter.is_const;
        }
        self.ty(function.result, span);
        self.block(&function.body);
    }

    fn block(&mut self, block: &hir::Block) {
        self.span(block.span);
        let _ = block.source;
        let mut observed = hir::Flow::FallsThrough;
        for statement in &block.statements {
            self.statement(statement);
            if observed.continues() {
                let flow = statement_flow(&statement.kind);
                if !flow.continues() {
                    observed = flow;
                }
            }
        }
        if block.flow != observed {
            self.diagnostics.push(
                Diagnostic::error("HIR block has an inconsistent control-flow summary")
                    .with_code("E4001")
                    .at(block.span),
            );
        }
    }

    fn statement(&mut self, statement: &hir::Stmt) {
        self.span(statement.span);
        let _ = statement.source;
        match &statement.kind {
            hir::StmtKind::Local {
                local,
                ty,
                is_const,
                initializer,
            } => {
                self.local(*local, statement.span);
                self.ty(*ty, statement.span);
                let _ = is_const;
                if let Some(initializer) = initializer {
                    self.expression(initializer);
                }
            }
            hir::StmtKind::Destructure { value, bindings } => {
                self.expression(value);
                for (field, local) in bindings {
                    self.field(*field, statement.span);
                    self.local(*local, statement.span);
                }
            }
            hir::StmtKind::Assignment { target, value } => {
                self.expression(target);
                self.expression(value);
            }
            hir::StmtKind::Expression(expression)
            | hir::StmtKind::Defer(expression)
            | hir::StmtKind::Print(expression) => self.expression(expression),
            hir::StmtKind::Return(expression) => {
                if let Some(expression) = expression {
                    self.expression(expression);
                }
            }
            hir::StmtKind::Input => {}
            hir::StmtKind::Block(block) | hir::StmtKind::Unsafe(block) => self.block(block),
            hir::StmtKind::If(branch) => {
                self.expression(&branch.condition);
                self.block(&branch.then_block);
                for (condition, block) in &branch.else_if {
                    self.expression(condition);
                    self.block(block);
                }
                if let Some(block) = &branch.else_block {
                    self.block(block);
                }
                let _ = branch.flow;
            }
            hir::StmtKind::Match(branch) => {
                self.expression(&branch.value);
                for case in &branch.cases {
                    match &case.pattern {
                        hir::MatchPattern::EnumVariant { variant } => {
                            self.variant(*variant, statement.span)
                        }
                        hir::MatchPattern::Struct { definition, fields } => {
                            self.definition(*definition, statement.span);
                            for field in fields {
                                self.field(*field, statement.span);
                            }
                        }
                    }
                    for local in &case.bindings {
                        self.local(*local, statement.span);
                    }
                    self.block(&case.body);
                }
                let _ = branch.flow;
            }
            hir::StmtKind::For(loop_statement) => {
                if let Some(initializer) = &loop_statement.initializer {
                    self.statement(initializer);
                }
                if let Some(condition) = &loop_statement.condition {
                    self.expression(condition);
                }
                if let Some(update) = &loop_statement.update {
                    self.statement(update);
                }
                self.block(&loop_statement.body);
                let _ = loop_statement.flow;
            }
        }
    }

    fn expression(&mut self, expression: &hir::Expr) {
        self.span(expression.span);
        let _ = expression.source;
        self.ty(expression.ty, expression.span);
        match &expression.kind {
            hir::ExprKind::Literal(literal) => {
                let _ = literal;
            }
            hir::ExprKind::Value(value) => match value {
                hir::Value::Definition(definition) => self.definition(*definition, expression.span),
                hir::Value::Local(local) => self.local(*local, expression.span),
            },
            hir::ExprKind::Unary { operator, operand } => {
                let _ = operator;
                self.expression(operand);
            }
            hir::ExprKind::Binary {
                left,
                operator,
                right,
            } => {
                self.expression(left);
                let _ = operator;
                self.expression(right);
            }
            hir::ExprKind::Call {
                callee,
                argument_groups,
            } => {
                self.expression(callee);
                self.argument_groups(argument_groups);
            }
            hir::ExprKind::MethodCall {
                receiver,
                method,
                argument_groups,
            } => {
                self.expression(receiver);
                match method {
                    hir::MethodTarget::Definition(definition) => {
                        self.definition(*definition, expression.span)
                    }
                    hir::MethodTarget::Dynamic { owner, method } => {
                        self.ty(*owner, expression.span);
                        self.definition(*method, expression.span);
                    }
                    hir::MethodTarget::Intrinsic { owner, name } => {
                        self.ty(*owner, expression.span);
                        let _ = name;
                    }
                }
                self.argument_groups(argument_groups);
            }
            hir::ExprKind::Field { receiver, field } => {
                self.expression(receiver);
                self.field(*field, expression.span);
            }
            hir::ExprKind::Length { receiver } => self.expression(receiver),
            hir::ExprKind::Index {
                receiver,
                coordinates,
            } => {
                self.expression(receiver);
                for coordinate in coordinates {
                    self.expression(coordinate);
                }
            }
            hir::ExprKind::Slice {
                receiver,
                start,
                end,
            } => {
                self.expression(receiver);
                if let Some(start) = start {
                    self.expression(start);
                }
                if let Some(end) = end {
                    self.expression(end);
                }
            }
            hir::ExprKind::StructInit { definition, fields } => {
                self.definition(*definition, expression.span);
                for (field, value) in fields {
                    self.field(*field, expression.span);
                    self.expression(value);
                }
            }
            hir::ExprKind::StaticCall { target, arguments } => {
                match target {
                    hir::StaticTarget::Definition(definition) => {
                        self.definition(*definition, expression.span)
                    }
                    hir::StaticTarget::Variant(variant) => self.variant(*variant, expression.span),
                    hir::StaticTarget::Intrinsic { owner, name } => {
                        match owner {
                            hir::StaticOwner::Definition(definition) => {
                                self.definition(*definition, expression.span)
                            }
                            hir::StaticOwner::Intrinsic(intrinsic) => {
                                let _ = intrinsic;
                            }
                            hir::StaticOwner::Builtin(builtin) => {
                                let _ = builtin;
                            }
                        }
                        let _ = name;
                    }
                }
                for argument in arguments {
                    self.expression(argument);
                }
            }
            hir::ExprKind::Array(elements) => {
                for element in elements {
                    self.expression(element);
                }
            }
            hir::ExprKind::Lambda(function) => self.function(function, expression.span),
            hir::ExprKind::Block(block) => self.block(block),
        }
    }

    fn argument_groups(&mut self, groups: &[Vec<hir::Expr>]) {
        for group in groups {
            for argument in group {
                self.expression(argument);
            }
        }
    }

    fn ty(&mut self, ty: TypeId, span: Span) {
        if ty.index() >= self.model.types.len() {
            self.invalid_id("type", ty.index(), span);
            return;
        }
        if !self.checked_types.insert(ty) {
            return;
        }
        match self.model.types.kind(ty).clone() {
            TypeKind::Error => self.diagnostics.push(
                Diagnostic::error("error type escaped into validated HIR")
                    .with_code("E4002")
                    .at(span),
            ),
            TypeKind::Never | TypeKind::Builtin(_) | TypeKind::Intrinsic(_) => {}
            TypeKind::GenericParameter(definition) => self.definition(definition, span),
            TypeKind::Nominal {
                definition,
                arguments,
            } => {
                self.definition(definition, span);
                for argument in arguments {
                    self.ty(argument, span);
                }
            }
            TypeKind::Const(inner) | TypeKind::Pointer(inner) | TypeKind::Slice(inner) => {
                self.ty(inner, span)
            }
            TypeKind::Array {
                element,
                dimensions,
            } => {
                self.ty(element, span);
                let _ = dimensions;
            }
            TypeKind::Reference { target, mutable } => {
                self.ty(target, span);
                let _ = mutable;
            }
            TypeKind::Union(types) | TypeKind::Intersection(types) => {
                for ty in types {
                    self.ty(ty, span);
                }
            }
            TypeKind::Function { parameters, result } => {
                for parameter in parameters {
                    self.ty(parameter, span);
                }
                self.ty(result, span);
            }
        }
    }

    fn definition(&mut self, definition: DefId, span: Span) {
        let Some(record) = self.model.resolutions.definitions.get(definition.index()) else {
            self.invalid_id("definition", definition.index(), span);
            return;
        };
        let _ = (
            &record.id,
            &record.name,
            record.kind,
            record.visibility,
            record.span,
        );
    }

    fn local(&mut self, local: LocalId, span: Span) {
        let Some(record) = self.model.resolutions.locals.get(local.index()) else {
            self.invalid_id("local", local.index(), span);
            return;
        };
        let _ = (&record.id, &record.name, record.span, record.is_const);
    }

    fn field(&mut self, field: FieldId, span: Span) {
        if !self.model.field_types.contains_key(&field) {
            self.invalid_id("field", field.index(), span);
        }
    }

    fn variant(&mut self, variant: VariantId, span: Span) {
        if !self.model.variant_payloads.contains_key(&variant) {
            self.invalid_id("variant", variant.index(), span);
        }
    }

    fn span(&mut self, span: Span) {
        if span.start > span.end {
            self.diagnostics.push(
                Diagnostic::error("HIR contains an inverted source span")
                    .with_code("E4003")
                    .at(span),
            );
        }
        let _ = span.file;
    }

    fn invalid_id(&mut self, kind: &str, index: usize, span: Span) {
        self.diagnostics.push(
            Diagnostic::error(format!("HIR references unknown {kind} id {index}"))
                .with_code("E4000")
                .at(span),
        );
    }
}

fn statement_flow(kind: &hir::StmtKind) -> hir::Flow {
    match kind {
        hir::StmtKind::Return(_) => hir::Flow::Returns,
        hir::StmtKind::Block(block) | hir::StmtKind::Unsafe(block) => block.flow,
        hir::StmtKind::If(branch) => branch.flow,
        hir::StmtKind::Match(branch) => branch.flow,
        hir::StmtKind::For(loop_statement) => loop_statement.flow,
        _ => hir::Flow::FallsThrough,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ids::FileId;

    #[test]
    fn accepts_well_formed_hir() {
        let source = "function main(): void { print(42); }";
        let legacy = crate::ast::try_parse(source).unwrap();
        let module = crate::syntax::from_legacy(&legacy, FileId::new(0), source.len()).unwrap();
        let resolutions = crate::resolver::resolve(&module).unwrap();
        let mut model = crate::semantics::analyze_declarations(&module, resolutions).unwrap();
        let hir = crate::hir_lowering::lower(&module, &mut model).unwrap();

        validate(&hir, &model).unwrap();
    }
}
