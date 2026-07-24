//! Name resolution and lexical scope construction.
//!
//! Resolution assigns stable identities without performing type checking. It
//! is the only phase that turns source-level names into `DefId`/`LocalId`.

use crate::diagnostic::Diagnostic;
use crate::ids::{DefId, FieldId, LocalId, NodeId, VariantId};
use crate::intrinsics::IntrinsicType;
use crate::source_map::Span;
use crate::syntax::ast::*;
use std::collections::HashMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DefinitionKind {
    TypeAlias,
    Struct,
    Enum,
    Trait,
    Shape,
    Function,
    ExternFunction,
    Global,
    Test,
    GenericParameter,
    Method,
}

#[derive(Clone, Debug)]
pub struct Definition {
    pub id: DefId,
    pub name: String,
    pub span: Span,
    pub kind: DefinitionKind,
    pub visibility: Visibility,
}

#[derive(Clone, Debug)]
pub struct LocalDefinition {
    pub id: LocalId,
    pub name: String,
    pub span: Span,
    pub is_const: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ValueResolution {
    Definition(DefId),
    Local(LocalId),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TypeResolution {
    Definition(DefId),
    GenericParameter(DefId),
    Builtin(BuiltinType),
    Intrinsic(IntrinsicType),
    SelfType,
}

#[derive(Debug, Default)]
pub struct Resolutions {
    pub definitions: Vec<Definition>,
    pub locals: Vec<LocalDefinition>,
    pub item_definitions: HashMap<NodeId, DefId>,
    pub function_definitions: HashMap<NodeId, DefId>,
    pub generic_definitions: HashMap<NodeId, DefId>,
    pub field_definitions: HashMap<NodeId, FieldId>,
    pub variant_definitions: HashMap<NodeId, VariantId>,
    pub supertrait_definitions: HashMap<NodeId, Vec<DefId>>,
    pub local_bindings: HashMap<NodeId, Vec<LocalId>>,
    pub values: HashMap<NodeId, ValueResolution>,
    pub types: HashMap<NodeId, TypeResolution>,
}

pub fn resolve(module: &Module) -> Result<Resolutions, Vec<Diagnostic>> {
    let mut resolver = Resolver::default();
    resolver.collect_top_level(module);
    resolver.resolve_module(module);
    if resolver.diagnostics.is_empty() {
        Ok(resolver.resolutions)
    } else {
        Err(resolver.diagnostics)
    }
}

#[derive(Default)]
struct Resolver {
    resolutions: Resolutions,
    value_definitions: HashMap<String, DefId>,
    type_definitions: HashMap<String, DefId>,
    value_scopes: Vec<HashMap<String, LocalId>>,
    type_scopes: Vec<HashMap<String, DefId>>,
    diagnostics: Vec<Diagnostic>,
}

impl Resolver {
    fn collect_top_level(&mut self, module: &Module) {
        for entry in &module.entries {
            let Some((name, kind, namespace)) = definition_header(&entry.kind) else {
                continue;
            };

            let existing = match namespace {
                Namespace::Value => self.value_definitions.get(name).copied(),
                Namespace::Type => self.type_definitions.get(name).copied(),
            };

            // Repeated extern declarations are allowed for imported C APIs.
            // Signature compatibility remains a semantic-checking concern.
            if let Some(previous) = existing {
                if kind == DefinitionKind::ExternFunction {
                    self.resolutions.item_definitions.insert(entry.id, previous);
                    continue;
                }
                let previous = &self.resolutions.definitions[previous.index()];
                self.diagnostics.push(
                    Diagnostic::error(format!("duplicate declaration of `{name}`"))
                        .with_code("E2001")
                        .at(entry.span)
                        .label(previous.span, "previous declaration is here"),
                );
                continue;
            }

            let id = self.allocate_definition(name.to_string(), entry.span, kind, entry.visibility);
            self.resolutions.item_definitions.insert(entry.id, id);
            match namespace {
                Namespace::Value => {
                    self.value_definitions.insert(name.to_string(), id);
                }
                Namespace::Type => {
                    self.type_definitions.insert(name.to_string(), id);
                }
            }
        }
    }

    fn allocate_definition(
        &mut self,
        name: String,
        span: Span,
        kind: DefinitionKind,
        visibility: Visibility,
    ) -> DefId {
        let id = DefId::new(self.resolutions.definitions.len() as u32);
        self.resolutions.definitions.push(Definition {
            id,
            name,
            span,
            kind,
            visibility,
        });
        id
    }

    fn resolve_module(&mut self, module: &Module) {
        for entry in &module.entries {
            self.resolve_top_level(entry);
        }
    }

    fn resolve_top_level(&mut self, entry: &TopLevel) {
        match &entry.kind {
            TopLevelKind::Import(_) => {}
            TopLevelKind::TypeAlias(alias) => {
                self.with_generic_parameters(&alias.generic_parameters, |resolver| {
                    resolver.resolve_type(&alias.target)
                });
            }
            TopLevelKind::Struct(declaration) => {
                self.with_generic_parameters(&declaration.generic_parameters, |resolver| {
                    resolver.check_duplicate_named_nodes(
                        declaration
                            .fields
                            .iter()
                            .map(|field| (&field.name, field.span)),
                        "field",
                    );
                    for field in &declaration.fields {
                        let id = FieldId::new(resolver.resolutions.field_definitions.len() as u32);
                        resolver.resolutions.field_definitions.insert(field.id, id);
                        resolver.resolve_type(&field.ty);
                    }
                    resolver.resolve_methods(&declaration.methods);
                });
            }
            TopLevelKind::Enum(declaration) => {
                self.with_generic_parameters(&declaration.generic_parameters, |resolver| {
                    resolver.check_duplicate_named_nodes(
                        declaration
                            .variants
                            .iter()
                            .map(|variant| (&variant.name, variant.span)),
                        "enum variant",
                    );
                    for variant in &declaration.variants {
                        let id =
                            VariantId::new(resolver.resolutions.variant_definitions.len() as u32);
                        resolver
                            .resolutions
                            .variant_definitions
                            .insert(variant.id, id);
                        for payload in &variant.payload {
                            resolver.resolve_type(payload);
                        }
                    }
                    resolver.resolve_methods(&declaration.methods);
                });
            }
            TopLevelKind::Trait(declaration) => {
                self.with_generic_parameters(&declaration.generic_parameters, |resolver| {
                    let supertraits = declaration
                        .supertraits
                        .iter()
                        .filter_map(|path| resolver.resolve_supertrait(path, entry.span))
                        .collect();
                    resolver
                        .resolutions
                        .supertrait_definitions
                        .insert(entry.id, supertraits);
                    resolver.resolve_trait_methods(&declaration.methods);
                });
            }
            TopLevelKind::Shape(declaration) => {
                self.resolve_trait_methods(&declaration.methods);
            }
            TopLevelKind::Attach(declaration) => {
                self.with_generic_parameters(&declaration.generic_parameters, |resolver| {
                    resolver.resolve_type(&declaration.target);
                    resolver.resolve_methods(&declaration.methods);
                });
            }
            TopLevelKind::Conformance(declaration) => {
                self.with_generic_parameters(&declaration.generic_parameters, |resolver| {
                    for trait_type in &declaration.traits {
                        resolver.resolve_type(trait_type);
                    }
                    resolver.resolve_type(&declaration.target);
                    resolver.resolve_methods(&declaration.methods);
                });
            }
            TopLevelKind::Implementation(declaration) => {
                self.with_generic_parameters(&declaration.generic_parameters, |resolver| {
                    for trait_type in &declaration.traits {
                        resolver.resolve_type(trait_type);
                    }
                    resolver.resolve_type(&declaration.target);
                });
            }
            TopLevelKind::Function(function) => self.resolve_function(function),
            TopLevelKind::ExternFunction(function) => {
                self.with_value_scope(|resolver| {
                    for parameter in &function.parameters {
                        resolver.resolve_parameter(parameter);
                    }
                    resolver.resolve_type(&function.return_type);
                });
            }
            TopLevelKind::Global(global) => {
                self.resolve_type(&global.ty);
                if let Some(initializer) = &global.initializer {
                    self.resolve_expr(initializer);
                }
            }
            TopLevelKind::Test(test) => {
                self.with_value_scope(|resolver| resolver.resolve_block_contents(&test.body));
            }
            TopLevelKind::Statement(statement) => self.resolve_stmt(statement),
        }
    }

    fn resolve_methods(&mut self, methods: &[FunctionDecl]) {
        let mut names = HashMap::<&str, Span>::new();
        for method in methods {
            if let Some(previous) = names.insert(&method.name, method.body.span) {
                self.diagnostics.push(
                    Diagnostic::error(format!("duplicate method `{}`", method.name))
                        .with_code("E2002")
                        .at(method.body.span)
                        .label(previous, "previous method is here"),
                );
            }
            let definition = self.allocate_definition(
                method.name.clone(),
                method.body.span,
                DefinitionKind::Method,
                Visibility::Private,
            );
            self.resolutions
                .function_definitions
                .insert(method.body.id, definition);
            self.resolve_function(method);
        }
    }

    fn resolve_trait_method(&mut self, method: &TraitMethod) {
        self.with_value_scope(|resolver| {
            for parameter in &method.parameters {
                resolver.resolve_parameter(parameter);
            }
            resolver.resolve_type(&method.return_type);
            if let Some(body) = &method.default_body {
                resolver.resolve_block_contents(body);
            }
        });
    }

    fn resolve_trait_methods(&mut self, methods: &[TraitMethod]) {
        self.check_duplicate_named_nodes(
            methods.iter().map(|method| (&method.name, method.span)),
            "trait method",
        );
        for method in methods {
            let definition = self.allocate_definition(
                method.name.clone(),
                method.span,
                DefinitionKind::Method,
                Visibility::Private,
            );
            self.resolutions
                .function_definitions
                .insert(method.id, definition);
            self.resolve_trait_method(method);
        }
    }

    fn resolve_supertrait(&mut self, path: &Path, span: Span) -> Option<DefId> {
        if path.segments.len() != 1 {
            self.diagnostics.push(
                Diagnostic::error(format!(
                    "qualified supertrait `{}` is not supported yet",
                    path.qualified_name()
                ))
                .with_code("E2006")
                .at(span),
            );
            return None;
        }
        let name = &path.segments[0];
        let Some(definition) = self.type_definitions.get(name).copied() else {
            self.diagnostics.push(
                Diagnostic::error(format!("unknown supertrait `{name}`"))
                    .with_code("E2006")
                    .at(span),
            );
            return None;
        };
        let kind = self.resolutions.definitions[definition.index()].kind;
        if kind != DefinitionKind::Trait {
            self.diagnostics.push(
                Diagnostic::error(format!("`{name}` is not a trait"))
                    .with_code("E2006")
                    .at(span),
            );
            return None;
        }
        Some(definition)
    }

    fn resolve_function(&mut self, function: &FunctionDecl) {
        self.with_generic_parameters(&function.generic_parameters, |resolver| {
            resolver.with_value_scope(|resolver| {
                for parameter in &function.parameters {
                    resolver.resolve_parameter(parameter);
                }
                resolver.resolve_type(&function.return_type);
                resolver.resolve_block_contents(&function.body);
            });
        });
    }

    fn resolve_parameter(&mut self, parameter: &Parameter) {
        match &parameter.kind {
            ParameterKind::Named { name, is_const, ty } => {
                self.resolve_type(ty);
                if let Some(local) = self.declare_local(name, parameter.span, *is_const) {
                    self.resolutions
                        .local_bindings
                        .insert(parameter.id, vec![local]);
                }
            }
            ParameterKind::Receiver { is_const, .. } => {
                if let Some(local) = self.declare_local("self", parameter.span, *is_const) {
                    self.resolutions
                        .local_bindings
                        .insert(parameter.id, vec![local]);
                }
            }
        }
    }

    fn resolve_block(&mut self, block: &Block) {
        self.with_value_scope(|resolver| resolver.resolve_block_contents(block));
    }

    fn resolve_block_contents(&mut self, block: &Block) {
        for statement in &block.statements {
            self.resolve_stmt(statement);
        }
    }

    fn resolve_stmt(&mut self, statement: &Stmt) {
        match &statement.kind {
            StmtKind::Local(local) => {
                self.resolve_type(&local.ty);
                // Bind before resolving the initializer so recursive lambdas
                // have a stable local identity.
                if let Some(binding) =
                    self.declare_local(&local.name, statement.span, local.is_const)
                {
                    self.resolutions
                        .local_bindings
                        .insert(statement.id, vec![binding]);
                }
                if let Some(initializer) = &local.initializer {
                    self.resolve_expr(initializer);
                }
            }
            StmtKind::StructDestructure(pattern) => {
                self.resolve_type(&pattern.ty);
                self.resolve_expr(&pattern.value);
                let mut bindings = Vec::new();
                for field in &pattern.fields {
                    if let Some(binding) = self.declare_local(&field.binding, statement.span, false)
                    {
                        bindings.push(binding);
                    }
                }
                self.resolutions
                    .local_bindings
                    .insert(statement.id, bindings);
            }
            StmtKind::Assignment { target, value } => {
                self.resolve_expr(target);
                self.resolve_expr(value);
            }
            StmtKind::Expression(expression)
            | StmtKind::Defer(expression)
            | StmtKind::Print(expression) => self.resolve_expr(expression),
            StmtKind::Return(expression) => {
                if let Some(expression) = expression {
                    self.resolve_expr(expression);
                }
            }
            StmtKind::Input => {}
            StmtKind::Declaration(declaration) => self.resolve_top_level(declaration),
            StmtKind::Block(block) | StmtKind::Unsafe(block) => self.resolve_block(block),
            StmtKind::If(expression) => {
                self.resolve_expr(&expression.condition);
                self.resolve_block(&expression.then_block);
                for (condition, block) in &expression.else_if {
                    self.resolve_expr(condition);
                    self.resolve_block(block);
                }
                if let Some(block) = &expression.else_block {
                    self.resolve_block(block);
                }
            }
            StmtKind::Match(expression) => {
                self.resolve_expr(&expression.value);
                for case in &expression.cases {
                    self.with_value_scope(|resolver| {
                        resolver.resolve_pattern(&case.pattern, case.id, case.span);
                        resolver.resolve_block_contents(&case.body);
                    });
                }
            }
            StmtKind::For(statement) => {
                self.with_value_scope(|resolver| {
                    if let Some(initializer) = statement.initializer.as_deref() {
                        resolver.resolve_stmt(initializer);
                    }
                    if let Some(condition) = &statement.condition {
                        resolver.resolve_expr(condition);
                    }
                    if let Some(update) = statement.update.as_deref() {
                        resolver.resolve_stmt(update);
                    }
                    resolver.resolve_block(&statement.body);
                });
            }
        }
    }

    fn resolve_pattern(&mut self, pattern: &Pattern, node: NodeId, span: Span) {
        let mut resolved_bindings = Vec::new();
        match pattern {
            Pattern::EnumVariant {
                enum_type,
                bindings,
                ..
            } => {
                if let Some(enum_type) = enum_type {
                    self.resolve_type(enum_type);
                }
                for binding in bindings {
                    if let Some(binding) = self.declare_local(binding, span, false) {
                        resolved_bindings.push(binding);
                    }
                }
            }
            Pattern::Struct { ty, fields } => {
                self.resolve_type(ty);
                for field in fields {
                    if let Some(binding) = self.declare_local(&field.binding, span, false) {
                        resolved_bindings.push(binding);
                    }
                }
            }
        }
        self.resolutions
            .local_bindings
            .insert(node, resolved_bindings);
    }

    fn resolve_expr(&mut self, expression: &Expr) {
        match &expression.kind {
            ExprKind::Literal(_) => {}
            ExprKind::Name(path) => {
                self.resolve_value_name(expression, path, "value");
            }
            ExprKind::Unary { operand, .. } => self.resolve_expr(operand),
            ExprKind::Binary { left, right, .. } => {
                self.resolve_expr(left);
                self.resolve_expr(right);
            }
            ExprKind::Call {
                callee,
                type_arguments,
                argument_groups,
            } => {
                if let ExprKind::Name(path) = &callee.kind {
                    self.resolve_value_name(callee, path, "function");
                } else {
                    self.resolve_expr(callee);
                }
                for ty in type_arguments {
                    self.resolve_type(ty);
                }
                for group in argument_groups {
                    for argument in group {
                        self.resolve_expr(argument);
                    }
                }
            }
            ExprKind::Field { receiver, .. } => self.resolve_expr(receiver),
            ExprKind::Index {
                receiver,
                coordinates,
            } => {
                self.resolve_expr(receiver);
                for coordinate in coordinates {
                    self.resolve_expr(coordinate);
                }
            }
            ExprKind::Slice {
                receiver,
                start,
                end,
            } => {
                self.resolve_expr(receiver);
                if let Some(start) = start {
                    self.resolve_expr(start);
                }
                if let Some(end) = end {
                    self.resolve_expr(end);
                }
            }
            ExprKind::StructInit { ty, fields } => {
                self.resolve_type(ty);
                for (_, value) in fields {
                    self.resolve_expr(value);
                }
            }
            ExprKind::StaticCall { ty, arguments, .. } => {
                self.resolve_type(ty);
                for argument in arguments {
                    self.resolve_expr(argument);
                }
            }
            ExprKind::Array(elements) => {
                for element in elements {
                    self.resolve_expr(element);
                }
            }
            ExprKind::Lambda(lambda) => {
                self.with_value_scope(|resolver| {
                    for parameter in &lambda.parameters {
                        resolver.resolve_parameter(parameter);
                    }
                    resolver.resolve_type(&lambda.return_type);
                    resolver.resolve_block_contents(&lambda.body);
                });
            }
            ExprKind::Block(block) => self.resolve_block(block),
        }
    }

    fn resolve_type(&mut self, ty: &TypeSyntax) {
        match &ty.kind {
            TypeSyntaxKind::Builtin(builtin) => {
                self.resolutions
                    .types
                    .insert(ty.id, TypeResolution::Builtin(*builtin));
            }
            TypeSyntaxKind::Named { path, arguments } => {
                if path.segments.len() == 1 {
                    let name = &path.segments[0];
                    if let Some(parameter) = self.lookup_type_parameter(name) {
                        self.resolutions
                            .types
                            .insert(ty.id, TypeResolution::GenericParameter(parameter));
                    } else if let Some(definition) = self.type_definitions.get(name).copied() {
                        self.resolutions
                            .types
                            .insert(ty.id, TypeResolution::Definition(definition));
                    } else if let Some(intrinsic) = IntrinsicType::from_name(name) {
                        self.resolutions
                            .types
                            .insert(ty.id, TypeResolution::Intrinsic(intrinsic));
                    } else {
                        self.diagnostics.push(
                            Diagnostic::error(format!("unknown type `{name}`"))
                                .with_code("E2004")
                                .at(ty.span),
                        );
                    }
                }
                for argument in arguments {
                    self.resolve_type(argument);
                }
            }
            TypeSyntaxKind::Const(inner)
            | TypeSyntaxKind::Pointer(inner)
            | TypeSyntaxKind::Slice(inner) => self.resolve_type(inner),
            TypeSyntaxKind::Array {
                element,
                dimensions,
            } => {
                self.resolve_type(element);
                for dimension in dimensions {
                    self.resolve_expr(dimension);
                }
            }
            TypeSyntaxKind::Reference { target, .. } => self.resolve_type(target),
            TypeSyntaxKind::Union(types) | TypeSyntaxKind::Intersection(types) => {
                for ty in types {
                    self.resolve_type(ty);
                }
            }
            TypeSyntaxKind::Function { parameters, result } => {
                for parameter in parameters {
                    self.resolve_type(parameter);
                }
                self.resolve_type(result);
            }
            TypeSyntaxKind::SelfType { .. } => {
                self.resolutions
                    .types
                    .insert(ty.id, TypeResolution::SelfType);
            }
        }
    }

    fn with_value_scope(&mut self, operation: impl FnOnce(&mut Self)) {
        self.value_scopes.push(HashMap::new());
        operation(self);
        self.value_scopes.pop();
    }

    fn resolve_value_name(&mut self, expression: &Expr, path: &Path, role: &str) {
        if path.segments.len() != 1 {
            return;
        }
        let name = &path.segments[0];
        if let Some(local) = self.lookup_local(name) {
            self.resolutions
                .values
                .insert(expression.id, ValueResolution::Local(local));
        } else if let Some(definition) = self.value_definitions.get(name).copied() {
            self.resolutions
                .values
                .insert(expression.id, ValueResolution::Definition(definition));
        } else {
            self.diagnostics.push(
                Diagnostic::error(format!("unknown {role} `{name}`"))
                    .with_code("E2003")
                    .at(expression.span),
            );
        }
    }

    fn with_generic_parameters(
        &mut self,
        parameters: &[GenericParameter],
        operation: impl FnOnce(&mut Self),
    ) {
        self.type_scopes.push(HashMap::new());
        for parameter in parameters {
            let duplicate = self
                .type_scopes
                .last()
                .is_some_and(|scope| scope.contains_key(&parameter.name));
            if duplicate {
                self.diagnostics.push(
                    Diagnostic::error(format!("duplicate generic parameter `{}`", parameter.name))
                        .with_code("E2005")
                        .at(parameter.span),
                );
                continue;
            }
            let id = self.allocate_definition(
                parameter.name.clone(),
                parameter.span,
                DefinitionKind::GenericParameter,
                Visibility::Private,
            );
            self.resolutions
                .generic_definitions
                .insert(parameter.id, id);
            if let Some(scope) = self.type_scopes.last_mut() {
                scope.insert(parameter.name.clone(), id);
            } else {
                self.diagnostics.push(
                    Diagnostic::error("internal resolver error: missing generic type scope")
                        .with_code("E2099")
                        .at(parameter.span),
                );
            }
        }

        for parameter in parameters {
            if let Some(lower) = &parameter.lower_bound {
                self.resolve_type(lower);
            }
            if let Some(upper) = &parameter.upper_bound {
                self.resolve_type(upper);
            }
        }
        operation(self);
        self.type_scopes.pop();
    }

    fn declare_local(&mut self, name: &str, span: Span, is_const: bool) -> Option<LocalId> {
        let scope = self.value_scopes.last()?;
        if let Some(previous_id) = scope.get(name).copied() {
            let previous = &self.resolutions.locals[previous_id.index()];
            self.diagnostics.push(
                Diagnostic::error(format!("duplicate local binding `{name}`"))
                    .with_code("E2006")
                    .at(span)
                    .label(previous.span, "previous binding is here"),
            );
            return None;
        }
        let id = LocalId::new(self.resolutions.locals.len() as u32);
        self.resolutions.locals.push(LocalDefinition {
            id,
            name: name.to_string(),
            span,
            is_const,
        });
        self.value_scopes.last_mut()?.insert(name.to_string(), id);
        Some(id)
    }

    fn lookup_local(&self, name: &str) -> Option<LocalId> {
        self.value_scopes
            .iter()
            .rev()
            .find_map(|scope| scope.get(name).copied())
    }

    fn lookup_type_parameter(&self, name: &str) -> Option<DefId> {
        self.type_scopes
            .iter()
            .rev()
            .find_map(|scope| scope.get(name).copied())
    }

    fn check_duplicate_named_nodes<'a>(
        &mut self,
        nodes: impl Iterator<Item = (&'a String, Span)>,
        description: &str,
    ) {
        let mut names = HashMap::<&str, Span>::new();
        for (name, span) in nodes {
            if let Some(previous) = names.insert(name, span) {
                self.diagnostics.push(
                    Diagnostic::error(format!("duplicate {description} `{name}`"))
                        .with_code("E2007")
                        .at(span)
                        .label(previous, format!("previous {description} is here")),
                );
            }
        }
    }
}

#[derive(Clone, Copy)]
enum Namespace {
    Value,
    Type,
}

fn definition_header(kind: &TopLevelKind) -> Option<(&str, DefinitionKind, Namespace)> {
    match kind {
        TopLevelKind::TypeAlias(declaration) => Some((
            &declaration.name,
            DefinitionKind::TypeAlias,
            Namespace::Type,
        )),
        TopLevelKind::Struct(declaration) => {
            Some((&declaration.name, DefinitionKind::Struct, Namespace::Type))
        }
        TopLevelKind::Enum(declaration) => {
            Some((&declaration.name, DefinitionKind::Enum, Namespace::Type))
        }
        TopLevelKind::Trait(declaration) => {
            Some((&declaration.name, DefinitionKind::Trait, Namespace::Type))
        }
        TopLevelKind::Shape(declaration) => {
            Some((&declaration.name, DefinitionKind::Shape, Namespace::Type))
        }
        TopLevelKind::Function(declaration) => Some((
            &declaration.name,
            DefinitionKind::Function,
            Namespace::Value,
        )),
        TopLevelKind::ExternFunction(declaration) => Some((
            &declaration.name,
            DefinitionKind::ExternFunction,
            Namespace::Value,
        )),
        TopLevelKind::Global(declaration) => {
            Some((&declaration.name, DefinitionKind::Global, Namespace::Value))
        }
        TopLevelKind::Test(declaration) => {
            Some((&declaration.name, DefinitionKind::Test, Namespace::Value))
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source_map::SourceMap;

    fn parse(source: &str) -> Module {
        let mut sources = SourceMap::default();
        let file = sources.add_file("test.skunk", source).unwrap();
        crate::syntax::parser::parse_module(&sources, file).unwrap()
    }

    #[test]
    fn rejects_duplicate_top_level_declarations() {
        let module = parse("function duplicate(): void {} function duplicate(): void {}");
        let diagnostics = resolve(&module).unwrap_err();
        assert!(diagnostics[0].message.contains("duplicate declaration"));
    }

    #[test]
    fn nested_block_bindings_do_not_escape() {
        let module = parse(
            r#"
                function main(): void {
                    { hidden: int = 1; }
                    print(hidden);
                }
            "#,
        );
        let diagnostics = resolve(&module).unwrap_err();
        assert!(diagnostics
            .iter()
            .any(|diagnostic| diagnostic.message == "unknown value `hidden`"));
    }

    #[test]
    fn shadowing_in_a_nested_block_is_allowed() {
        let module = parse(
            r#"
                function main(): void {
                    value: int = 1;
                    { value: int = 2; print(value); }
                    print(value);
                }
            "#,
        );
        resolve(&module).unwrap();
    }
}
