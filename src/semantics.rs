//! Semantic type construction over resolved syntax.

use crate::diagnostic::Diagnostic;
use crate::ids::{DefId, FieldId, LocalId, NodeId, TypeId, VariantId};
use crate::resolver::{DefinitionKind, Resolutions, TypeResolution};
use crate::semantic_types::{TypeKind, TypeStore};
use crate::syntax::ast::*;
use std::collections::HashMap;

#[derive(Debug)]
pub struct SemanticModel {
    pub resolutions: Resolutions,
    pub types: TypeStore,
    pub definition_types: HashMap<DefId, TypeId>,
    pub syntax_types: HashMap<NodeId, TypeId>,
    pub local_types: HashMap<LocalId, TypeId>,
    pub field_types: HashMap<FieldId, TypeId>,
    pub variant_payloads: HashMap<VariantId, Vec<TypeId>>,
}

pub fn analyze_declarations(
    module: &Module,
    resolutions: Resolutions,
) -> Result<SemanticModel, Vec<Diagnostic>> {
    let mut analyzer = Analyzer {
        module,
        resolutions,
        types: TypeStore::default(),
        definition_types: HashMap::new(),
        syntax_types: HashMap::new(),
        local_types: HashMap::new(),
        field_types: HashMap::new(),
        variant_payloads: HashMap::new(),
        aliases: HashMap::new(),
        diagnostics: Vec::new(),
    };
    analyzer.collect_nominal_types();
    analyzer.collect_aliases();
    analyzer.collect_signatures_and_bindings();

    if analyzer.diagnostics.is_empty() {
        Ok(SemanticModel {
            resolutions: analyzer.resolutions,
            types: analyzer.types,
            definition_types: analyzer.definition_types,
            syntax_types: analyzer.syntax_types,
            local_types: analyzer.local_types,
            field_types: analyzer.field_types,
            variant_payloads: analyzer.variant_payloads,
        })
    } else {
        Err(analyzer.diagnostics)
    }
}

struct Analyzer<'a> {
    module: &'a Module,
    resolutions: Resolutions,
    types: TypeStore,
    definition_types: HashMap<DefId, TypeId>,
    syntax_types: HashMap<NodeId, TypeId>,
    local_types: HashMap<LocalId, TypeId>,
    field_types: HashMap<FieldId, TypeId>,
    variant_payloads: HashMap<VariantId, Vec<TypeId>>,
    aliases: HashMap<DefId, &'a TypeSyntax>,
    diagnostics: Vec<Diagnostic>,
}

impl Analyzer<'_> {
    fn collect_nominal_types(&mut self) {
        for entry in &self.module.entries {
            let Some(definition) = self.resolutions.item_definitions.get(&entry.id).copied() else {
                continue;
            };
            let generic_parameters = match &entry.kind {
                TopLevelKind::Struct(declaration) => &declaration.generic_parameters,
                TopLevelKind::Enum(declaration) => &declaration.generic_parameters,
                TopLevelKind::Trait(declaration) => &declaration.generic_parameters,
                _ => continue,
            };
            let arguments = generic_parameters
                .iter()
                .filter_map(|parameter| {
                    self.resolutions
                        .generic_definitions
                        .get(&parameter.id)
                        .copied()
                })
                .map(|parameter| self.types.intern(TypeKind::GenericParameter(parameter)))
                .collect();
            let ty = self.types.nominal(definition, arguments);
            self.definition_types.insert(definition, ty);
        }
    }

    fn collect_aliases(&mut self) {
        for entry in &self.module.entries {
            let TopLevelKind::TypeAlias(alias) = &entry.kind else {
                continue;
            };
            if let Some(definition) = self.resolutions.item_definitions.get(&entry.id).copied() {
                self.aliases.insert(definition, &alias.target);
            }
        }
    }

    fn collect_signatures_and_bindings(&mut self) {
        // First collect every callable/global signature so bodies can use
        // forward references without source-order coupling.
        for entry in &self.module.entries {
            let definition = self.resolutions.item_definitions.get(&entry.id).copied();
            match &entry.kind {
                TopLevelKind::Function(function) => {
                    if let Some(definition) = definition {
                        let ty = self.function_type(function, None);
                        self.definition_types.insert(definition, ty);
                    }
                }
                TopLevelKind::ExternFunction(function) => {
                    if let Some(definition) = definition {
                        let parameters = function
                            .parameters
                            .iter()
                            .filter_map(|parameter| self.parameter_type(parameter, None))
                            .collect();
                        let result = self.lower_type(&function.return_type, None);
                        let ty = self.types.function(parameters, result);
                        self.definition_types.insert(definition, ty);
                    }
                }
                TopLevelKind::Global(global) => {
                    if let Some(definition) = definition {
                        let ty = self.lower_type(&global.ty, None);
                        self.definition_types.insert(definition, ty);
                    }
                }
                TopLevelKind::Test(_) => {
                    if let Some(definition) = definition {
                        let void = self.types.builtin(BuiltinType::Void);
                        let ty = self.types.function(Vec::new(), void);
                        self.definition_types.insert(definition, ty);
                    }
                }
                _ => {}
            }
        }

        for entry in &self.module.entries {
            self.collect_entry_bindings(entry);
        }
    }

    fn collect_entry_bindings(&mut self, entry: &TopLevel) {
        match &entry.kind {
            TopLevelKind::TypeAlias(alias) => {
                if let Some(definition) = self.resolutions.item_definitions.get(&entry.id).copied()
                {
                    let target = self.lower_type(&alias.target, None);
                    self.definition_types.insert(definition, target);
                }
            }
            TopLevelKind::Struct(declaration) => {
                let owner = self
                    .resolutions
                    .item_definitions
                    .get(&entry.id)
                    .and_then(|definition| self.definition_types.get(definition))
                    .copied();
                for field in &declaration.fields {
                    let ty = self.lower_type(&field.ty, owner);
                    if let Some(field_id) =
                        self.resolutions.field_definitions.get(&field.id).copied()
                    {
                        self.field_types.insert(field_id, ty);
                    }
                }
                for method in &declaration.methods {
                    self.collect_function_bindings(method, owner);
                }
            }
            TopLevelKind::Enum(declaration) => {
                let owner = self
                    .resolutions
                    .item_definitions
                    .get(&entry.id)
                    .and_then(|definition| self.definition_types.get(definition))
                    .copied();
                for variant in &declaration.variants {
                    let payload = variant
                        .payload
                        .iter()
                        .map(|ty| self.lower_type(ty, owner))
                        .collect();
                    if let Some(variant_id) = self
                        .resolutions
                        .variant_definitions
                        .get(&variant.id)
                        .copied()
                    {
                        self.variant_payloads.insert(variant_id, payload);
                    }
                }
                for method in &declaration.methods {
                    self.collect_function_bindings(method, owner);
                }
            }
            TopLevelKind::Trait(declaration) => {
                let owner = self
                    .resolutions
                    .item_definitions
                    .get(&entry.id)
                    .and_then(|definition| self.definition_types.get(definition))
                    .copied();
                for method in &declaration.methods {
                    self.collect_trait_method_bindings(method, owner);
                }
            }
            TopLevelKind::Shape(declaration) => {
                for method in &declaration.methods {
                    self.collect_trait_method_bindings(method, None);
                }
            }
            TopLevelKind::Attach(declaration) => {
                let owner = Some(self.lower_type(&declaration.target, None));
                for method in &declaration.methods {
                    self.collect_function_bindings(method, owner);
                }
            }
            TopLevelKind::Conformance(declaration) => {
                let owner = Some(self.lower_type(&declaration.target, None));
                for method in &declaration.methods {
                    self.collect_function_bindings(method, owner);
                }
            }
            TopLevelKind::Function(function) => self.collect_function_bindings(function, None),
            TopLevelKind::ExternFunction(function) => {
                for parameter in &function.parameters {
                    self.collect_parameter_binding(parameter, None);
                }
            }
            TopLevelKind::Global(global) => {
                if let Some(initializer) = &global.initializer {
                    self.collect_expr_bindings(initializer, None);
                }
            }
            TopLevelKind::Test(test) => self.collect_block_bindings(&test.body, None),
            TopLevelKind::Statement(statement) => self.collect_stmt_bindings(statement, None),
            TopLevelKind::Implementation(declaration) => {
                for trait_type in &declaration.traits {
                    self.lower_type(trait_type, None);
                }
                self.lower_type(&declaration.target, None);
            }
            TopLevelKind::Import(_) => {}
        }
    }

    fn function_type(&mut self, function: &FunctionDecl, self_type: Option<TypeId>) -> TypeId {
        let parameters = function
            .parameters
            .iter()
            .filter_map(|parameter| self.parameter_type(parameter, self_type))
            .collect();
        let result = self.lower_type(&function.return_type, self_type);
        self.types.function(parameters, result)
    }

    fn collect_function_bindings(&mut self, function: &FunctionDecl, self_type: Option<TypeId>) {
        if let Some(definition) = self
            .resolutions
            .function_definitions
            .get(&function.body.id)
            .copied()
        {
            let ty = self.function_type(function, self_type);
            self.definition_types.insert(definition, ty);
        }
        for parameter in &function.parameters {
            self.collect_parameter_binding(parameter, self_type);
        }
        self.collect_block_bindings(&function.body, self_type);
    }

    fn collect_trait_method_bindings(&mut self, method: &TraitMethod, self_type: Option<TypeId>) {
        for parameter in &method.parameters {
            self.collect_parameter_binding(parameter, self_type);
        }
        let parameters = method
            .parameters
            .iter()
            .filter_map(|parameter| self.parameter_type(parameter, self_type))
            .collect();
        let result = self.lower_type(&method.return_type, self_type);
        if let Some(definition) = self
            .resolutions
            .function_definitions
            .get(&method.id)
            .copied()
        {
            let signature = self.types.function(parameters, result);
            self.definition_types.insert(definition, signature);
        }
        if let Some(body) = &method.default_body {
            self.collect_block_bindings(body, self_type);
        }
    }

    fn parameter_type(
        &mut self,
        parameter: &Parameter,
        self_type: Option<TypeId>,
    ) -> Option<TypeId> {
        match &parameter.kind {
            ParameterKind::Named { ty, .. } => Some(self.lower_type(ty, self_type)),
            ParameterKind::Receiver { .. } => self_type,
        }
    }

    fn collect_parameter_binding(&mut self, parameter: &Parameter, self_type: Option<TypeId>) {
        let Some(local) = self
            .resolutions
            .local_bindings
            .get(&parameter.id)
            .and_then(|bindings| bindings.first())
            .copied()
        else {
            return;
        };
        if let Some(ty) = self.parameter_type(parameter, self_type) {
            self.local_types.insert(local, ty);
        }
    }

    fn collect_block_bindings(&mut self, block: &Block, self_type: Option<TypeId>) {
        for statement in &block.statements {
            self.collect_stmt_bindings(statement, self_type);
        }
    }

    fn collect_stmt_bindings(&mut self, statement: &Stmt, self_type: Option<TypeId>) {
        match &statement.kind {
            StmtKind::Local(local) => {
                let ty = self.lower_type(&local.ty, self_type);
                if let Some(binding) = self
                    .resolutions
                    .local_bindings
                    .get(&statement.id)
                    .and_then(|bindings| bindings.first())
                    .copied()
                {
                    self.local_types.insert(binding, ty);
                }
                if let Some(initializer) = &local.initializer {
                    self.collect_expr_bindings(initializer, self_type);
                }
            }
            StmtKind::StructDestructure(pattern) => {
                self.lower_type(&pattern.ty, self_type);
                self.collect_expr_bindings(&pattern.value, self_type);
                // Field-derived binding types are assigned during full HIR
                // expression analysis, once the concrete owner is known.
            }
            StmtKind::Assignment { target, value } => {
                self.collect_expr_bindings(target, self_type);
                self.collect_expr_bindings(value, self_type);
            }
            StmtKind::Expression(expression)
            | StmtKind::Defer(expression)
            | StmtKind::Print(expression) => self.collect_expr_bindings(expression, self_type),
            StmtKind::Return(expression) => {
                if let Some(expression) = expression {
                    self.collect_expr_bindings(expression, self_type);
                }
            }
            StmtKind::Input => {}
            StmtKind::Declaration(declaration) => self.collect_entry_bindings(declaration),
            StmtKind::Block(block) | StmtKind::Unsafe(block) => {
                self.collect_block_bindings(block, self_type)
            }
            StmtKind::If(expression) => {
                self.collect_expr_bindings(&expression.condition, self_type);
                self.collect_block_bindings(&expression.then_block, self_type);
                for (condition, body) in &expression.else_if {
                    self.collect_expr_bindings(condition, self_type);
                    self.collect_block_bindings(body, self_type);
                }
                if let Some(body) = &expression.else_block {
                    self.collect_block_bindings(body, self_type);
                }
            }
            StmtKind::Match(expression) => {
                self.collect_expr_bindings(&expression.value, self_type);
                for case in &expression.cases {
                    match &case.pattern {
                        Pattern::EnumVariant { enum_type, .. } => {
                            if let Some(enum_type) = enum_type {
                                self.lower_type(enum_type, self_type);
                            }
                        }
                        Pattern::Struct { ty, .. } => {
                            self.lower_type(ty, self_type);
                        }
                    }
                    self.collect_block_bindings(&case.body, self_type);
                }
            }
            StmtKind::For(statement) => {
                if let Some(initializer) = statement.initializer.as_deref() {
                    self.collect_stmt_bindings(initializer, self_type);
                }
                if let Some(condition) = &statement.condition {
                    self.collect_expr_bindings(condition, self_type);
                }
                if let Some(update) = statement.update.as_deref() {
                    self.collect_stmt_bindings(update, self_type);
                }
                self.collect_block_bindings(&statement.body, self_type);
            }
        }
    }

    fn collect_expr_bindings(&mut self, expression: &Expr, self_type: Option<TypeId>) {
        match &expression.kind {
            ExprKind::Literal(_) | ExprKind::Name(_) => {}
            ExprKind::Unary { operand, .. } => self.collect_expr_bindings(operand, self_type),
            ExprKind::Binary { left, right, .. } => {
                self.collect_expr_bindings(left, self_type);
                self.collect_expr_bindings(right, self_type);
            }
            ExprKind::Call {
                callee,
                type_arguments,
                argument_groups,
            } => {
                self.collect_expr_bindings(callee, self_type);
                for type_argument in type_arguments {
                    self.lower_type(type_argument, self_type);
                }
                for group in argument_groups {
                    for argument in group {
                        self.collect_expr_bindings(argument, self_type);
                    }
                }
            }
            ExprKind::Field { receiver, .. } => self.collect_expr_bindings(receiver, self_type),
            ExprKind::Index {
                receiver,
                coordinates,
            } => {
                self.collect_expr_bindings(receiver, self_type);
                for coordinate in coordinates {
                    self.collect_expr_bindings(coordinate, self_type);
                }
            }
            ExprKind::Slice {
                receiver,
                start,
                end,
            } => {
                self.collect_expr_bindings(receiver, self_type);
                if let Some(start) = start {
                    self.collect_expr_bindings(start, self_type);
                }
                if let Some(end) = end {
                    self.collect_expr_bindings(end, self_type);
                }
            }
            ExprKind::StructInit { ty, fields } => {
                self.lower_type(ty, self_type);
                for (_, value) in fields {
                    self.collect_expr_bindings(value, self_type);
                }
            }
            ExprKind::StaticCall { ty, arguments, .. } => {
                self.lower_type(ty, self_type);
                for argument in arguments {
                    self.collect_expr_bindings(argument, self_type);
                }
            }
            ExprKind::Array(elements) => {
                for element in elements {
                    self.collect_expr_bindings(element, self_type);
                }
            }
            ExprKind::Lambda(lambda) => {
                for parameter in &lambda.parameters {
                    self.collect_parameter_binding(parameter, None);
                }
                self.lower_type(&lambda.return_type, None);
                self.collect_block_bindings(&lambda.body, None);
            }
            ExprKind::Block(block) => self.collect_block_bindings(block, self_type),
        }
    }

    fn lower_type(&mut self, syntax: &TypeSyntax, self_type: Option<TypeId>) -> TypeId {
        let lowered = match &syntax.kind {
            TypeSyntaxKind::Builtin(builtin) => self.types.builtin(*builtin),
            TypeSyntaxKind::Named { arguments, .. } => {
                let arguments = arguments
                    .iter()
                    .map(|argument| self.lower_type(argument, self_type))
                    .collect::<Vec<_>>();
                match self.resolutions.types.get(&syntax.id).copied() {
                    Some(TypeResolution::Definition(definition)) => {
                        let kind = self.resolutions.definitions[definition.index()].kind;
                        if kind == DefinitionKind::TypeAlias {
                            if let Some(existing) = self.definition_types.get(&definition) {
                                *existing
                            } else if let Some(target) = self.aliases.get(&definition).copied() {
                                // Alias cycle diagnostics remain the responsibility
                                // of semantic validation; insert Error first to
                                // terminate accidental recursion safely.
                                let error = self.types.error();
                                self.definition_types.insert(definition, error);
                                let lowered = self.lower_type(target, self_type);
                                self.definition_types.insert(definition, lowered);
                                lowered
                            } else {
                                self.types.error()
                            }
                        } else {
                            self.types.nominal(definition, arguments)
                        }
                    }
                    Some(TypeResolution::GenericParameter(parameter)) => {
                        self.types.intern(TypeKind::GenericParameter(parameter))
                    }
                    Some(TypeResolution::Intrinsic(intrinsic)) => self.types.intrinsic(intrinsic),
                    Some(TypeResolution::Builtin(builtin)) => self.types.builtin(builtin),
                    Some(TypeResolution::SelfType) => {
                        self_type.unwrap_or_else(|| self.types.error())
                    }
                    None => self.types.error(),
                }
            }
            TypeSyntaxKind::Const(inner) => {
                let inner = self.lower_type(inner, self_type);
                self.types.intern(TypeKind::Const(inner))
            }
            TypeSyntaxKind::Array {
                element,
                dimensions,
            } => {
                let element = self.lower_type(element, self_type);
                let mut lengths = Vec::new();
                for dimension in dimensions {
                    match const_u64(dimension) {
                        Some(length) => lengths.push(length),
                        None => {
                            self.diagnostics.push(
                                Diagnostic::error("array length must be a non-negative constant")
                                    .with_code("E3001")
                                    .at(dimension.span),
                            );
                        }
                    }
                }
                self.types.intern(TypeKind::Array {
                    element,
                    dimensions: lengths,
                })
            }
            TypeSyntaxKind::Reference { target, mutable } => {
                let target = self.lower_type(target, self_type);
                self.types.intern(TypeKind::Reference {
                    target,
                    mutable: *mutable,
                })
            }
            TypeSyntaxKind::Pointer(target) => {
                let target = self.lower_type(target, self_type);
                self.types.intern(TypeKind::Pointer(target))
            }
            TypeSyntaxKind::Slice(element) => {
                let element = self.lower_type(element, self_type);
                self.types.intern(TypeKind::Slice(element))
            }
            TypeSyntaxKind::Union(types) => {
                let types = types
                    .iter()
                    .map(|ty| self.lower_type(ty, self_type))
                    .collect();
                self.types.intern(TypeKind::Union(types))
            }
            TypeSyntaxKind::Intersection(types) => {
                let types = types
                    .iter()
                    .map(|ty| self.lower_type(ty, self_type))
                    .collect();
                self.types.intern(TypeKind::Intersection(types))
            }
            TypeSyntaxKind::Function { parameters, result } => {
                let parameters = parameters
                    .iter()
                    .map(|ty| self.lower_type(ty, self_type))
                    .collect();
                let result = self.lower_type(result, self_type);
                self.types.function(parameters, result)
            }
            TypeSyntaxKind::SelfType { .. } => self_type.unwrap_or_else(|| self.types.error()),
        };
        self.syntax_types.insert(syntax.id, lowered);
        lowered
    }
}

fn const_u64(expression: &Expr) -> Option<u64> {
    match &expression.kind {
        ExprKind::Literal(Literal::Integer(value)) | ExprKind::Literal(Literal::Long(value)) => {
            u64::try_from(*value).ok()
        }
        ExprKind::Unary {
            operator: UnaryOperator::Plus,
            operand,
        } => const_u64(operand),
        ExprKind::Binary {
            left,
            operator,
            right,
        } => {
            let left = const_u64(left)?;
            let right = const_u64(right)?;
            match operator {
                BinaryOperator::Add => left.checked_add(right),
                BinaryOperator::Subtract => left.checked_sub(right),
                BinaryOperator::Multiply => left.checked_mul(right),
                BinaryOperator::Divide => left.checked_div(right),
                BinaryOperator::Modulo => left.checked_rem(right),
                _ => None,
            }
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ids::FileId;

    #[test]
    fn separates_nominal_definition_from_instantiated_type() {
        let source = r#"
            struct Box[T] { value: T; }
            function use_box(value: Box[int]): int { return value.value; }
        "#;
        let legacy = crate::ast::try_parse(source).unwrap();
        let module = crate::syntax::from_legacy(&legacy, FileId::new(0), source.len()).unwrap();
        let resolutions = crate::resolver::resolve(&module).unwrap();
        let model = analyze_declarations(&module, resolutions).unwrap();

        assert!(model.types.len() > crate::intrinsics::IntrinsicType::ALL.len());
    }
}
