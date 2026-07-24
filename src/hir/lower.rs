//! Lowers resolved, semantically typed syntax into HIR.

use crate::analysis::model::SemanticModel;
use crate::analysis::resolver::{TypeResolution, ValueResolution};
use crate::analysis::types::TypeKind;
use crate::diagnostic::Diagnostic;
use crate::hir;
use crate::ids::{DefId, FieldId, LocalId, TypeId, VariantId};
use crate::intrinsics::IntrinsicType;
use crate::syntax::ast as syntax;
use std::collections::HashMap;

pub fn lower(
    module: &syntax::Module,
    model: &mut SemanticModel,
) -> Result<hir::Module, Vec<Diagnostic>> {
    let mut lowerer = Lowerer::new(module, model);
    let module = lowerer.lower_module();
    if lowerer.diagnostics.is_empty() {
        Ok(module)
    } else {
        Err(lowerer.diagnostics)
    }
}

struct Lowerer<'a> {
    module: &'a syntax::Module,
    model: &'a mut SemanticModel,
    fields: HashMap<DefId, HashMap<String, FieldId>>,
    variants: HashMap<DefId, HashMap<String, VariantId>>,
    methods: HashMap<DefId, HashMap<String, DefId>>,
    dynamic_methods: HashMap<DefId, HashMap<String, (DefId, TypeId)>>,
    trait_supers: HashMap<DefId, Vec<DefId>>,
    diagnostics: Vec<Diagnostic>,
}

impl<'a> Lowerer<'a> {
    fn new(module: &'a syntax::Module, model: &'a mut SemanticModel) -> Self {
        let mut fields = HashMap::new();
        let mut variants = HashMap::new();
        let mut methods = HashMap::new();
        let mut dynamic_methods = HashMap::new();
        let mut trait_supers = HashMap::new();

        for entry in &module.entries {
            let owner = model.resolutions.item_definitions.get(&entry.id).copied();
            match (&entry.kind, owner) {
                (syntax::TopLevelKind::Struct(declaration), Some(owner)) => {
                    let owner_fields = fields.entry(owner).or_insert_with(HashMap::new);
                    for field in &declaration.fields {
                        if let Some(id) =
                            model.resolutions.field_definitions.get(&field.id).copied()
                        {
                            owner_fields.insert(field.name.clone(), id);
                        }
                    }
                    index_methods(&mut methods, model, owner, &declaration.methods);
                }
                (syntax::TopLevelKind::Enum(declaration), Some(owner)) => {
                    let owner_variants = variants.entry(owner).or_insert_with(HashMap::new);
                    for variant in &declaration.variants {
                        if let Some(id) = model
                            .resolutions
                            .variant_definitions
                            .get(&variant.id)
                            .copied()
                        {
                            owner_variants.insert(variant.name.clone(), id);
                        }
                    }
                    index_methods(&mut methods, model, owner, &declaration.methods);
                }
                (syntax::TopLevelKind::Trait(declaration), Some(owner)) => {
                    trait_supers.insert(
                        owner,
                        model
                            .resolutions
                            .supertrait_definitions
                            .get(&entry.id)
                            .cloned()
                            .unwrap_or_default(),
                    );
                    let owner_methods = dynamic_methods.entry(owner).or_insert_with(HashMap::new);
                    for method in &declaration.methods {
                        if let (Some(definition), Some(result)) = (
                            model
                                .resolutions
                                .function_definitions
                                .get(&method.id)
                                .copied(),
                            model.syntax_types.get(&method.return_type.id).copied(),
                        ) {
                            owner_methods.insert(method.name.clone(), (definition, result));
                        }
                    }
                }
                _ => {}
            }
        }

        Self {
            module,
            model,
            fields,
            variants,
            methods,
            dynamic_methods,
            trait_supers,
            diagnostics: Vec::new(),
        }
    }

    fn lower_module(&mut self) -> hir::Module {
        let mut items = Vec::new();
        for entry in &self.module.entries {
            if let Some(item) = self.lower_item(entry) {
                items.push(item);
            }
        }
        hir::Module { items }
    }

    fn lower_item(&mut self, entry: &syntax::TopLevel) -> Option<hir::Item> {
        let definition = self
            .model
            .resolutions
            .item_definitions
            .get(&entry.id)
            .copied();
        let kind = match &entry.kind {
            syntax::TopLevelKind::Import(_) | syntax::TopLevelKind::TypeAlias(_) => return None,
            syntax::TopLevelKind::Struct(declaration) => {
                let owner_type = definition
                    .and_then(|definition| self.model.definition_types.get(&definition).copied());
                let fields = declaration
                    .fields
                    .iter()
                    .filter_map(|field| {
                        let id = self
                            .model
                            .resolutions
                            .field_definitions
                            .get(&field.id)
                            .copied()?;
                        let ty = self.model.field_types.get(&id).copied()?;
                        Some(hir::Field {
                            id,
                            name: field.name.clone(),
                            ty,
                            is_const: field.is_const,
                        })
                    })
                    .collect();
                let methods = declaration
                    .methods
                    .iter()
                    .map(|method| self.lower_function(method, owner_type, None))
                    .collect();
                hir::ItemKind::Struct(hir::Struct { fields, methods })
            }
            syntax::TopLevelKind::Enum(declaration) => {
                let owner_type = definition
                    .and_then(|definition| self.model.definition_types.get(&definition).copied());
                let variants = declaration
                    .variants
                    .iter()
                    .filter_map(|variant| {
                        let id = self
                            .model
                            .resolutions
                            .variant_definitions
                            .get(&variant.id)
                            .copied()?;
                        Some(hir::Variant {
                            id,
                            name: variant.name.clone(),
                            payload: self
                                .model
                                .variant_payloads
                                .get(&id)
                                .cloned()
                                .unwrap_or_default(),
                        })
                    })
                    .collect();
                let methods = declaration
                    .methods
                    .iter()
                    .map(|method| self.lower_function(method, owner_type, None))
                    .collect();
                hir::ItemKind::Enum(hir::Enum { variants, methods })
            }
            syntax::TopLevelKind::Trait(declaration) => {
                let owner_type = definition
                    .and_then(|definition| self.model.definition_types.get(&definition).copied());
                let supertraits = self
                    .model
                    .resolutions
                    .supertrait_definitions
                    .get(&entry.id)
                    .cloned()
                    .unwrap_or_default();
                let methods = declaration
                    .methods
                    .iter()
                    .map(|method| self.lower_trait_method(method, owner_type))
                    .collect();
                hir::ItemKind::Trait(hir::Trait {
                    supertraits,
                    methods,
                })
            }
            syntax::TopLevelKind::Shape(declaration) => {
                let methods = declaration
                    .methods
                    .iter()
                    .map(|method| self.lower_trait_method(method, None))
                    .collect();
                hir::ItemKind::Shape(hir::Trait {
                    supertraits: Vec::new(),
                    methods,
                })
            }
            syntax::TopLevelKind::Implementation(declaration) => {
                let traits = declaration
                    .traits
                    .iter()
                    .map(|ty| self.syntax_type(ty))
                    .collect();
                let target = self.syntax_type(&declaration.target);
                hir::ItemKind::Implementation { traits, target }
            }
            syntax::TopLevelKind::Attach(_) | syntax::TopLevelKind::Conformance(_) => {
                // Normalized syntax has already merged behavior methods and
                // retained conformances as explicit implementation records.
                return None;
            }
            syntax::TopLevelKind::Function(function) => {
                hir::ItemKind::Function(self.lower_function(function, None, definition))
            }
            syntax::TopLevelKind::ExternFunction(function) => {
                hir::ItemKind::ExternFunction(hir::FunctionSignature {
                    parameters: function
                        .parameters
                        .iter()
                        .filter_map(|parameter| self.parameter_type(parameter))
                        .collect(),
                    result: self.syntax_type(&function.return_type),
                })
            }
            syntax::TopLevelKind::Global(global) => {
                let ty = self.syntax_type(&global.ty);
                let initializer = global
                    .initializer
                    .as_ref()
                    .map(|expression| self.lower_expr(expression, Some(ty)));
                hir::ItemKind::Global(hir::Global {
                    ty,
                    is_const: global.is_const,
                    initializer,
                })
            }
            syntax::TopLevelKind::Test(test) => {
                hir::ItemKind::Test(self.lower_block(&test.body, None))
            }
            syntax::TopLevelKind::Statement(statement) => {
                hir::ItemKind::Statement(self.lower_stmt(statement, None).0)
            }
        };
        Some(hir::Item {
            source: entry.id,
            span: entry.span,
            definition,
            visibility: entry.visibility,
            kind,
        })
    }

    fn lower_function(
        &mut self,
        function: &syntax::FunctionDecl,
        self_type: Option<TypeId>,
        top_level_definition: Option<DefId>,
    ) -> hir::Function {
        let definition = top_level_definition.or_else(|| {
            self.model
                .resolutions
                .function_definitions
                .get(&function.body.id)
                .copied()
        });
        let parameters = function
            .parameters
            .iter()
            .filter_map(|parameter| {
                let local = self
                    .model
                    .resolutions
                    .local_bindings
                    .get(&parameter.id)
                    .and_then(|bindings| bindings.first())
                    .copied()?;
                let ty = self
                    .model
                    .local_types
                    .get(&local)
                    .copied()
                    .or(self_type)
                    .unwrap_or_else(|| self.model.types.error());
                let is_const = match parameter.kind {
                    syntax::ParameterKind::Named { is_const, .. }
                    | syntax::ParameterKind::Receiver { is_const, .. } => is_const,
                };
                Some(hir::Parameter {
                    local,
                    ty,
                    is_const,
                })
            })
            .collect();
        let result = self.syntax_type(&function.return_type);
        let body = self.lower_block(&function.body, Some(result));
        hir::Function {
            definition,
            parameters,
            result,
            body,
        }
    }

    fn lower_trait_method(
        &mut self,
        method: &syntax::TraitMethod,
        _self_type: Option<TypeId>,
    ) -> hir::TraitMethod {
        let definition = self
            .model
            .resolutions
            .function_definitions
            .get(&method.id)
            .copied()
            .unwrap_or_else(|| DefId::new(u32::MAX));
        let receiver = method.parameters.iter().find_map(|parameter| {
            let syntax::ParameterKind::Receiver { mutable, is_const } = parameter.kind else {
                return None;
            };
            Some(hir::Receiver { mutable, is_const })
        });
        let parameters = method
            .parameters
            .iter()
            .filter_map(|parameter| match parameter.kind {
                syntax::ParameterKind::Named { .. } => self.parameter_type(parameter),
                syntax::ParameterKind::Receiver { .. } => None,
            })
            .collect();
        let result = self.syntax_type(&method.return_type);
        let default_body = method
            .default_body
            .as_ref()
            .map(|body| self.lower_block(body, Some(result)));
        hir::TraitMethod {
            definition,
            name: method.name.clone(),
            receiver,
            parameters,
            result,
            default_body,
        }
    }

    fn parameter_type(&self, parameter: &syntax::Parameter) -> Option<TypeId> {
        let local = self
            .model
            .resolutions
            .local_bindings
            .get(&parameter.id)
            .and_then(|bindings| bindings.first())?;
        self.model.local_types.get(local).copied()
    }

    fn lower_block(&mut self, block: &syntax::Block, return_type: Option<TypeId>) -> hir::Block {
        let mut statements = Vec::new();
        let mut flow = hir::Flow::FallsThrough;
        for statement in &block.statements {
            let (statement, statement_flow) = self.lower_stmt(statement, return_type);
            statements.push(statement);
            if flow.continues() && !statement_flow.continues() {
                flow = statement_flow;
            }
        }
        hir::Block {
            source: block.id,
            span: block.span,
            statements,
            flow,
        }
    }

    fn lower_stmt(
        &mut self,
        statement: &syntax::Stmt,
        return_type: Option<TypeId>,
    ) -> (hir::Stmt, hir::Flow) {
        let (kind, flow) = match &statement.kind {
            syntax::StmtKind::Local(local) => {
                let ty = self.syntax_type(&local.ty);
                let local_id = self.binding(statement.id, 0);
                let initializer = local
                    .initializer
                    .as_ref()
                    .map(|expression| self.lower_expr(expression, Some(ty)));
                (
                    hir::StmtKind::Local {
                        local: local_id,
                        ty,
                        is_const: local.is_const,
                        initializer,
                    },
                    hir::Flow::FallsThrough,
                )
            }
            syntax::StmtKind::StructDestructure(pattern) => {
                let expected = self.syntax_type(&pattern.ty);
                let value = self.lower_expr(&pattern.value, Some(expected));
                let owner = self.nominal_definition(expected);
                let mut bindings = Vec::new();
                for (index, field) in pattern.fields.iter().enumerate() {
                    let field_id = owner
                        .and_then(|owner| self.fields.get(&owner))
                        .and_then(|fields| fields.get(&field.name))
                        .copied()
                        .unwrap_or_else(|| self.missing_field(statement.span, &field.name));
                    let local = self.binding(statement.id, index);
                    if let Some(field_type) = self.model.field_types.get(&field_id).copied() {
                        self.model.local_types.insert(local, field_type);
                    }
                    bindings.push((field_id, local));
                }
                (
                    hir::StmtKind::Destructure { value, bindings },
                    hir::Flow::FallsThrough,
                )
            }
            syntax::StmtKind::Assignment { target, value } => {
                let target = self.lower_expr(target, None);
                let value = self.lower_expr(value, Some(target.ty));
                (
                    hir::StmtKind::Assignment { target, value },
                    hir::Flow::FallsThrough,
                )
            }
            syntax::StmtKind::Expression(expression) => (
                hir::StmtKind::Expression(self.lower_expr(expression, None)),
                hir::Flow::FallsThrough,
            ),
            syntax::StmtKind::Return(expression) => (
                hir::StmtKind::Return(
                    expression
                        .as_ref()
                        .map(|expression| self.lower_expr(expression, return_type)),
                ),
                hir::Flow::Returns,
            ),
            syntax::StmtKind::Defer(expression) => (
                hir::StmtKind::Defer(self.lower_expr(expression, None)),
                hir::Flow::FallsThrough,
            ),
            syntax::StmtKind::Print(expression) => (
                hir::StmtKind::Print(self.lower_expr(expression, None)),
                hir::Flow::FallsThrough,
            ),
            syntax::StmtKind::Input => (hir::StmtKind::Input, hir::Flow::FallsThrough),
            syntax::StmtKind::Declaration(_) => {
                // Nested named declarations are excluded from prepared backend
                // input. Preserve a no-op expression if one appears.
                let void = self.model.types.builtin(syntax::BuiltinType::Void);
                let expression = hir::Expr {
                    source: statement.id,
                    span: statement.span,
                    ty: void,
                    kind: hir::ExprKind::Block(hir::Block {
                        source: statement.id,
                        span: statement.span,
                        statements: Vec::new(),
                        flow: hir::Flow::FallsThrough,
                    }),
                };
                (
                    hir::StmtKind::Expression(expression),
                    hir::Flow::FallsThrough,
                )
            }
            syntax::StmtKind::Block(block) => {
                let block = self.lower_block(block, return_type);
                let flow = block.flow;
                (hir::StmtKind::Block(block), flow)
            }
            syntax::StmtKind::Unsafe(block) => {
                let block = self.lower_block(block, return_type);
                let flow = block.flow;
                (hir::StmtKind::Unsafe(block), flow)
            }
            syntax::StmtKind::If(expression) => {
                let boolean = self.model.types.builtin(syntax::BuiltinType::Boolean);
                let condition = self.lower_expr(&expression.condition, Some(boolean));
                let then_block = self.lower_block(&expression.then_block, return_type);
                let else_if = expression
                    .else_if
                    .iter()
                    .map(|(condition, body)| {
                        (
                            self.lower_expr(condition, Some(boolean)),
                            self.lower_block(body, return_type),
                        )
                    })
                    .collect::<Vec<_>>();
                let else_block = expression
                    .else_block
                    .as_ref()
                    .map(|body| self.lower_block(body, return_type));
                let all_branches = else_block.is_some()
                    && !then_block.flow.continues()
                    && else_if.iter().all(|(_, block)| !block.flow.continues())
                    && else_block
                        .as_ref()
                        .is_some_and(|block| !block.flow.continues());
                let flow = if all_branches {
                    if then_block.flow == hir::Flow::Returns
                        && else_if
                            .iter()
                            .all(|(_, block)| block.flow == hir::Flow::Returns)
                        && else_block
                            .as_ref()
                            .is_some_and(|block| block.flow == hir::Flow::Returns)
                    {
                        hir::Flow::Returns
                    } else {
                        hir::Flow::Diverges
                    }
                } else {
                    hir::Flow::FallsThrough
                };
                (
                    hir::StmtKind::If(hir::If {
                        condition,
                        then_block,
                        else_if,
                        else_block,
                        flow,
                    }),
                    flow,
                )
            }
            syntax::StmtKind::Match(expression) => {
                let value = self.lower_expr(&expression.value, None);
                let mut cases = Vec::new();
                for case in &expression.cases {
                    let pattern = self.lower_pattern(case, value.ty);
                    self.assign_pattern_types(case, value.ty);
                    cases.push(hir::MatchCase {
                        pattern,
                        bindings: self
                            .model
                            .resolutions
                            .local_bindings
                            .get(&case.id)
                            .cloned()
                            .unwrap_or_default(),
                        body: self.lower_block(&case.body, return_type),
                    });
                }
                let flow =
                    if !cases.is_empty() && cases.iter().all(|case| !case.body.flow.continues()) {
                        if cases
                            .iter()
                            .all(|case| case.body.flow == hir::Flow::Returns)
                        {
                            hir::Flow::Returns
                        } else {
                            hir::Flow::Diverges
                        }
                    } else {
                        hir::Flow::FallsThrough
                    };
                (
                    hir::StmtKind::Match(hir::Match { value, cases, flow }),
                    flow,
                )
            }
            syntax::StmtKind::For(loop_statement) => {
                let initializer = loop_statement
                    .initializer
                    .as_deref()
                    .map(|statement| Box::new(self.lower_stmt(statement, return_type).0));
                let boolean = self.model.types.builtin(syntax::BuiltinType::Boolean);
                let condition = loop_statement
                    .condition
                    .as_ref()
                    .map(|condition| self.lower_expr(condition, Some(boolean)));
                let update = loop_statement
                    .update
                    .as_deref()
                    .map(|statement| Box::new(self.lower_stmt(statement, return_type).0));
                let body = self.lower_block(&loop_statement.body, return_type);
                let flow = hir::Flow::FallsThrough;
                (
                    hir::StmtKind::For(hir::For {
                        initializer,
                        condition,
                        update,
                        body,
                        flow,
                    }),
                    flow,
                )
            }
        };
        (
            hir::Stmt {
                source: statement.id,
                span: statement.span,
                kind,
            },
            flow,
        )
    }

    fn lower_expr(&mut self, expression: &syntax::Expr, expected: Option<TypeId>) -> hir::Expr {
        let (kind, ty) = match &expression.kind {
            syntax::ExprKind::Literal(literal) => {
                let ty = self.literal_type(literal, expected);
                (hir::ExprKind::Literal(literal.clone()), ty)
            }
            syntax::ExprKind::Name(_) => {
                let resolution = self.model.resolutions.values.get(&expression.id).copied();
                match resolution {
                    Some(ValueResolution::Definition(definition)) => (
                        hir::ExprKind::Value(hir::Value::Definition(definition)),
                        self.model
                            .definition_types
                            .get(&definition)
                            .copied()
                            .unwrap_or_else(|| self.model.types.error()),
                    ),
                    Some(ValueResolution::Local(local)) => (
                        hir::ExprKind::Value(hir::Value::Local(local)),
                        self.model
                            .local_types
                            .get(&local)
                            .copied()
                            .unwrap_or_else(|| self.model.types.error()),
                    ),
                    None => (
                        hir::ExprKind::Value(hir::Value::Local(LocalId::new(u32::MAX))),
                        self.model.types.error(),
                    ),
                }
            }
            syntax::ExprKind::Unary { operator, operand } => {
                let operand = self.lower_expr(operand, expected);
                let ty = match operator {
                    syntax::UnaryOperator::Plus | syntax::UnaryOperator::Minus => operand.ty,
                    syntax::UnaryOperator::Not => {
                        self.model.types.builtin(syntax::BuiltinType::Boolean)
                    }
                    syntax::UnaryOperator::AddressOf | syntax::UnaryOperator::AddressOfMut => {
                        if let Some(expected) = expected {
                            if matches!(self.model.types.kind(expected), TypeKind::Pointer(_)) {
                                expected
                            } else {
                                self.model.types.intern(TypeKind::Reference {
                                    target: operand.ty,
                                    mutable: matches!(
                                        operator,
                                        syntax::UnaryOperator::AddressOfMut
                                    ),
                                })
                            }
                        } else {
                            self.model.types.intern(TypeKind::Reference {
                                target: operand.ty,
                                mutable: matches!(operator, syntax::UnaryOperator::AddressOfMut),
                            })
                        }
                    }
                    syntax::UnaryOperator::Dereference => {
                        match self.model.types.kind(operand.ty).clone() {
                            TypeKind::Pointer(target) | TypeKind::Reference { target, .. } => {
                                target
                            }
                            _ => self.model.types.error(),
                        }
                    }
                };
                (
                    hir::ExprKind::Unary {
                        operator: *operator,
                        operand: Box::new(operand),
                    },
                    ty,
                )
            }
            syntax::ExprKind::Binary {
                left,
                operator,
                right,
            } => {
                let left = self.lower_expr(left, expected);
                let right = self.lower_expr(right, Some(left.ty));
                let ty = self.binary_result(*operator, left.ty, right.ty, expected);
                (
                    hir::ExprKind::Binary {
                        left: Box::new(left),
                        operator: *operator,
                        right: Box::new(right),
                    },
                    ty,
                )
            }
            syntax::ExprKind::Call {
                callee,
                argument_groups,
                ..
            } => {
                if let syntax::ExprKind::Field { receiver, name } = &callee.kind {
                    return self.lower_method_call(
                        expression,
                        receiver,
                        name,
                        argument_groups,
                        expected,
                    );
                }
                let callee = self.lower_expr(callee, None);
                let (argument_groups, result) =
                    self.lower_call_groups(callee.ty, argument_groups, expression.span);
                (
                    hir::ExprKind::Call {
                        callee: Box::new(callee),
                        argument_groups,
                    },
                    result,
                )
            }
            syntax::ExprKind::Field { receiver, name } => {
                let receiver = self.lower_expr(receiver, None);
                if name == "len"
                    && matches!(
                        self.model.types.kind(receiver.ty),
                        TypeKind::Array { .. } | TypeKind::Slice(_)
                    )
                {
                    let ty = self.model.types.builtin(syntax::BuiltinType::Int);
                    (
                        hir::ExprKind::Length {
                            receiver: Box::new(receiver),
                        },
                        ty,
                    )
                } else {
                    let field = self
                        .nominal_definition(receiver.ty)
                        .and_then(|owner| self.fields.get(&owner))
                        .and_then(|fields| fields.get(name))
                        .copied()
                        .unwrap_or_else(|| self.missing_field(expression.span, name));
                    let ty = self
                        .model
                        .field_types
                        .get(&field)
                        .copied()
                        .unwrap_or_else(|| self.model.types.error());
                    (
                        hir::ExprKind::Field {
                            receiver: Box::new(receiver),
                            field,
                        },
                        ty,
                    )
                }
            }
            syntax::ExprKind::Index {
                receiver,
                coordinates,
            } => {
                let receiver = self.lower_expr(receiver, None);
                let integer = self.model.types.builtin(syntax::BuiltinType::Int);
                let coordinates = coordinates
                    .iter()
                    .map(|coordinate| self.lower_expr(coordinate, Some(integer)))
                    .collect::<Vec<_>>();
                let ty = self.index_result(receiver.ty, coordinates.len());
                (
                    hir::ExprKind::Index {
                        receiver: Box::new(receiver),
                        coordinates,
                    },
                    ty,
                )
            }
            syntax::ExprKind::Slice {
                receiver,
                start,
                end,
            } => {
                let receiver = self.lower_expr(receiver, None);
                let integer = self.model.types.builtin(syntax::BuiltinType::Int);
                let start = start
                    .as_deref()
                    .map(|value| Box::new(self.lower_expr(value, Some(integer))));
                let end = end
                    .as_deref()
                    .map(|value| Box::new(self.lower_expr(value, Some(integer))));
                let ty = match self.model.types.kind(receiver.ty).clone() {
                    TypeKind::Array { element, .. } | TypeKind::Slice(element) => {
                        self.model.types.intern(TypeKind::Slice(element))
                    }
                    _ => self.model.types.error(),
                };
                (
                    hir::ExprKind::Slice {
                        receiver: Box::new(receiver),
                        start,
                        end,
                    },
                    ty,
                )
            }
            syntax::ExprKind::StructInit { ty, fields } => {
                let ty = self.syntax_type(ty);
                let definition = self
                    .nominal_definition(ty)
                    .unwrap_or_else(|| DefId::new(u32::MAX));
                let fields = fields
                    .iter()
                    .map(|(name, value)| {
                        let field = self
                            .fields
                            .get(&definition)
                            .and_then(|fields| fields.get(name))
                            .copied()
                            .unwrap_or_else(|| self.missing_field(expression.span, name));
                        let expected = self.model.field_types.get(&field).copied();
                        (field, self.lower_expr(value, expected))
                    })
                    .collect();
                (hir::ExprKind::StructInit { definition, fields }, ty)
            }
            syntax::ExprKind::StaticCall {
                ty,
                name,
                arguments,
            } => {
                let owner_type = self.syntax_type(ty);
                let target = self.static_target(ty, owner_type, name);
                let result = self.static_result(owner_type, name);
                let arguments = arguments
                    .iter()
                    .map(|argument| self.lower_expr(argument, None))
                    .collect();
                (hir::ExprKind::StaticCall { target, arguments }, result)
            }
            syntax::ExprKind::Array(elements) => {
                let expected_element =
                    expected.and_then(|expected| match self.model.types.kind(expected).clone() {
                        TypeKind::Array {
                            element,
                            dimensions,
                        } if dimensions.len() > 1 => {
                            Some(self.model.types.intern(TypeKind::Array {
                                element,
                                dimensions: dimensions[1..].to_vec(),
                            }))
                        }
                        TypeKind::Array { element, .. } | TypeKind::Slice(element) => Some(element),
                        _ => None,
                    });
                let elements = elements
                    .iter()
                    .map(|element| self.lower_expr(element, expected_element))
                    .collect::<Vec<_>>();
                let ty = expected.unwrap_or_else(|| {
                    let element = elements
                        .first()
                        .map(|element| element.ty)
                        .unwrap_or_else(|| self.model.types.error());
                    self.model.types.intern(TypeKind::Array {
                        element,
                        dimensions: vec![elements.len() as u64],
                    })
                });
                (hir::ExprKind::Array(elements), ty)
            }
            syntax::ExprKind::Lambda(lambda) => {
                let parameters = lambda
                    .parameters
                    .iter()
                    .filter_map(|parameter| self.parameter_type(parameter))
                    .collect::<Vec<_>>();
                let result = self.syntax_type(&lambda.return_type);
                let function_type = self.model.types.function(parameters, result);
                let function = hir::Function {
                    definition: None,
                    parameters: lambda
                        .parameters
                        .iter()
                        .filter_map(|parameter| {
                            let local = self
                                .model
                                .resolutions
                                .local_bindings
                                .get(&parameter.id)?
                                .first()
                                .copied()?;
                            let ty = self.model.local_types.get(&local).copied()?;
                            let is_const = match parameter.kind {
                                syntax::ParameterKind::Named { is_const, .. }
                                | syntax::ParameterKind::Receiver { is_const, .. } => is_const,
                            };
                            Some(hir::Parameter {
                                local,
                                ty,
                                is_const,
                            })
                        })
                        .collect(),
                    result,
                    body: self.lower_block(&lambda.body, Some(result)),
                };
                (hir::ExprKind::Lambda(function), function_type)
            }
            syntax::ExprKind::Block(block) => {
                let block = self.lower_block(block, expected);
                let ty = self.model.types.builtin(syntax::BuiltinType::Void);
                (hir::ExprKind::Block(block), ty)
            }
        };

        if ty == self.model.types.error() {
            self.diagnostics.push(
                Diagnostic::error("could not determine expression type while lowering HIR")
                    .with_code("E3900")
                    .at(expression.span),
            );
        }
        hir::Expr {
            source: expression.id,
            span: expression.span,
            ty,
            kind,
        }
    }

    fn lower_method_call(
        &mut self,
        expression: &syntax::Expr,
        receiver: &syntax::Expr,
        name: &str,
        groups: &[Vec<syntax::Expr>],
        _expected: Option<TypeId>,
    ) -> hir::Expr {
        let receiver = self.lower_expr(receiver, None);
        let owner = self.nominal_definition(receiver.ty);
        let definition = owner
            .and_then(|owner| self.methods.get(&owner))
            .and_then(|methods| methods.get(name))
            .copied();
        let dynamic = self.dynamic_method(receiver.ty, name);
        let (target, result) = if let Some(definition) = definition {
            (
                hir::MethodTarget::Definition(definition),
                self.model
                    .definition_types
                    .get(&definition)
                    .copied()
                    .and_then(|signature| self.function_result_ignoring_receiver(signature)),
            )
        } else if self.is_intrinsic_receiver(receiver.ty) {
            (
                hir::MethodTarget::Intrinsic {
                    owner: receiver.ty,
                    name: name.to_string(),
                },
                None,
            )
        } else if let Some((method, result)) = dynamic {
            (
                hir::MethodTarget::Dynamic {
                    owner: receiver.ty,
                    method,
                },
                Some(result),
            )
        } else {
            (
                hir::MethodTarget::Intrinsic {
                    owner: receiver.ty,
                    name: name.to_string(),
                },
                None,
            )
        };

        let mut result = result.unwrap_or_else(|| self.intrinsic_method_result(receiver.ty, name));
        let mut lowered_groups = Vec::new();
        for group in groups {
            lowered_groups.push(
                group
                    .iter()
                    .map(|argument| self.lower_expr(argument, None))
                    .collect(),
            );
            if let TypeKind::Function { result: nested, .. } = self.model.types.kind(result).clone()
            {
                result = nested;
            }
        }
        if result == self.model.types.error() {
            self.diagnostics.push(
                Diagnostic::error(format!("could not resolve method `{name}` for typed HIR"))
                    .with_code("E3901")
                    .at(expression.span),
            );
        }
        hir::Expr {
            source: expression.id,
            span: expression.span,
            ty: result,
            kind: hir::ExprKind::MethodCall {
                receiver: Box::new(receiver),
                method: target,
                argument_groups: lowered_groups,
            },
        }
    }

    fn lower_call_groups(
        &mut self,
        mut callee_type: TypeId,
        groups: &[Vec<syntax::Expr>],
        span: crate::source_map::Span,
    ) -> (Vec<Vec<hir::Expr>>, TypeId) {
        let mut lowered = Vec::new();
        for group in groups {
            let (parameters, result) = match self.model.types.kind(callee_type).clone() {
                TypeKind::Function { parameters, result } => (parameters, result),
                _ => {
                    self.diagnostics.push(
                        Diagnostic::error("attempted to lower a call of a non-function value")
                            .with_code("E3902")
                            .at(span),
                    );
                    (Vec::new(), self.model.types.error())
                }
            };
            lowered.push(
                group
                    .iter()
                    .enumerate()
                    .map(|(index, argument)| {
                        self.lower_expr(argument, parameters.get(index).copied())
                    })
                    .collect(),
            );
            callee_type = result;
        }
        (lowered, callee_type)
    }

    fn literal_type(&mut self, literal: &syntax::Literal, expected: Option<TypeId>) -> TypeId {
        match literal {
            syntax::Literal::Integer(_) => expected
                .filter(|ty| self.is_integral(*ty))
                .unwrap_or_else(|| self.model.types.builtin(syntax::BuiltinType::Int)),
            syntax::Literal::Long(_) => self.model.types.builtin(syntax::BuiltinType::Long),
            syntax::Literal::Float(_) => self.model.types.builtin(syntax::BuiltinType::Float),
            syntax::Literal::Double(_) => self.model.types.builtin(syntax::BuiltinType::Double),
            syntax::Literal::String(_) => self.model.types.builtin(syntax::BuiltinType::String),
            syntax::Literal::Boolean(_) => self.model.types.builtin(syntax::BuiltinType::Boolean),
            syntax::Literal::Char(_) => self.model.types.builtin(syntax::BuiltinType::Char),
        }
    }

    fn binary_result(
        &mut self,
        operator: syntax::BinaryOperator,
        left: TypeId,
        right: TypeId,
        expected: Option<TypeId>,
    ) -> TypeId {
        use syntax::BinaryOperator::*;
        match operator {
            Equals | NotEquals | LessThan | GreaterThan | LessThanOrEqual | GreaterThanOrEqual
            | And | Or => self.model.types.builtin(syntax::BuiltinType::Boolean),
            Add => {
                let string = self.model.types.builtin(syntax::BuiltinType::String);
                if left == string && right == string {
                    string
                } else {
                    expected
                        .filter(|ty| self.is_numeric(*ty))
                        .unwrap_or_else(|| self.promote_numeric(left, right))
                }
            }
            Subtract | Multiply | Divide | Modulo | Power => expected
                .filter(|ty| self.is_numeric(*ty))
                .unwrap_or_else(|| self.promote_numeric(left, right)),
        }
    }

    fn promote_numeric(&mut self, left: TypeId, right: TypeId) -> TypeId {
        if numeric_rank(self.model.types.kind(left)) >= numeric_rank(self.model.types.kind(right)) {
            left
        } else {
            right
        }
    }

    fn index_result(&mut self, ty: TypeId, coordinate_count: usize) -> TypeId {
        match self.model.types.kind(ty).clone() {
            TypeKind::Array {
                element,
                dimensions,
            } => {
                if coordinate_count >= dimensions.len() {
                    element
                } else {
                    self.model.types.intern(TypeKind::Array {
                        element,
                        dimensions: dimensions[coordinate_count..].to_vec(),
                    })
                }
            }
            TypeKind::Slice(element) | TypeKind::Pointer(element) => element,
            TypeKind::Reference { target, .. } | TypeKind::Const(target) => {
                self.index_result(target, coordinate_count)
            }
            _ => self.model.types.error(),
        }
    }

    fn static_target(
        &self,
        ty: &syntax::TypeSyntax,
        owner: TypeId,
        name: &str,
    ) -> hir::StaticTarget {
        if let Some(TypeResolution::Definition(definition)) =
            self.model.resolutions.types.get(&ty.id).copied()
        {
            if let Some(variant) = self
                .variants
                .get(&definition)
                .and_then(|variants| variants.get(name))
                .copied()
            {
                return hir::StaticTarget::Variant(variant);
            }
            if let Some(method) = self
                .methods
                .get(&definition)
                .and_then(|methods| methods.get(name))
                .copied()
            {
                return hir::StaticTarget::Definition(method);
            }
        }
        hir::StaticTarget::Intrinsic {
            owner,
            name: name.to_string(),
        }
    }

    fn static_result(&mut self, owner: TypeId, name: &str) -> TypeId {
        if name == "size_of" || name == "align_of" {
            return self.model.types.builtin(syntax::BuiltinType::Int);
        }
        match self.model.types.kind(owner).clone() {
            TypeKind::Intrinsic(intrinsic) => match intrinsic {
                IntrinsicType::System => self.model.types.builtin(syntax::BuiltinType::Allocator),
                IntrinsicType::Memory | IntrinsicType::Bounds | IntrinsicType::Testing => {
                    self.model.types.builtin(syntax::BuiltinType::Void)
                }
                IntrinsicType::Window => self.model.types.intrinsic(IntrinsicType::Window),
                IntrinsicType::Color => self.model.types.intrinsic(IntrinsicType::Color),
                IntrinsicType::Keyboard => self.model.types.builtin(syntax::BuiltinType::Boolean),
            },
            TypeKind::Builtin(syntax::BuiltinType::Arena) => {
                self.model.types.builtin(syntax::BuiltinType::Arena)
            }
            TypeKind::Array { .. } | TypeKind::Slice(_) | TypeKind::Pointer(_) => owner,
            TypeKind::Nominal { definition, .. } => {
                if self
                    .variants
                    .get(&definition)
                    .is_some_and(|variants| variants.contains_key(name))
                {
                    owner
                } else if let Some(method) = self
                    .methods
                    .get(&definition)
                    .and_then(|methods| methods.get(name))
                    .copied()
                {
                    self.model
                        .definition_types
                        .get(&method)
                        .copied()
                        .and_then(|ty| self.function_result_ignoring_receiver(ty))
                        .unwrap_or_else(|| self.model.types.error())
                } else if name == "create" {
                    self.model.types.intern(TypeKind::Pointer(owner))
                } else {
                    self.model.types.error()
                }
            }
            _ => self.model.types.error(),
        }
    }

    fn intrinsic_method_result(&mut self, owner: TypeId, name: &str) -> TypeId {
        match self.model.types.kind(owner).clone() {
            TypeKind::Builtin(syntax::BuiltinType::Allocator) => {
                self.model.types.builtin(syntax::BuiltinType::Void)
            }
            TypeKind::Builtin(syntax::BuiltinType::Arena) => {
                if name == "allocator" {
                    self.model.types.builtin(syntax::BuiltinType::Allocator)
                } else {
                    self.model.types.builtin(syntax::BuiltinType::Void)
                }
            }
            TypeKind::Intrinsic(IntrinsicType::Window) => match name {
                "is_open" => self.model.types.builtin(syntax::BuiltinType::Boolean),
                "delta_time" => self.model.types.builtin(syntax::BuiltinType::Double),
                _ => self.model.types.builtin(syntax::BuiltinType::Void),
            },
            _ => self.model.types.error(),
        }
    }

    fn function_result_ignoring_receiver(&self, ty: TypeId) -> Option<TypeId> {
        match self.model.types.kind(ty) {
            TypeKind::Function { result, .. } => Some(*result),
            _ => None,
        }
    }

    fn dynamic_method(&self, owner: TypeId, name: &str) -> Option<(DefId, TypeId)> {
        match self.model.types.kind(owner) {
            TypeKind::Nominal { definition, .. } => {
                self.dynamic_method_for_definition(*definition, name, &mut Vec::new())
            }
            TypeKind::Intersection(members) => members
                .iter()
                .find_map(|member| self.dynamic_method(*member, name)),
            TypeKind::Const(inner) | TypeKind::Reference { target: inner, .. } => {
                self.dynamic_method(*inner, name)
            }
            _ => None,
        }
    }

    fn dynamic_method_for_definition(
        &self,
        definition: DefId,
        name: &str,
        visiting: &mut Vec<DefId>,
    ) -> Option<(DefId, TypeId)> {
        if visiting.contains(&definition) {
            return None;
        }
        if let Some(method) = self
            .dynamic_methods
            .get(&definition)
            .and_then(|methods| methods.get(name))
            .copied()
        {
            return Some(method);
        }
        visiting.push(definition);
        let result = self
            .trait_supers
            .get(&definition)
            .into_iter()
            .flatten()
            .find_map(|supertrait| self.dynamic_method_for_definition(*supertrait, name, visiting));
        visiting.pop();
        result
    }

    fn assign_pattern_types(&mut self, case: &syntax::MatchCase, value_type: TypeId) {
        let bindings = self
            .model
            .resolutions
            .local_bindings
            .get(&case.id)
            .cloned()
            .unwrap_or_default();
        let binding_types = match &case.pattern {
            syntax::Pattern::EnumVariant {
                enum_type, variant, ..
            } => {
                let owner_type = enum_type
                    .as_ref()
                    .map(|ty| self.syntax_type(ty))
                    .unwrap_or(value_type);
                self.nominal_definition(owner_type)
                    .and_then(|owner| self.variants.get(&owner))
                    .and_then(|variants| variants.get(variant))
                    .and_then(|variant| self.model.variant_payloads.get(variant))
                    .cloned()
                    .unwrap_or_default()
            }
            syntax::Pattern::Struct { ty, fields } => {
                let owner_type = self.syntax_type(ty);
                let owner = self.nominal_definition(owner_type);
                fields
                    .iter()
                    .filter_map(|field| {
                        owner
                            .and_then(|owner| self.fields.get(&owner))
                            .and_then(|fields| fields.get(&field.name))
                            .and_then(|field| self.model.field_types.get(field))
                            .copied()
                    })
                    .collect()
            }
        };
        for (binding, ty) in bindings.into_iter().zip(binding_types) {
            self.model.local_types.insert(binding, ty);
        }
    }

    fn lower_pattern(&mut self, case: &syntax::MatchCase, value_type: TypeId) -> hir::MatchPattern {
        match &case.pattern {
            syntax::Pattern::EnumVariant {
                enum_type, variant, ..
            } => {
                let owner_type = enum_type
                    .as_ref()
                    .map(|ty| self.syntax_type(ty))
                    .unwrap_or(value_type);
                let variant = self
                    .nominal_definition(owner_type)
                    .and_then(|owner| self.variants.get(&owner))
                    .and_then(|variants| variants.get(variant))
                    .copied()
                    .unwrap_or_else(|| {
                        self.diagnostics.push(
                            Diagnostic::error(format!(
                                "could not resolve enum variant `{variant}` while lowering HIR"
                            ))
                            .with_code("E3904")
                            .at(case.span),
                        );
                        VariantId::new(u32::MAX)
                    });
                hir::MatchPattern::EnumVariant { variant }
            }
            syntax::Pattern::Struct { ty, fields } => {
                let owner_type = self.syntax_type(ty);
                let definition = self
                    .nominal_definition(owner_type)
                    .unwrap_or_else(|| DefId::new(u32::MAX));
                let fields = fields
                    .iter()
                    .map(|field| {
                        self.fields
                            .get(&definition)
                            .and_then(|fields| fields.get(&field.name))
                            .copied()
                            .unwrap_or_else(|| self.missing_field(case.span, &field.name))
                    })
                    .collect();
                hir::MatchPattern::Struct { definition, fields }
            }
        }
    }

    fn syntax_type(&self, ty: &syntax::TypeSyntax) -> TypeId {
        self.model
            .syntax_types
            .get(&ty.id)
            .copied()
            .unwrap_or_else(|| self.model.types.error())
    }

    fn nominal_definition(&self, ty: TypeId) -> Option<DefId> {
        match self.model.types.kind(ty) {
            TypeKind::Nominal { definition, .. } => Some(*definition),
            TypeKind::Const(inner)
            | TypeKind::Pointer(inner)
            | TypeKind::Reference { target: inner, .. } => self.nominal_definition(*inner),
            _ => None,
        }
    }

    fn is_intrinsic_receiver(&self, ty: TypeId) -> bool {
        matches!(
            self.model.types.kind(ty),
            TypeKind::Intrinsic(_)
                | TypeKind::Builtin(syntax::BuiltinType::Allocator)
                | TypeKind::Builtin(syntax::BuiltinType::Arena)
        )
    }

    fn is_numeric(&self, ty: TypeId) -> bool {
        numeric_rank(self.model.types.kind(ty)).is_some()
    }

    fn is_integral(&self, ty: TypeId) -> bool {
        matches!(
            self.model.types.kind(ty),
            TypeKind::Builtin(
                syntax::BuiltinType::Byte
                    | syntax::BuiltinType::Short
                    | syntax::BuiltinType::Int
                    | syntax::BuiltinType::Long
                    | syntax::BuiltinType::Char
            )
        )
    }

    fn binding(&mut self, node: crate::ids::NodeId, index: usize) -> LocalId {
        self.model
            .resolutions
            .local_bindings
            .get(&node)
            .and_then(|bindings| bindings.get(index))
            .copied()
            .unwrap_or_else(|| LocalId::new(u32::MAX))
    }

    fn missing_field(&mut self, span: crate::source_map::Span, name: &str) -> FieldId {
        self.diagnostics.push(
            Diagnostic::error(format!(
                "could not resolve field `{name}` while lowering HIR"
            ))
            .with_code("E3903")
            .at(span),
        );
        FieldId::new(u32::MAX)
    }
}

fn index_methods(
    methods: &mut HashMap<DefId, HashMap<String, DefId>>,
    model: &SemanticModel,
    owner: DefId,
    declarations: &[syntax::FunctionDecl],
) {
    let owner_methods = methods.entry(owner).or_default();
    for method in declarations {
        if let Some(definition) = model
            .resolutions
            .function_definitions
            .get(&method.body.id)
            .copied()
        {
            owner_methods.insert(method.name.clone(), definition);
        }
    }
}

fn numeric_rank(kind: &TypeKind) -> Option<u8> {
    match kind {
        TypeKind::Builtin(syntax::BuiltinType::Byte) => Some(0),
        TypeKind::Builtin(syntax::BuiltinType::Short) => Some(1),
        TypeKind::Builtin(syntax::BuiltinType::Int) => Some(2),
        TypeKind::Builtin(syntax::BuiltinType::Long) => Some(3),
        TypeKind::Builtin(syntax::BuiltinType::Float) => Some(4),
        TypeKind::Builtin(syntax::BuiltinType::Double) => Some(5),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nested_return_is_explicit_in_hir_flow() {
        let source = r#"
            function value(): int {
                { return 42; }
            }
        "#;
        let module = crate::syntax::parser::parse_test_module(source);
        let resolutions = crate::analysis::resolver::resolve(&module).unwrap();
        let mut model = crate::analysis::model::analyze_declarations(&module, resolutions).unwrap();
        let hir = lower(&module, &mut model).unwrap();
        let hir::ItemKind::Function(function) = &hir.items[0].kind else {
            panic!("expected function HIR");
        };
        assert_eq!(function.body.flow, hir::Flow::Returns);
    }

    #[test]
    fn resolves_struct_fields_from_local_types() {
        let source = r#"
            struct Point { x: int; y: int; }
            function main(): void {
                p: Point = Point { x: 1, y: 2 };
                print(p.x);
            }
        "#;
        let module = crate::syntax::parser::parse_test_module(source);
        let resolutions = crate::analysis::resolver::resolve(&module).unwrap();
        let mut model = crate::analysis::model::analyze_declarations(&module, resolutions).unwrap();
        lower(&module, &mut model).unwrap();
    }
}
