//! Lowers typed HIR into the compact tree consumed by LLVM instruction
//! selection. This keeps specialization state out of `CheckedProgram`: the
//! backend derives all body input from HIR and semantic identities.

use super::*;
use crate::hir;
use crate::ids::{FieldId, LocalId, VariantId};
use crate::specialization::tree::{
    EnumVariant, MatchCase, MatchPattern, Metadata, StructPatternField,
};

pub(super) fn lower_program(module: &hir::Module, model: &SemanticModel) -> Result<Node, String> {
    HirTreeLowerer::new(module, model).program()
}

struct HirTreeLowerer<'a> {
    module: &'a hir::Module,
    model: &'a SemanticModel,
    fields: HashMap<FieldId, (DefId, String)>,
    variants: HashMap<VariantId, (DefId, String)>,
    methods: HashMap<DefId, DefId>,
}

impl<'a> HirTreeLowerer<'a> {
    fn new(module: &'a hir::Module, model: &'a SemanticModel) -> Self {
        let mut fields = HashMap::new();
        let mut variants = HashMap::new();
        let mut methods = HashMap::new();
        for item in &module.items {
            let Some(owner) = item.definition else {
                continue;
            };
            match &item.kind {
                hir::ItemKind::Struct(declaration) => {
                    for field in &declaration.fields {
                        fields.insert(field.id, (owner, field.name.clone()));
                    }
                    for method in &declaration.methods {
                        if let Some(definition) = method.definition {
                            methods.insert(definition, owner);
                        }
                    }
                }
                hir::ItemKind::Enum(declaration) => {
                    for variant in &declaration.variants {
                        variants.insert(variant.id, (owner, variant.name.clone()));
                    }
                    for method in &declaration.methods {
                        if let Some(definition) = method.definition {
                            methods.insert(definition, owner);
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
        }
    }

    fn program(&self) -> Result<Node, String> {
        let mut statements = Vec::new();
        for item in &self.module.items {
            match (&item.kind, item.definition) {
                (hir::ItemKind::Function(function), Some(definition)) => {
                    statements.push(self.function(function, self.definition_name(definition)?)?);
                }
                (hir::ItemKind::ExternFunction(signature), Some(definition)) => {
                    statements.push(Node::ExternFunctionDeclaration {
                        name: self.definition_name(definition)?.to_string(),
                        parameters: signature
                            .parameters
                            .iter()
                            .enumerate()
                            .map(|(index, ty)| Ok((format!("arg{index}"), self.ty(*ty)?)))
                            .collect::<Result<_, String>>()?,
                        return_type: self.ty(signature.result)?,
                    });
                }
                (hir::ItemKind::Struct(declaration), Some(definition)) => {
                    statements.push(Node::StructDeclaration {
                        name: self.definition_name(definition)?.to_string(),
                        fields: declaration
                            .fields
                            .iter()
                            .map(|field| Ok((field.name.clone(), self.ty(field.ty)?)))
                            .collect::<Result<_, String>>()?,
                        functions: declaration
                            .methods
                            .iter()
                            .map(|method| {
                                let definition = method.definition.ok_or_else(|| {
                                    "HIR method is missing its definition".to_string()
                                })?;
                                self.function(method, self.definition_name(definition)?)
                            })
                            .collect::<Result<_, String>>()?,
                    });
                }
                (hir::ItemKind::Enum(declaration), Some(definition)) => {
                    statements.push(Node::EnumDeclaration {
                        name: self.definition_name(definition)?.to_string(),
                        variants: declaration
                            .variants
                            .iter()
                            .map(|variant| {
                                Ok(EnumVariant {
                                    name: variant.name.clone(),
                                    payload_types: variant
                                        .payload
                                        .iter()
                                        .map(|ty| self.ty(*ty))
                                        .collect::<Result<_, String>>()?,
                                })
                            })
                            .collect::<Result<_, String>>()?,
                        functions: declaration
                            .methods
                            .iter()
                            .map(|method| {
                                let definition = method.definition.ok_or_else(|| {
                                    "HIR method is missing its definition".to_string()
                                })?;
                                self.function(method, self.definition_name(definition)?)
                            })
                            .collect::<Result<_, String>>()?,
                    });
                }
                // Layouts, trait records, and implementations are consumed
                // directly from typed HIR before function-body lowering.
                _ => {}
            }
        }
        statements.push(Node::End);
        Ok(Node::Program { statements })
    }

    fn function(&self, function: &hir::Function, name: &str) -> Result<Node, String> {
        Ok(Node::FunctionDeclaration {
            name: name.to_string(),
            parameters: function
                .parameters
                .iter()
                .map(|parameter| {
                    let name = self.local_name(parameter.local)?.to_string();
                    let ty = if name == "self" {
                        Type::SkSelf
                    } else {
                        let ty = self.ty(parameter.ty)?;
                        if parameter.is_const {
                            Type::BindingConst {
                                inner: Box::new(ty),
                            }
                        } else {
                            ty
                        }
                    };
                    Ok((name, ty))
                })
                .collect::<Result<_, String>>()?,
            return_type: self.ty(function.result)?,
            body: self.block(&function.body)?,
            lambda: function.definition.is_none(),
        })
    }

    fn block(&self, block: &hir::Block) -> Result<Vec<Node>, String> {
        block
            .statements
            .iter()
            .map(|statement| self.statement(statement))
            .collect()
    }

    fn statement(&self, statement: &hir::Stmt) -> Result<Node, String> {
        Ok(match &statement.kind {
            hir::StmtKind::Local {
                local,
                ty,
                is_const,
                initializer,
            } => {
                let ty = self.ty(*ty)?;
                Node::VariableDeclaration {
                    var_type: if *is_const {
                        Type::BindingConst {
                            inner: Box::new(ty),
                        }
                    } else {
                        ty
                    },
                    name: self.local_name(*local)?.to_string(),
                    value: initializer
                        .as_ref()
                        .map(|expression| self.expression(expression).map(Box::new))
                        .transpose()?,
                    metadata: Metadata::EMPTY,
                }
            }
            hir::StmtKind::Destructure { value, bindings } => {
                let struct_type = self.ty(value.ty)?;
                Node::StructDestructure {
                    struct_type,
                    fields: bindings
                        .iter()
                        .map(|(field, local)| {
                            Ok(StructPatternField {
                                field_name: self.field_name(*field)?.to_string(),
                                binding: self.local_name(*local)?.to_string(),
                            })
                        })
                        .collect::<Result<_, String>>()?,
                    value: Box::new(self.expression(value)?),
                    metadata: Metadata::EMPTY,
                }
            }
            hir::StmtKind::Assignment { target, value } => Node::Assignment {
                var: Box::new(self.expression(target)?),
                value: Box::new(self.expression(value)?),
                metadata: Metadata::EMPTY,
            },
            hir::StmtKind::Expression(expression) => self.expression(expression)?,
            hir::StmtKind::Return(expression) => Node::Return(
                expression
                    .as_ref()
                    .map(|expression| self.expression(expression).map(Box::new))
                    .transpose()?,
            ),
            hir::StmtKind::Defer(expression) => Node::Defer(Box::new(self.expression(expression)?)),
            hir::StmtKind::Print(expression) => Node::Print(Box::new(self.expression(expression)?)),
            hir::StmtKind::Input => Node::Input,
            hir::StmtKind::Block(block) => Node::Block {
                statements: self.block(block)?,
            },
            hir::StmtKind::Unsafe(block) => Node::UnsafeBlock {
                statements: self.block(block)?,
            },
            hir::StmtKind::If(branch) => Node::If {
                condition: Box::new(self.expression(&branch.condition)?),
                body: self.block(&branch.then_block)?,
                else_if_blocks: branch
                    .else_if
                    .iter()
                    .map(|(condition, block)| {
                        Ok(Node::If {
                            condition: Box::new(self.expression(condition)?),
                            body: self.block(block)?,
                            else_if_blocks: Vec::new(),
                            else_block: None,
                        })
                    })
                    .collect::<Result<_, String>>()?,
                else_block: branch
                    .else_block
                    .as_ref()
                    .map(|block| self.block(block))
                    .transpose()?,
            },
            hir::StmtKind::Match(branch) => Node::Match {
                value: Box::new(self.expression(&branch.value)?),
                cases: branch
                    .cases
                    .iter()
                    .map(|case| self.match_case(case))
                    .collect::<Result<_, String>>()?,
            },
            hir::StmtKind::For(loop_statement) => Node::For {
                init: loop_statement
                    .initializer
                    .as_deref()
                    .map(|statement| self.statement(statement).map(Box::new))
                    .transpose()?,
                condition: loop_statement
                    .condition
                    .as_ref()
                    .map(|expression| self.expression(expression).map(Box::new))
                    .transpose()?,
                update: loop_statement
                    .update
                    .as_deref()
                    .map(|statement| self.statement(statement).map(Box::new))
                    .transpose()?,
                body: self.block(&loop_statement.body)?,
            },
        })
    }

    fn match_case(&self, case: &hir::MatchCase) -> Result<MatchCase, String> {
        let pattern = match &case.pattern {
            hir::MatchPattern::EnumVariant { variant } => {
                let (owner, name) = self.variant(*variant)?;
                MatchPattern::EnumVariant {
                    enum_type: Some(Type::Custom(self.definition_name(owner)?.to_string())),
                    variant: name.to_string(),
                    bindings: case
                        .bindings
                        .iter()
                        .map(|local| self.local_name(*local).map(str::to_string))
                        .collect::<Result<_, String>>()?,
                }
            }
            hir::MatchPattern::Struct { definition, fields } => MatchPattern::Struct {
                struct_type: Type::Custom(self.definition_name(*definition)?.to_string()),
                fields: fields
                    .iter()
                    .zip(&case.bindings)
                    .map(|(field, local)| {
                        Ok(StructPatternField {
                            field_name: self.field_name(*field)?.to_string(),
                            binding: self.local_name(*local)?.to_string(),
                        })
                    })
                    .collect::<Result<_, String>>()?,
            },
        };
        Ok(MatchCase {
            pattern,
            body: self.block(&case.body)?,
        })
    }

    fn expression(&self, expression: &hir::Expr) -> Result<Node, String> {
        Ok(match &expression.kind {
            hir::ExprKind::Literal(literal) => Node::Literal(match literal {
                crate::syntax::ast::Literal::Integer(value) => Literal::Integer(*value),
                crate::syntax::ast::Literal::Long(value) => Literal::Long(*value),
                crate::syntax::ast::Literal::Float(value) => Literal::Float(*value),
                crate::syntax::ast::Literal::Double(value) => Literal::Double(*value),
                crate::syntax::ast::Literal::String(value) => Literal::String(value.clone()),
                crate::syntax::ast::Literal::Boolean(value) => Literal::Boolean(*value),
                crate::syntax::ast::Literal::Char(value) => Literal::Char(*value),
            }),
            hir::ExprKind::Value(value) => Node::Access {
                nodes: vec![Node::Identifier(match value {
                    hir::Value::Definition(definition) => {
                        self.definition_name(*definition)?.to_string()
                    }
                    hir::Value::Local(local) => self.local_name(*local)?.to_string(),
                })],
            },
            hir::ExprKind::Unary { operator, operand } => {
                if *operator == crate::syntax::ast::UnaryOperator::Dereference {
                    let mut nodes = self.access_parts(self.expression(operand)?)?;
                    nodes.push(Node::Dereference {
                        metadata: Metadata::EMPTY,
                    });
                    Node::Access { nodes }
                } else {
                    Node::UnaryOp {
                        operator: match operator {
                            crate::syntax::ast::UnaryOperator::Plus => UnaryOperator::Plus,
                            crate::syntax::ast::UnaryOperator::Minus => UnaryOperator::Minus,
                            crate::syntax::ast::UnaryOperator::Not => UnaryOperator::Negate,
                            crate::syntax::ast::UnaryOperator::AddressOf => {
                                UnaryOperator::AddressOf
                            }
                            crate::syntax::ast::UnaryOperator::AddressOfMut => {
                                UnaryOperator::AddressOfMut
                            }
                            crate::syntax::ast::UnaryOperator::Dereference => unreachable!(),
                        },
                        operand: Box::new(self.expression(operand)?),
                    }
                }
            }
            hir::ExprKind::Binary {
                left,
                operator,
                right,
            } => Node::BinaryOp {
                left: Box::new(self.expression(left)?),
                operator: match operator {
                    crate::syntax::ast::BinaryOperator::Add => Operator::Add,
                    crate::syntax::ast::BinaryOperator::Subtract => Operator::Subtract,
                    crate::syntax::ast::BinaryOperator::Multiply => Operator::Multiply,
                    crate::syntax::ast::BinaryOperator::Divide => Operator::Divide,
                    crate::syntax::ast::BinaryOperator::Modulo => Operator::Mod,
                    crate::syntax::ast::BinaryOperator::Power => Operator::Power,
                    crate::syntax::ast::BinaryOperator::Equals => Operator::Equals,
                    crate::syntax::ast::BinaryOperator::NotEquals => Operator::NotEquals,
                    crate::syntax::ast::BinaryOperator::LessThan => Operator::LessThan,
                    crate::syntax::ast::BinaryOperator::GreaterThan => Operator::GreaterThan,
                    crate::syntax::ast::BinaryOperator::LessThanOrEqual => {
                        Operator::LessThanOrEqual
                    }
                    crate::syntax::ast::BinaryOperator::GreaterThanOrEqual => {
                        Operator::GreaterThanOrEqual
                    }
                    crate::syntax::ast::BinaryOperator::And => Operator::And,
                    crate::syntax::ast::BinaryOperator::Or => Operator::Or,
                },
                right: Box::new(self.expression(right)?),
            },
            hir::ExprKind::Call {
                callee,
                argument_groups,
            } => {
                let mut nodes = self.access_parts(self.expression(callee)?)?;
                let last = nodes
                    .pop()
                    .ok_or_else(|| "call target has no value".to_string())?;
                let Node::Identifier(name) = last else {
                    return Err("HIR call target is not a named function value".to_string());
                };
                nodes.push(Node::FunctionCall {
                    name,
                    type_arguments: Vec::new(),
                    arguments: self.argument_groups(argument_groups)?,
                    metadata: Metadata::EMPTY,
                });
                if nodes.len() == 1 {
                    nodes
                        .pop()
                        .ok_or_else(|| "call target has no value".to_string())?
                } else {
                    Node::Access { nodes }
                }
            }
            hir::ExprKind::MethodCall {
                receiver,
                method,
                argument_groups,
            } => {
                let mut nodes = self.access_parts(self.expression(receiver)?)?;
                nodes.push(Node::MemberAccess {
                    member: Box::new(Node::FunctionCall {
                        name: self.method_name(method)?.to_string(),
                        type_arguments: Vec::new(),
                        arguments: self.argument_groups(argument_groups)?,
                        metadata: Metadata::EMPTY,
                    }),
                    metadata: Metadata::EMPTY,
                });
                Node::Access { nodes }
            }
            hir::ExprKind::Field { receiver, field } => {
                let mut nodes = self.access_parts(self.expression(receiver)?)?;
                nodes.push(Node::MemberAccess {
                    member: Box::new(Node::Identifier(self.field_name(*field)?.to_string())),
                    metadata: Metadata::EMPTY,
                });
                Node::Access { nodes }
            }
            hir::ExprKind::Length { receiver } => {
                let mut nodes = self.access_parts(self.expression(receiver)?)?;
                nodes.push(Node::MemberAccess {
                    member: Box::new(Node::Identifier("len".to_string())),
                    metadata: Metadata::EMPTY,
                });
                Node::Access { nodes }
            }
            hir::ExprKind::Index {
                receiver,
                coordinates,
            } => {
                let mut nodes = self.access_parts(self.expression(receiver)?)?;
                nodes.push(Node::ArrayAccess {
                    coordinates: coordinates
                        .iter()
                        .map(|coordinate| self.expression(coordinate))
                        .collect::<Result<_, String>>()?,
                });
                Node::Access { nodes }
            }
            hir::ExprKind::Slice {
                receiver,
                start,
                end,
            } => {
                let mut nodes = self.access_parts(self.expression(receiver)?)?;
                nodes.push(Node::SliceAccess {
                    start: start
                        .as_deref()
                        .map(|value| self.expression(value).map(Box::new))
                        .transpose()?,
                    end: end
                        .as_deref()
                        .map(|value| self.expression(value).map(Box::new))
                        .transpose()?,
                });
                Node::Access { nodes }
            }
            hir::ExprKind::StructInit { definition, fields } => Node::StructInitialization {
                _type: Type::Custom(self.definition_name(*definition)?.to_string()),
                fields: fields
                    .iter()
                    .map(|(field, value)| {
                        Ok((
                            self.field_name(*field)?.to_string(),
                            self.expression(value)?,
                        ))
                    })
                    .collect::<Result<_, String>>()?,
            },
            hir::ExprKind::StaticCall { target, arguments } => {
                let (owner, name) = self.static_target(target)?;
                Node::StaticFunctionCall {
                    _type: owner,
                    name,
                    arguments: arguments
                        .iter()
                        .map(|argument| self.expression(argument))
                        .collect::<Result<_, String>>()?,
                    metadata: Metadata::EMPTY,
                }
            }
            hir::ExprKind::Array(elements) => Node::ArrayInit {
                elements: elements
                    .iter()
                    .map(|element| self.expression(element))
                    .collect::<Result<_, String>>()?,
            },
            hir::ExprKind::Lambda(function) => self.function(function, "anonymous")?,
            hir::ExprKind::Block(block) => Node::Block {
                statements: self.block(block)?,
            },
        })
    }

    fn argument_groups(&self, groups: &[Vec<hir::Expr>]) -> Result<Vec<Vec<Node>>, String> {
        groups
            .iter()
            .map(|group| {
                group
                    .iter()
                    .map(|argument| self.expression(argument))
                    .collect()
            })
            .collect()
    }

    fn access_parts(&self, node: Node) -> Result<Vec<Node>, String> {
        match node {
            Node::Access { nodes } => Ok(nodes),
            other => Err(format!(
                "HIR access receiver cannot be represented for LLVM lowering: {other:?}"
            )),
        }
    }

    fn static_target(&self, target: &hir::StaticTarget) -> Result<(Type, String), String> {
        match target {
            hir::StaticTarget::Definition(definition) => {
                let owner =
                    self.methods.get(definition).copied().ok_or_else(|| {
                        format!("static method {} has no owner", definition.index())
                    })?;
                Ok((
                    Type::Custom(self.definition_name(owner)?.to_string()),
                    self.definition_name(*definition)?.to_string(),
                ))
            }
            hir::StaticTarget::Variant(variant) => {
                let (owner, name) = self.variant(*variant)?;
                Ok((
                    Type::Custom(self.definition_name(owner)?.to_string()),
                    name.to_string(),
                ))
            }
            hir::StaticTarget::Intrinsic { owner, name } => Ok((self.ty(*owner)?, name.clone())),
        }
    }

    fn method_name<'b>(&'b self, target: &'b hir::MethodTarget) -> Result<&'b str, String> {
        match target {
            hir::MethodTarget::Definition(definition)
            | hir::MethodTarget::Dynamic {
                method: definition, ..
            } => self.definition_name(*definition),
            hir::MethodTarget::Intrinsic { name, .. } => Ok(name),
        }
    }

    fn ty(&self, ty: TypeId) -> Result<Type, String> {
        Ok(match self.model.types.kind(ty) {
            SemanticTypeKind::Builtin(builtin) => self.builtin(*builtin),
            SemanticTypeKind::Intrinsic(intrinsic) => Type::Custom(intrinsic.name().to_string()),
            SemanticTypeKind::Nominal {
                definition,
                arguments,
            } => {
                let name = self.definition_name(*definition)?.to_string();
                if arguments.is_empty() {
                    Type::Custom(name)
                } else {
                    Type::GenericInstance {
                        base: name,
                        type_arguments: arguments
                            .iter()
                            .map(|argument| self.ty(*argument))
                            .collect::<Result<_, String>>()?,
                    }
                }
            }
            SemanticTypeKind::Const(inner) => Type::Const {
                inner: Box::new(self.ty(*inner)?),
            },
            SemanticTypeKind::Array {
                element,
                dimensions,
            } => Type::Array {
                elem_type: Box::new(self.ty(*element)?),
                dimensions: dimensions
                    .iter()
                    .map(|dimension| {
                        i64::try_from(*dimension)
                            .map(|value| Node::Literal(Literal::Integer(value)))
                            .map_err(|_| format!("array dimension `{dimension}` is too large"))
                    })
                    .collect::<Result<_, String>>()?,
            },
            SemanticTypeKind::Reference { target, mutable } => Type::Reference {
                target_type: Box::new(self.ty(*target)?),
                mutable: *mutable,
            },
            SemanticTypeKind::Pointer(target) => Type::Pointer {
                target_type: Box::new(self.ty(*target)?),
            },
            SemanticTypeKind::Slice(element) => Type::Slice {
                elem_type: Box::new(self.ty(*element)?),
            },
            SemanticTypeKind::Union(members) => Type::Union(
                members
                    .iter()
                    .map(|member| self.ty(*member))
                    .collect::<Result<_, String>>()?,
            ),
            SemanticTypeKind::Intersection(members) => Type::Intersection(
                members
                    .iter()
                    .map(|member| self.ty(*member))
                    .collect::<Result<_, String>>()?,
            ),
            SemanticTypeKind::Function { parameters, result } => Type::Function {
                parameters: parameters
                    .iter()
                    .map(|parameter| self.ty(*parameter))
                    .collect::<Result<_, String>>()?,
                return_type: Box::new(self.ty(*result)?),
            },
            SemanticTypeKind::GenericParameter(definition) => {
                return Err(format!(
                    "unspecialized generic `{}` reached LLVM body lowering",
                    self.definition_name(*definition)?
                ));
            }
            SemanticTypeKind::Error => {
                return Err("error type reached LLVM body lowering".to_string());
            }
            SemanticTypeKind::Never => {
                return Err("never type reached LLVM body lowering".to_string());
            }
        })
    }

    fn builtin(&self, builtin: crate::syntax::ast::BuiltinType) -> Type {
        match builtin {
            crate::syntax::ast::BuiltinType::Void => Type::Void,
            crate::syntax::ast::BuiltinType::Byte => Type::Byte,
            crate::syntax::ast::BuiltinType::Short => Type::Short,
            crate::syntax::ast::BuiltinType::Int => Type::Int,
            crate::syntax::ast::BuiltinType::Long => Type::Long,
            crate::syntax::ast::BuiltinType::Float => Type::Float,
            crate::syntax::ast::BuiltinType::Double => Type::Double,
            crate::syntax::ast::BuiltinType::String => Type::String,
            crate::syntax::ast::BuiltinType::Boolean => Type::Boolean,
            crate::syntax::ast::BuiltinType::Char => Type::Char,
            crate::syntax::ast::BuiltinType::Allocator => Type::Allocator,
            crate::syntax::ast::BuiltinType::Arena => Type::Arena,
        }
    }

    fn definition_name(&self, definition: DefId) -> Result<&str, String> {
        self.model
            .resolutions
            .definitions
            .get(definition.index())
            .map(|definition| definition.name.as_str())
            .ok_or_else(|| format!("unknown definition id {}", definition.index()))
    }

    fn local_name(&self, local: LocalId) -> Result<&str, String> {
        self.model
            .resolutions
            .locals
            .get(local.index())
            .map(|local| local.name.as_str())
            .ok_or_else(|| format!("unknown local id {}", local.index()))
    }

    fn field_name(&self, field: FieldId) -> Result<&str, String> {
        self.fields
            .get(&field)
            .map(|(_, name)| name.as_str())
            .ok_or_else(|| format!("unknown field id {}", field.index()))
    }

    fn variant(&self, variant: VariantId) -> Result<(DefId, &str), String> {
        self.variants
            .get(&variant)
            .map(|(owner, name)| (*owner, name.as_str()))
            .ok_or_else(|| format!("unknown variant id {}", variant.index()))
    }
}
