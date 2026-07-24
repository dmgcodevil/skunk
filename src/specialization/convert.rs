//! Conversion between source syntax and the internal generic-expansion tree.

use crate::diagnostic::Diagnostic;
use crate::ids::{FileId, NodeId, NodeIdAllocator};
use crate::source_map::Span;
use crate::specialization::tree;
use crate::syntax::ast as syntax;

type TreeGenericParts = (
    Vec<String>,
    HashMap<String, Vec<String>>,
    HashMap<String, tree::SubtypeBounds>,
);
use std::collections::HashMap;

pub(crate) fn from_tree(
    program: &tree::Node,
    file: FileId,
    source_len: usize,
) -> Result<syntax::Module, Diagnostic> {
    let span = Span::new(file, 0, source_len).ok_or_else(|| {
        Diagnostic::error("source file is too large to represent with compiler spans")
            .with_code("E0001")
    })?;
    TreeToSyntax {
        ids: NodeIdAllocator::for_file(file),
        span,
    }
    .module(program)
}

pub(crate) fn to_tree(module: &syntax::Module) -> tree::Node {
    SyntaxToTree.module(module)
}

struct TreeToSyntax {
    ids: NodeIdAllocator,
    span: Span,
}

impl TreeToSyntax {
    fn id(&mut self) -> NodeId {
        self.ids.allocate()
    }

    fn module(&mut self, node: &tree::Node) -> Result<syntax::Module, Diagnostic> {
        let tree::Node::Program { statements } = node else {
            return Err(
                Diagnostic::error("specialization input must have a program root")
                    .with_code("E9001"),
            );
        };

        let mut name = None;
        let mut entries = Vec::new();
        for statement in statements {
            match statement {
                tree::Node::Module { name: module_name } => {
                    name = Some(syntax::Path::from_qualified(module_name));
                }
                tree::Node::End => {}
                tree::Node::Export { declaration } => {
                    entries.push(self.top_level(declaration, syntax::Visibility::Public)?);
                }
                other => {
                    entries.push(self.top_level(other, syntax::Visibility::Private)?);
                }
            }
        }

        let id = self.id();
        Ok(syntax::Module {
            id,
            span: self.span,
            name,
            entries,
        })
    }

    fn top_level(
        &mut self,
        node: &tree::Node,
        visibility: syntax::Visibility,
    ) -> Result<syntax::TopLevel, Diagnostic> {
        let kind = match node {
            tree::Node::Import { name } => syntax::TopLevelKind::Import(syntax::ImportDecl {
                module: syntax::Path::from_qualified(name),
            }),
            tree::Node::TypeAliasDeclaration {
                name,
                generic_params,
                generic_bounds,
                subtype_bounds,
                target_type,
            } => syntax::TopLevelKind::TypeAlias(syntax::TypeAliasDecl {
                name: name.clone(),
                generic_parameters: self.generic_parameters(
                    generic_params,
                    generic_bounds,
                    subtype_bounds,
                ),
                target: self.ty(target_type),
            }),
            tree::Node::StructDeclaration {
                name,
                fields,
                functions,
            } => syntax::TopLevelKind::Struct(self.struct_decl(
                name,
                &[],
                &HashMap::new(),
                &HashMap::new(),
                fields,
                functions,
            )?),
            tree::Node::GenericStructDeclaration {
                name,
                generic_params,
                generic_bounds,
                subtype_bounds,
                fields,
                functions,
            } => syntax::TopLevelKind::Struct(self.struct_decl(
                name,
                generic_params,
                generic_bounds,
                subtype_bounds,
                fields,
                functions,
            )?),
            tree::Node::EnumDeclaration {
                name,
                variants,
                functions,
            } => syntax::TopLevelKind::Enum(self.enum_decl(
                name,
                &[],
                &HashMap::new(),
                &HashMap::new(),
                variants,
                functions,
            )?),
            tree::Node::GenericEnumDeclaration {
                name,
                generic_params,
                generic_bounds,
                subtype_bounds,
                variants,
                functions,
            } => syntax::TopLevelKind::Enum(self.enum_decl(
                name,
                generic_params,
                generic_bounds,
                subtype_bounds,
                variants,
                functions,
            )?),
            tree::Node::TraitDeclaration {
                name,
                generic_params,
                generic_bounds,
                subtype_bounds,
                supertraits,
                methods,
            } => {
                let methods = methods
                    .iter()
                    .map(|method| self.trait_method(method))
                    .collect::<Result<Vec<_>, _>>()?;
                syntax::TopLevelKind::Trait(syntax::TraitDecl {
                    name: name.clone(),
                    generic_parameters: self.generic_parameters(
                        generic_params,
                        generic_bounds,
                        subtype_bounds,
                    ),
                    supertraits: supertraits
                        .iter()
                        .map(|name| syntax::Path::from_qualified(name))
                        .collect(),
                    methods,
                })
            }
            tree::Node::ShapeDeclaration { name, methods } => {
                let methods = methods
                    .iter()
                    .map(|method| self.trait_method(method))
                    .collect::<Result<Vec<_>, _>>()?;
                syntax::TopLevelKind::Shape(syntax::ShapeDecl {
                    name: name.clone(),
                    methods,
                })
            }
            tree::Node::AttachDeclaration {
                generic_params,
                generic_bounds,
                subtype_bounds,
                target_type,
                functions,
            } => {
                let methods = functions
                    .iter()
                    .map(|function| self.function(function))
                    .collect::<Result<Vec<_>, _>>()?;
                syntax::TopLevelKind::Attach(syntax::AttachDecl {
                    generic_parameters: self.generic_parameters(
                        generic_params,
                        generic_bounds,
                        subtype_bounds,
                    ),
                    target: self.ty(target_type),
                    methods,
                })
            }
            tree::Node::ConformDeclaration {
                generic_params,
                generic_bounds,
                subtype_bounds,
                trait_type,
                target_type,
                functions,
            } => {
                let methods = functions
                    .iter()
                    .map(|function| self.function(function))
                    .collect::<Result<Vec<_>, _>>()?;
                syntax::TopLevelKind::Conformance(syntax::ConformanceDecl {
                    generic_parameters: self.generic_parameters(
                        generic_params,
                        generic_bounds,
                        subtype_bounds,
                    ),
                    traits: vec![self.ty(trait_type)],
                    target: self.ty(target_type),
                    methods,
                })
            }
            tree::Node::ImplDeclaration {
                generic_params,
                generic_bounds,
                subtype_bounds,
                trait_types,
                target_type,
            } => syntax::TopLevelKind::Implementation(syntax::ImplementationDecl {
                generic_parameters: self.generic_parameters(
                    generic_params,
                    generic_bounds,
                    subtype_bounds,
                ),
                traits: trait_types.iter().map(|ty| self.ty(ty)).collect(),
                target: self.ty(target_type),
            }),
            tree::Node::FunctionDeclaration { lambda: false, .. }
            | tree::Node::GenericFunctionDeclaration { lambda: false, .. } => {
                syntax::TopLevelKind::Function(self.function(node)?)
            }
            tree::Node::ExternFunctionDeclaration {
                name,
                parameters,
                return_type,
            } => syntax::TopLevelKind::ExternFunction(syntax::ExternFunctionDecl {
                abi: "C".to_string(),
                name: name.clone(),
                parameters: self.parameters(parameters),
                return_type: self.ty(return_type),
            }),
            tree::Node::TestDeclaration { name, body } => {
                syntax::TopLevelKind::Test(syntax::TestDecl {
                    name: name.clone(),
                    body: self.block(body)?,
                })
            }
            tree::Node::VariableDeclaration {
                var_type,
                name,
                value,
                ..
            } => syntax::TopLevelKind::Global(self.local_decl(name, var_type, value.as_deref())?),
            tree::Node::Module { .. } | tree::Node::Export { .. } | tree::Node::End => {
                return Err(Diagnostic::error(format!(
                    "invalid nested specialization node: {node:?}"
                ))
                .with_code("E9002"));
            }
            other => syntax::TopLevelKind::Statement(self.stmt(other)?),
        };

        let id = self.id();
        Ok(syntax::TopLevel {
            id,
            span: self.span,
            visibility,
            kind,
        })
    }

    fn struct_decl(
        &mut self,
        name: &str,
        generic_params: &[String],
        generic_bounds: &HashMap<String, Vec<String>>,
        subtype_bounds: &HashMap<String, tree::SubtypeBounds>,
        fields: &[(String, tree::Type)],
        functions: &[tree::Node],
    ) -> Result<syntax::StructDecl, Diagnostic> {
        let fields = fields
            .iter()
            .map(|(name, ty)| {
                let (is_const, ty) = split_binding_const(ty);
                let id = self.id();
                syntax::StructField {
                    id,
                    span: self.span,
                    name: name.clone(),
                    is_const,
                    ty: self.ty(ty),
                }
            })
            .collect();
        let methods = functions
            .iter()
            .map(|function| self.function(function))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(syntax::StructDecl {
            name: name.to_string(),
            generic_parameters: self.generic_parameters(
                generic_params,
                generic_bounds,
                subtype_bounds,
            ),
            fields,
            methods,
        })
    }

    fn enum_decl(
        &mut self,
        name: &str,
        generic_params: &[String],
        generic_bounds: &HashMap<String, Vec<String>>,
        subtype_bounds: &HashMap<String, tree::SubtypeBounds>,
        variants: &[tree::EnumVariant],
        functions: &[tree::Node],
    ) -> Result<syntax::EnumDecl, Diagnostic> {
        let variants = variants
            .iter()
            .map(|variant| {
                let id = self.id();
                syntax::EnumVariant {
                    id,
                    span: self.span,
                    name: variant.name.clone(),
                    payload: variant.payload_types.iter().map(|ty| self.ty(ty)).collect(),
                }
            })
            .collect();
        let methods = functions
            .iter()
            .map(|function| self.function(function))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(syntax::EnumDecl {
            name: name.to_string(),
            generic_parameters: self.generic_parameters(
                generic_params,
                generic_bounds,
                subtype_bounds,
            ),
            variants,
            methods,
        })
    }

    fn trait_method(
        &mut self,
        method: &tree::TraitMethodSignature,
    ) -> Result<syntax::TraitMethod, Diagnostic> {
        let default_body = method
            .default_body
            .as_deref()
            .map(|body| self.block(body))
            .transpose()?;
        let id = self.id();
        Ok(syntax::TraitMethod {
            id,
            span: self.span,
            name: method.name.clone(),
            parameters: self.parameters(&method.parameters),
            return_type: self.ty(&method.return_type),
            default_body,
        })
    }

    fn function(&mut self, node: &tree::Node) -> Result<syntax::FunctionDecl, Diagnostic> {
        let (name, generic_parameters, parameters, return_type, body) = match node {
            tree::Node::FunctionDeclaration {
                name,
                parameters,
                return_type,
                body,
                ..
            } => (
                name.clone(),
                Vec::new(),
                self.parameters(parameters),
                self.ty(return_type),
                self.block(body)?,
            ),
            tree::Node::GenericFunctionDeclaration {
                name,
                generic_params,
                generic_bounds,
                subtype_bounds,
                parameters,
                return_type,
                body,
                ..
            } => (
                name.clone(),
                self.generic_parameters(generic_params, generic_bounds, subtype_bounds),
                self.parameters(parameters),
                self.ty(return_type),
                self.block(body)?,
            ),
            other => {
                return Err(Diagnostic::error(format!(
                    "expected a function in specialization conversion, found {other:?}"
                ))
                .with_code("E9003"));
            }
        };

        Ok(syntax::FunctionDecl {
            name,
            generic_parameters,
            parameters,
            return_type,
            body,
        })
    }

    fn generic_parameters(
        &mut self,
        names: &[String],
        capabilities: &HashMap<String, Vec<String>>,
        subtype_bounds: &HashMap<String, tree::SubtypeBounds>,
    ) -> Vec<syntax::GenericParameter> {
        names
            .iter()
            .map(|name| {
                let bounds = subtype_bounds.get(name);
                let id = self.id();
                syntax::GenericParameter {
                    id,
                    span: self.span,
                    name: name.clone(),
                    capabilities: capabilities
                        .get(name)
                        .into_iter()
                        .flatten()
                        .map(|name| syntax::Path::from_qualified(name))
                        .collect(),
                    lower_bound: bounds
                        .and_then(|bound| bound.lower.as_ref())
                        .map(|ty| self.ty(ty)),
                    upper_bound: bounds
                        .and_then(|bound| bound.upper.as_ref())
                        .map(|ty| self.ty(ty)),
                }
            })
            .collect()
    }

    fn parameters(&mut self, parameters: &[(String, tree::Type)]) -> Vec<syntax::Parameter> {
        parameters
            .iter()
            .map(|(name, ty)| {
                let kind = match ty {
                    tree::Type::SkSelf => syntax::ParameterKind::Receiver {
                        mutable: false,
                        is_const: false,
                    },
                    tree::Type::MutSelf => syntax::ParameterKind::Receiver {
                        mutable: true,
                        is_const: false,
                    },
                    _ => {
                        let (is_const, ty) = split_binding_const(ty);
                        syntax::ParameterKind::Named {
                            name: name.clone(),
                            is_const,
                            ty: self.ty(ty),
                        }
                    }
                };
                let id = self.id();
                syntax::Parameter {
                    id,
                    span: self.span,
                    kind,
                }
            })
            .collect()
    }

    fn block(&mut self, statements: &[tree::Node]) -> Result<syntax::Block, Diagnostic> {
        let statements = statements
            .iter()
            .map(|statement| self.stmt(statement))
            .collect::<Result<Vec<_>, _>>()?;
        let id = self.id();
        Ok(syntax::Block {
            id,
            span: self.span,
            statements,
        })
    }

    fn stmt(&mut self, node: &tree::Node) -> Result<syntax::Stmt, Diagnostic> {
        let kind = match node {
            tree::Node::VariableDeclaration {
                var_type,
                name,
                value,
                ..
            } => syntax::StmtKind::Local(self.local_decl(name, var_type, value.as_deref())?),
            tree::Node::StructDestructure {
                struct_type,
                fields,
                value,
                ..
            } => syntax::StmtKind::StructDestructure(syntax::StructDestructure {
                ty: self.ty(struct_type),
                fields: fields
                    .iter()
                    .map(|field| syntax::PatternField {
                        name: field.field_name.clone(),
                        binding: field.binding.clone(),
                    })
                    .collect(),
                value: self.expr(value)?,
            }),
            tree::Node::Assignment { var, value, .. } => syntax::StmtKind::Assignment {
                target: self.expr(var)?,
                value: self.expr(value)?,
            },
            tree::Node::Block { statements } => syntax::StmtKind::Block(self.block(statements)?),
            tree::Node::UnsafeBlock { statements } => {
                syntax::StmtKind::Unsafe(self.block(statements)?)
            }
            tree::Node::If {
                condition,
                body,
                else_if_blocks,
                else_block,
            } => {
                let mut else_if = Vec::new();
                for branch in else_if_blocks {
                    let tree::Node::If {
                        condition, body, ..
                    } = branch
                    else {
                        return Err(Diagnostic::error(
                            "tree else-if branch was not represented by an if node",
                        )
                        .with_code("E9004"));
                    };
                    else_if.push((self.expr(condition)?, self.block(body)?));
                }
                syntax::StmtKind::If(syntax::IfExpr {
                    condition: self.expr(condition)?,
                    then_block: self.block(body)?,
                    else_if,
                    else_block: else_block
                        .as_deref()
                        .map(|body| self.block(body))
                        .transpose()?,
                })
            }
            tree::Node::Match { value, cases } => {
                let mut converted = Vec::new();
                for case in cases {
                    let pattern = match &case.pattern {
                        tree::MatchPattern::EnumVariant {
                            enum_type,
                            variant,
                            bindings,
                        } => syntax::Pattern::EnumVariant {
                            enum_type: enum_type.as_ref().map(|ty| self.ty(ty)),
                            variant: variant.clone(),
                            bindings: bindings.clone(),
                        },
                        tree::MatchPattern::Struct {
                            struct_type,
                            fields,
                        } => syntax::Pattern::Struct {
                            ty: self.ty(struct_type),
                            fields: fields
                                .iter()
                                .map(|field| syntax::PatternField {
                                    name: field.field_name.clone(),
                                    binding: field.binding.clone(),
                                })
                                .collect(),
                        },
                    };
                    let id = self.id();
                    converted.push(syntax::MatchCase {
                        id,
                        span: self.span,
                        pattern,
                        body: self.block(&case.body)?,
                    });
                }
                syntax::StmtKind::Match(syntax::MatchExpr {
                    value: self.expr(value)?,
                    cases: converted,
                })
            }
            tree::Node::For {
                init,
                condition,
                update,
                body,
            } => syntax::StmtKind::For(syntax::ForStmt {
                initializer: init
                    .as_deref()
                    .map(|statement| self.stmt(statement).map(Box::new))
                    .transpose()?,
                condition: condition
                    .as_deref()
                    .map(|expression| self.expr(expression))
                    .transpose()?,
                update: update
                    .as_deref()
                    .map(|statement| self.stmt(statement).map(Box::new))
                    .transpose()?,
                body: self.block(body)?,
            }),
            tree::Node::Defer(expression) => syntax::StmtKind::Defer(self.expr(expression)?),
            tree::Node::Return(expression) => syntax::StmtKind::Return(
                expression
                    .as_deref()
                    .map(|expression| self.expr(expression))
                    .transpose()?,
            ),
            tree::Node::Print(expression) => syntax::StmtKind::Print(self.expr(expression)?),
            tree::Node::Input => syntax::StmtKind::Input,
            tree::Node::FunctionDeclaration { lambda: false, .. }
            | tree::Node::GenericFunctionDeclaration { lambda: false, .. }
            | tree::Node::ExternFunctionDeclaration { .. }
            | tree::Node::TestDeclaration { .. }
            | tree::Node::TypeAliasDeclaration { .. }
            | tree::Node::StructDeclaration { .. }
            | tree::Node::GenericStructDeclaration { .. }
            | tree::Node::EnumDeclaration { .. }
            | tree::Node::GenericEnumDeclaration { .. }
            | tree::Node::TraitDeclaration { .. }
            | tree::Node::ShapeDeclaration { .. }
            | tree::Node::AttachDeclaration { .. }
            | tree::Node::ConformDeclaration { .. }
            | tree::Node::ImplDeclaration { .. } => syntax::StmtKind::Declaration(Box::new(
                self.top_level(node, syntax::Visibility::Private)?,
            )),
            other => syntax::StmtKind::Expression(self.expr(other)?),
        };
        let id = self.id();
        Ok(syntax::Stmt {
            id,
            span: self.span,
            kind,
        })
    }

    fn local_decl(
        &mut self,
        name: &str,
        ty: &tree::Type,
        initializer: Option<&tree::Node>,
    ) -> Result<syntax::LocalDecl, Diagnostic> {
        let (is_const, ty) = split_binding_const(ty);
        Ok(syntax::LocalDecl {
            name: name.to_string(),
            is_const,
            ty: self.ty(ty),
            initializer: initializer
                .map(|expression| self.expr(expression))
                .transpose()?,
        })
    }

    fn expr(&mut self, node: &tree::Node) -> Result<syntax::Expr, Diagnostic> {
        let kind = match node {
            tree::Node::Literal(literal) => syntax::ExprKind::Literal(match literal {
                tree::Literal::Integer(value) => syntax::Literal::Integer(*value),
                tree::Literal::Long(value) => syntax::Literal::Long(*value),
                tree::Literal::Float(value) => syntax::Literal::Float(*value),
                tree::Literal::Double(value) => syntax::Literal::Double(*value),
                tree::Literal::String(value) => syntax::Literal::String(value.clone()),
                tree::Literal::Boolean(value) => syntax::Literal::Boolean(*value),
                tree::Literal::Char(value) => syntax::Literal::Char(*value),
            }),
            tree::Node::Identifier(name) => syntax::ExprKind::Name(syntax::Path::single(name)),
            tree::Node::BinaryOp {
                left,
                operator,
                right,
            } => syntax::ExprKind::Binary {
                left: Box::new(self.expr(left)?),
                operator: binary_operator(operator.clone()),
                right: Box::new(self.expr(right)?),
            },
            tree::Node::UnaryOp { operator, operand } => syntax::ExprKind::Unary {
                operator: unary_operator(operator.clone()),
                operand: Box::new(self.expr(operand)?),
            },
            tree::Node::FunctionCall {
                name,
                type_arguments,
                arguments,
                ..
            } => syntax::ExprKind::Call {
                callee: Box::new(self.name_expr(name)),
                type_arguments: type_arguments.iter().map(|ty| self.ty(ty)).collect(),
                argument_groups: arguments
                    .iter()
                    .map(|group| {
                        group
                            .iter()
                            .map(|argument| self.expr(argument))
                            .collect::<Result<Vec<_>, _>>()
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            },
            tree::Node::Access { nodes } => return self.access_expr(nodes),
            tree::Node::StructInitialization { _type, fields } => syntax::ExprKind::StructInit {
                ty: self.ty(_type),
                fields: fields
                    .iter()
                    .map(|(name, value)| Ok((name.clone(), self.expr(value)?)))
                    .collect::<Result<Vec<_>, Diagnostic>>()?,
            },
            tree::Node::StaticFunctionCall {
                _type,
                name,
                arguments,
                ..
            } => syntax::ExprKind::StaticCall {
                ty: self.ty(_type),
                name: name.clone(),
                arguments: arguments
                    .iter()
                    .map(|argument| self.expr(argument))
                    .collect::<Result<Vec<_>, _>>()?,
            },
            tree::Node::ArrayInit { elements } => syntax::ExprKind::Array(
                elements
                    .iter()
                    .map(|element| self.expr(element))
                    .collect::<Result<Vec<_>, _>>()?,
            ),
            tree::Node::FunctionDeclaration {
                parameters,
                return_type,
                body,
                lambda: true,
                ..
            } => syntax::ExprKind::Lambda(syntax::LambdaExpr {
                parameters: self.parameters(parameters),
                return_type: self.ty(return_type),
                body: self.block(body)?,
            }),
            tree::Node::GenericFunctionDeclaration { lambda: true, .. } => {
                return Err(Diagnostic::error(
                    "generic lambdas are not supported during specialization conversion",
                )
                .with_code("E9005"));
            }
            tree::Node::Block { statements } => syntax::ExprKind::Block(self.block(statements)?),
            other => {
                return Err(Diagnostic::error(format!(
                    "expected expression in specialization conversion, found {other:?}"
                ))
                .with_code("E9006"));
            }
        };

        let id = self.id();
        Ok(syntax::Expr {
            id,
            span: self.span,
            kind,
        })
    }

    fn name_expr(&mut self, name: &str) -> syntax::Expr {
        let id = self.id();
        syntax::Expr {
            id,
            span: self.span,
            kind: syntax::ExprKind::Name(syntax::Path::single(name)),
        }
    }

    fn access_expr(&mut self, nodes: &[tree::Node]) -> Result<syntax::Expr, Diagnostic> {
        let Some((first, rest)) = nodes.split_first() else {
            return Err(Diagnostic::error("empty access expression").with_code("E9007"));
        };
        let mut receiver = self.expr(first)?;
        for step in rest {
            let kind = match step {
                tree::Node::ArrayAccess { coordinates } => syntax::ExprKind::Index {
                    receiver: Box::new(receiver),
                    coordinates: coordinates
                        .iter()
                        .map(|coordinate| self.expr(coordinate))
                        .collect::<Result<Vec<_>, _>>()?,
                },
                tree::Node::SliceAccess { start, end } => syntax::ExprKind::Slice {
                    receiver: Box::new(receiver),
                    start: start
                        .as_deref()
                        .map(|expression| self.expr(expression).map(Box::new))
                        .transpose()?,
                    end: end
                        .as_deref()
                        .map(|expression| self.expr(expression).map(Box::new))
                        .transpose()?,
                },
                tree::Node::MemberAccess { member, .. } => match member.as_ref() {
                    tree::Node::Identifier(name) => syntax::ExprKind::Field {
                        receiver: Box::new(receiver),
                        name: name.clone(),
                    },
                    tree::Node::FunctionCall {
                        name,
                        type_arguments,
                        arguments,
                        ..
                    } => {
                        let field_id = self.id();
                        let callee = syntax::Expr {
                            id: field_id,
                            span: self.span,
                            kind: syntax::ExprKind::Field {
                                receiver: Box::new(receiver),
                                name: name.clone(),
                            },
                        };
                        syntax::ExprKind::Call {
                            callee: Box::new(callee),
                            type_arguments: type_arguments.iter().map(|ty| self.ty(ty)).collect(),
                            argument_groups: arguments
                                .iter()
                                .map(|group| {
                                    group
                                        .iter()
                                        .map(|argument| self.expr(argument))
                                        .collect::<Result<Vec<_>, _>>()
                                })
                                .collect::<Result<Vec<_>, _>>()?,
                        }
                    }
                    other => {
                        return Err(Diagnostic::error(format!(
                            "unsupported member access in specialization conversion: {other:?}"
                        ))
                        .with_code("E9008"));
                    }
                },
                tree::Node::Dereference { .. } => syntax::ExprKind::Unary {
                    operator: syntax::UnaryOperator::Dereference,
                    operand: Box::new(receiver),
                },
                other => {
                    return Err(Diagnostic::error(format!(
                        "unsupported access step in specialization conversion: {other:?}"
                    ))
                    .with_code("E9009"));
                }
            };
            let id = self.id();
            receiver = syntax::Expr {
                id,
                span: self.span,
                kind,
            };
        }
        Ok(receiver)
    }

    fn ty(&mut self, ty: &tree::Type) -> syntax::TypeSyntax {
        let kind = match ty {
            tree::Type::Void => syntax::TypeSyntaxKind::Builtin(syntax::BuiltinType::Void),
            tree::Type::Byte => syntax::TypeSyntaxKind::Builtin(syntax::BuiltinType::Byte),
            tree::Type::Short => syntax::TypeSyntaxKind::Builtin(syntax::BuiltinType::Short),
            tree::Type::Int => syntax::TypeSyntaxKind::Builtin(syntax::BuiltinType::Int),
            tree::Type::Long => syntax::TypeSyntaxKind::Builtin(syntax::BuiltinType::Long),
            tree::Type::Float => syntax::TypeSyntaxKind::Builtin(syntax::BuiltinType::Float),
            tree::Type::Double => syntax::TypeSyntaxKind::Builtin(syntax::BuiltinType::Double),
            tree::Type::String => syntax::TypeSyntaxKind::Builtin(syntax::BuiltinType::String),
            tree::Type::Boolean => syntax::TypeSyntaxKind::Builtin(syntax::BuiltinType::Boolean),
            tree::Type::Char => syntax::TypeSyntaxKind::Builtin(syntax::BuiltinType::Char),
            tree::Type::Allocator => {
                syntax::TypeSyntaxKind::Builtin(syntax::BuiltinType::Allocator)
            }
            tree::Type::Arena => syntax::TypeSyntaxKind::Builtin(syntax::BuiltinType::Arena),
            tree::Type::Const { inner } | tree::Type::BindingConst { inner } => {
                syntax::TypeSyntaxKind::Const(Box::new(self.ty(inner)))
            }
            tree::Type::Array {
                elem_type,
                dimensions,
            } => syntax::TypeSyntaxKind::Array {
                element: Box::new(self.ty(elem_type)),
                dimensions: dimensions
                    .iter()
                    .filter_map(|dimension| self.expr(dimension).ok())
                    .collect(),
            },
            tree::Type::Reference {
                target_type,
                mutable,
            } => syntax::TypeSyntaxKind::Reference {
                target: Box::new(self.ty(target_type)),
                mutable: *mutable,
            },
            tree::Type::Pointer { target_type } => {
                syntax::TypeSyntaxKind::Pointer(Box::new(self.ty(target_type)))
            }
            tree::Type::Slice { elem_type } => {
                syntax::TypeSyntaxKind::Slice(Box::new(self.ty(elem_type)))
            }
            tree::Type::GenericInstance {
                base,
                type_arguments,
            } => syntax::TypeSyntaxKind::Named {
                path: syntax::Path::from_qualified(base),
                arguments: type_arguments.iter().map(|ty| self.ty(ty)).collect(),
            },
            tree::Type::Union(types) => {
                syntax::TypeSyntaxKind::Union(types.iter().map(|ty| self.ty(ty)).collect())
            }
            tree::Type::Intersection(types) => {
                syntax::TypeSyntaxKind::Intersection(types.iter().map(|ty| self.ty(ty)).collect())
            }
            tree::Type::Custom(name) => syntax::TypeSyntaxKind::Named {
                path: syntax::Path::from_qualified(name),
                arguments: Vec::new(),
            },
            tree::Type::Function {
                parameters,
                return_type,
            } => syntax::TypeSyntaxKind::Function {
                parameters: parameters.iter().map(|ty| self.ty(ty)).collect(),
                result: Box::new(self.ty(return_type)),
            },
            tree::Type::MutSelf => syntax::TypeSyntaxKind::SelfType { mutable: true },
            tree::Type::SkSelf => syntax::TypeSyntaxKind::SelfType { mutable: false },
        };
        let id = self.id();
        syntax::TypeSyntax {
            id,
            span: self.span,
            kind,
        }
    }
}

struct SyntaxToTree;

impl SyntaxToTree {
    fn module(&self, module: &syntax::Module) -> tree::Node {
        let mut statements = Vec::new();
        if let Some(name) = &module.name {
            statements.push(tree::Node::Module {
                name: name.qualified_name(),
            });
        }
        statements.extend(module.entries.iter().map(|entry| self.top_level(entry)));
        statements.push(tree::Node::End);
        tree::Node::Program { statements }
    }

    fn top_level(&self, entry: &syntax::TopLevel) -> tree::Node {
        let declaration = match &entry.kind {
            syntax::TopLevelKind::Import(import) => tree::Node::Import {
                name: import.module.qualified_name(),
            },
            syntax::TopLevelKind::TypeAlias(alias) => {
                let (generic_params, generic_bounds, subtype_bounds) =
                    self.generic_parts(&alias.generic_parameters);
                tree::Node::TypeAliasDeclaration {
                    name: alias.name.clone(),
                    generic_params,
                    generic_bounds,
                    subtype_bounds,
                    target_type: self.ty(&alias.target),
                }
            }
            syntax::TopLevelKind::Struct(declaration) => self.struct_decl(declaration),
            syntax::TopLevelKind::Enum(declaration) => self.enum_decl(declaration),
            syntax::TopLevelKind::Trait(declaration) => {
                let (generic_params, generic_bounds, subtype_bounds) =
                    self.generic_parts(&declaration.generic_parameters);
                tree::Node::TraitDeclaration {
                    name: declaration.name.clone(),
                    generic_params,
                    generic_bounds,
                    subtype_bounds,
                    supertraits: declaration
                        .supertraits
                        .iter()
                        .map(syntax::Path::qualified_name)
                        .collect(),
                    methods: declaration
                        .methods
                        .iter()
                        .map(|method| self.trait_method(method))
                        .collect(),
                }
            }
            syntax::TopLevelKind::Shape(declaration) => tree::Node::ShapeDeclaration {
                name: declaration.name.clone(),
                methods: declaration
                    .methods
                    .iter()
                    .map(|method| self.trait_method(method))
                    .collect(),
            },
            syntax::TopLevelKind::Attach(declaration) => {
                let (generic_params, generic_bounds, subtype_bounds) =
                    self.generic_parts(&declaration.generic_parameters);
                tree::Node::AttachDeclaration {
                    generic_params,
                    generic_bounds,
                    subtype_bounds,
                    target_type: self.ty(&declaration.target),
                    functions: declaration
                        .methods
                        .iter()
                        .map(|method| self.function(method, false))
                        .collect(),
                }
            }
            syntax::TopLevelKind::Conformance(declaration) => {
                let (generic_params, generic_bounds, subtype_bounds) =
                    self.generic_parts(&declaration.generic_parameters);
                tree::Node::ConformDeclaration {
                    generic_params,
                    generic_bounds,
                    subtype_bounds,
                    trait_type: declaration
                        .traits
                        .first()
                        .map(|ty| self.ty(ty))
                        .unwrap_or_else(|| tree::Type::Custom("<missing-trait>".to_string())),
                    target_type: self.ty(&declaration.target),
                    functions: declaration
                        .methods
                        .iter()
                        .map(|method| self.function(method, false))
                        .collect(),
                }
            }
            syntax::TopLevelKind::Implementation(declaration) => {
                let (generic_params, generic_bounds, subtype_bounds) =
                    self.generic_parts(&declaration.generic_parameters);
                tree::Node::ImplDeclaration {
                    generic_params,
                    generic_bounds,
                    subtype_bounds,
                    trait_types: declaration.traits.iter().map(|ty| self.ty(ty)).collect(),
                    target_type: self.ty(&declaration.target),
                }
            }
            syntax::TopLevelKind::Function(function) => self.function(function, false),
            syntax::TopLevelKind::ExternFunction(function) => {
                tree::Node::ExternFunctionDeclaration {
                    name: function.name.clone(),
                    parameters: self.parameters(&function.parameters),
                    return_type: self.ty(&function.return_type),
                }
            }
            syntax::TopLevelKind::Global(global) => self.local(global),
            syntax::TopLevelKind::Test(test) => tree::Node::TestDeclaration {
                name: test.name.clone(),
                body: self.block_statements(&test.body),
            },
            syntax::TopLevelKind::Statement(statement) => self.stmt(statement),
        };

        if entry.visibility == syntax::Visibility::Public
            && !matches!(declaration, tree::Node::Import { .. })
        {
            tree::Node::Export {
                declaration: Box::new(declaration),
            }
        } else {
            declaration
        }
    }

    fn struct_decl(&self, declaration: &syntax::StructDecl) -> tree::Node {
        let fields = declaration
            .fields
            .iter()
            .map(|field| {
                let ty = self.ty(&field.ty);
                (
                    field.name.clone(),
                    if field.is_const {
                        tree::Type::Const {
                            inner: Box::new(ty),
                        }
                    } else {
                        ty
                    },
                )
            })
            .collect();
        let functions = declaration
            .methods
            .iter()
            .map(|method| self.function(method, false))
            .collect();
        if declaration.generic_parameters.is_empty() {
            tree::Node::StructDeclaration {
                name: declaration.name.clone(),
                fields,
                functions,
            }
        } else {
            let (generic_params, generic_bounds, subtype_bounds) =
                self.generic_parts(&declaration.generic_parameters);
            tree::Node::GenericStructDeclaration {
                name: declaration.name.clone(),
                generic_params,
                generic_bounds,
                subtype_bounds,
                fields,
                functions,
            }
        }
    }

    fn enum_decl(&self, declaration: &syntax::EnumDecl) -> tree::Node {
        let variants = declaration
            .variants
            .iter()
            .map(|variant| tree::EnumVariant {
                name: variant.name.clone(),
                payload_types: variant.payload.iter().map(|ty| self.ty(ty)).collect(),
            })
            .collect();
        let functions = declaration
            .methods
            .iter()
            .map(|method| self.function(method, false))
            .collect();
        if declaration.generic_parameters.is_empty() {
            tree::Node::EnumDeclaration {
                name: declaration.name.clone(),
                variants,
                functions,
            }
        } else {
            let (generic_params, generic_bounds, subtype_bounds) =
                self.generic_parts(&declaration.generic_parameters);
            tree::Node::GenericEnumDeclaration {
                name: declaration.name.clone(),
                generic_params,
                generic_bounds,
                subtype_bounds,
                variants,
                functions,
            }
        }
    }

    fn function(&self, function: &syntax::FunctionDecl, lambda: bool) -> tree::Node {
        let parameters = self.parameters(&function.parameters);
        let return_type = self.ty(&function.return_type);
        let body = self.block_statements(&function.body);
        if function.generic_parameters.is_empty() {
            tree::Node::FunctionDeclaration {
                name: function.name.clone(),
                parameters,
                return_type,
                body,
                lambda,
            }
        } else {
            let (generic_params, generic_bounds, subtype_bounds) =
                self.generic_parts(&function.generic_parameters);
            tree::Node::GenericFunctionDeclaration {
                name: function.name.clone(),
                generic_params,
                generic_bounds,
                subtype_bounds,
                parameters,
                return_type,
                body,
                lambda,
            }
        }
    }

    fn trait_method(&self, method: &syntax::TraitMethod) -> tree::TraitMethodSignature {
        tree::TraitMethodSignature {
            name: method.name.clone(),
            parameters: self.parameters(&method.parameters),
            return_type: self.ty(&method.return_type),
            default_body: method
                .default_body
                .as_ref()
                .map(|body| self.block_statements(body)),
        }
    }

    fn generic_parts(&self, parameters: &[syntax::GenericParameter]) -> TreeGenericParts {
        let mut names = Vec::new();
        let mut capabilities = HashMap::new();
        let mut subtype_bounds = HashMap::new();
        for parameter in parameters {
            names.push(parameter.name.clone());
            if !parameter.capabilities.is_empty() {
                capabilities.insert(
                    parameter.name.clone(),
                    parameter
                        .capabilities
                        .iter()
                        .map(syntax::Path::qualified_name)
                        .collect(),
                );
            }
            if parameter.lower_bound.is_some() || parameter.upper_bound.is_some() {
                subtype_bounds.insert(
                    parameter.name.clone(),
                    tree::SubtypeBounds {
                        lower: parameter.lower_bound.as_ref().map(|ty| self.ty(ty)),
                        upper: parameter.upper_bound.as_ref().map(|ty| self.ty(ty)),
                    },
                );
            }
        }
        (names, capabilities, subtype_bounds)
    }

    fn parameters(&self, parameters: &[syntax::Parameter]) -> Vec<(String, tree::Type)> {
        parameters
            .iter()
            .map(|parameter| match &parameter.kind {
                syntax::ParameterKind::Named { name, is_const, ty } => {
                    let ty = self.ty(ty);
                    (
                        name.clone(),
                        if *is_const {
                            tree::Type::BindingConst {
                                inner: Box::new(ty),
                            }
                        } else {
                            ty
                        },
                    )
                }
                syntax::ParameterKind::Receiver { mutable: true, .. } => {
                    ("self".to_string(), tree::Type::MutSelf)
                }
                syntax::ParameterKind::Receiver { mutable: false, .. } => {
                    ("self".to_string(), tree::Type::SkSelf)
                }
            })
            .collect()
    }

    fn block_statements(&self, block: &syntax::Block) -> Vec<tree::Node> {
        block
            .statements
            .iter()
            .map(|statement| self.stmt(statement))
            .collect()
    }

    fn stmt(&self, statement: &syntax::Stmt) -> tree::Node {
        match &statement.kind {
            syntax::StmtKind::Local(local) => self.local(local),
            syntax::StmtKind::StructDestructure(pattern) => tree::Node::StructDestructure {
                struct_type: self.ty(&pattern.ty),
                fields: pattern
                    .fields
                    .iter()
                    .map(|field| tree::StructPatternField {
                        field_name: field.name.clone(),
                        binding: field.binding.clone(),
                    })
                    .collect(),
                value: Box::new(self.expr(&pattern.value)),
                metadata: tree::Metadata::EMPTY,
            },
            syntax::StmtKind::Assignment { target, value } => tree::Node::Assignment {
                var: Box::new(self.expr(target)),
                value: Box::new(self.expr(value)),
                metadata: tree::Metadata::EMPTY,
            },
            syntax::StmtKind::Expression(expression) => self.expr(expression),
            syntax::StmtKind::Return(expression) => tree::Node::Return(
                expression
                    .as_ref()
                    .map(|expression| Box::new(self.expr(expression))),
            ),
            syntax::StmtKind::Defer(expression) => {
                tree::Node::Defer(Box::new(self.expr(expression)))
            }
            syntax::StmtKind::Print(expression) => {
                tree::Node::Print(Box::new(self.expr(expression)))
            }
            syntax::StmtKind::Input => tree::Node::Input,
            syntax::StmtKind::Declaration(declaration) => self.top_level(declaration),
            syntax::StmtKind::Block(block) => tree::Node::Block {
                statements: self.block_statements(block),
            },
            syntax::StmtKind::Unsafe(block) => tree::Node::UnsafeBlock {
                statements: self.block_statements(block),
            },
            syntax::StmtKind::If(expression) => tree::Node::If {
                condition: Box::new(self.expr(&expression.condition)),
                body: self.block_statements(&expression.then_block),
                else_if_blocks: expression
                    .else_if
                    .iter()
                    .map(|(condition, body)| tree::Node::If {
                        condition: Box::new(self.expr(condition)),
                        body: self.block_statements(body),
                        else_if_blocks: Vec::new(),
                        else_block: None,
                    })
                    .collect(),
                else_block: expression
                    .else_block
                    .as_ref()
                    .map(|body| self.block_statements(body)),
            },
            syntax::StmtKind::Match(expression) => tree::Node::Match {
                value: Box::new(self.expr(&expression.value)),
                cases: expression
                    .cases
                    .iter()
                    .map(|case| tree::MatchCase {
                        pattern: match &case.pattern {
                            syntax::Pattern::EnumVariant {
                                enum_type,
                                variant,
                                bindings,
                            } => tree::MatchPattern::EnumVariant {
                                enum_type: enum_type.as_ref().map(|ty| self.ty(ty)),
                                variant: variant.clone(),
                                bindings: bindings.clone(),
                            },
                            syntax::Pattern::Struct { ty, fields } => tree::MatchPattern::Struct {
                                struct_type: self.ty(ty),
                                fields: fields
                                    .iter()
                                    .map(|field| tree::StructPatternField {
                                        field_name: field.name.clone(),
                                        binding: field.binding.clone(),
                                    })
                                    .collect(),
                            },
                        },
                        body: self.block_statements(&case.body),
                    })
                    .collect(),
            },
            syntax::StmtKind::For(statement) => tree::Node::For {
                init: statement
                    .initializer
                    .as_deref()
                    .map(|statement| Box::new(self.stmt(statement))),
                condition: statement
                    .condition
                    .as_ref()
                    .map(|expression| Box::new(self.expr(expression))),
                update: statement
                    .update
                    .as_deref()
                    .map(|statement| Box::new(self.stmt(statement))),
                body: self.block_statements(&statement.body),
            },
        }
    }

    fn local(&self, local: &syntax::LocalDecl) -> tree::Node {
        let ty = self.ty(&local.ty);
        tree::Node::VariableDeclaration {
            var_type: if local.is_const {
                tree::Type::BindingConst {
                    inner: Box::new(ty),
                }
            } else {
                ty
            },
            name: local.name.clone(),
            value: local
                .initializer
                .as_ref()
                .map(|expression| Box::new(self.expr(expression))),
            metadata: tree::Metadata::EMPTY,
        }
    }

    fn expr(&self, expression: &syntax::Expr) -> tree::Node {
        if let Some(parts) = self.access_parts(expression) {
            return if parts.len() == 1
                && matches!(parts.first(), Some(tree::Node::FunctionCall { .. }))
            {
                parts.into_iter().next().unwrap()
            } else {
                tree::Node::Access { nodes: parts }
            };
        }

        match &expression.kind {
            syntax::ExprKind::Literal(literal) => tree::Node::Literal(match literal {
                syntax::Literal::Integer(value) => tree::Literal::Integer(*value),
                syntax::Literal::Long(value) => tree::Literal::Long(*value),
                syntax::Literal::Float(value) => tree::Literal::Float(*value),
                syntax::Literal::Double(value) => tree::Literal::Double(*value),
                syntax::Literal::String(value) => tree::Literal::String(value.clone()),
                syntax::Literal::Boolean(value) => tree::Literal::Boolean(*value),
                syntax::Literal::Char(value) => tree::Literal::Char(*value),
            }),
            syntax::ExprKind::Unary { operator, operand } => tree::Node::UnaryOp {
                operator: match operator {
                    syntax::UnaryOperator::Plus => tree::UnaryOperator::Plus,
                    syntax::UnaryOperator::Minus => tree::UnaryOperator::Minus,
                    syntax::UnaryOperator::Not => tree::UnaryOperator::Negate,
                    syntax::UnaryOperator::AddressOf => tree::UnaryOperator::AddressOf,
                    syntax::UnaryOperator::AddressOfMut => tree::UnaryOperator::AddressOfMut,
                    syntax::UnaryOperator::Dereference => {
                        unreachable!("dereference expressions are converted through access_parts")
                    }
                },
                operand: Box::new(self.expr(operand)),
            },
            syntax::ExprKind::Binary {
                left,
                operator,
                right,
            } => tree::Node::BinaryOp {
                left: Box::new(self.expr(left)),
                operator: tree_binary_operator(*operator),
                right: Box::new(self.expr(right)),
            },
            syntax::ExprKind::StructInit { ty, fields } => tree::Node::StructInitialization {
                _type: self.ty(ty),
                fields: fields
                    .iter()
                    .map(|(name, value)| (name.clone(), self.expr(value)))
                    .collect(),
            },
            syntax::ExprKind::StaticCall {
                ty,
                name,
                arguments,
            } => tree::Node::StaticFunctionCall {
                _type: self.ty(ty),
                name: name.clone(),
                arguments: arguments
                    .iter()
                    .map(|argument| self.expr(argument))
                    .collect(),
                metadata: tree::Metadata::EMPTY,
            },
            syntax::ExprKind::Array(elements) => tree::Node::ArrayInit {
                elements: elements.iter().map(|element| self.expr(element)).collect(),
            },
            syntax::ExprKind::Lambda(lambda) => tree::Node::FunctionDeclaration {
                name: String::new(),
                parameters: self.parameters(&lambda.parameters),
                return_type: self.ty(&lambda.return_type),
                body: self.block_statements(&lambda.body),
                lambda: true,
            },
            syntax::ExprKind::Block(block) => tree::Node::Block {
                statements: self.block_statements(block),
            },
            syntax::ExprKind::Name(_)
            | syntax::ExprKind::Call { .. }
            | syntax::ExprKind::Field { .. }
            | syntax::ExprKind::Index { .. }
            | syntax::ExprKind::Slice { .. } => {
                unreachable!("access-like expressions are converted before this match")
            }
        }
    }

    fn access_parts(&self, expression: &syntax::Expr) -> Option<Vec<tree::Node>> {
        match &expression.kind {
            syntax::ExprKind::Name(path) if path.segments.len() == 1 => {
                Some(vec![tree::Node::Identifier(path.qualified_name())])
            }
            syntax::ExprKind::Call {
                callee,
                type_arguments,
                argument_groups,
            } => match &callee.kind {
                syntax::ExprKind::Name(path) if path.segments.len() == 1 => {
                    Some(vec![self.function_call(
                        &path.qualified_name(),
                        type_arguments,
                        argument_groups,
                    )])
                }
                syntax::ExprKind::Field { receiver, name } => {
                    let mut parts = self.access_parts(receiver)?;
                    parts.push(tree::Node::MemberAccess {
                        member: Box::new(self.function_call(name, type_arguments, argument_groups)),
                        metadata: tree::Metadata::EMPTY,
                    });
                    Some(parts)
                }
                _ => None,
            },
            syntax::ExprKind::Field { receiver, name } => {
                let mut parts = self.access_parts(receiver)?;
                parts.push(tree::Node::MemberAccess {
                    member: Box::new(tree::Node::Identifier(name.clone())),
                    metadata: tree::Metadata::EMPTY,
                });
                Some(parts)
            }
            syntax::ExprKind::Index {
                receiver,
                coordinates,
            } => {
                let mut parts = self.access_parts(receiver)?;
                parts.push(tree::Node::ArrayAccess {
                    coordinates: coordinates
                        .iter()
                        .map(|coordinate| self.expr(coordinate))
                        .collect(),
                });
                Some(parts)
            }
            syntax::ExprKind::Slice {
                receiver,
                start,
                end,
            } => {
                let mut parts = self.access_parts(receiver)?;
                parts.push(tree::Node::SliceAccess {
                    start: start
                        .as_deref()
                        .map(|expression| Box::new(self.expr(expression))),
                    end: end
                        .as_deref()
                        .map(|expression| Box::new(self.expr(expression))),
                });
                Some(parts)
            }
            syntax::ExprKind::Unary {
                operator: syntax::UnaryOperator::Dereference,
                operand,
            } => {
                let mut parts = self.access_parts(operand)?;
                parts.push(tree::Node::Dereference {
                    metadata: tree::Metadata::EMPTY,
                });
                Some(parts)
            }
            _ => None,
        }
    }

    fn function_call(
        &self,
        name: &str,
        type_arguments: &[syntax::TypeSyntax],
        argument_groups: &[Vec<syntax::Expr>],
    ) -> tree::Node {
        tree::Node::FunctionCall {
            name: name.to_string(),
            type_arguments: type_arguments.iter().map(|ty| self.ty(ty)).collect(),
            arguments: argument_groups
                .iter()
                .map(|group| group.iter().map(|argument| self.expr(argument)).collect())
                .collect(),
            metadata: tree::Metadata::EMPTY,
        }
    }

    fn ty(&self, ty: &syntax::TypeSyntax) -> tree::Type {
        match &ty.kind {
            syntax::TypeSyntaxKind::Builtin(builtin) => match builtin {
                syntax::BuiltinType::Void => tree::Type::Void,
                syntax::BuiltinType::Byte => tree::Type::Byte,
                syntax::BuiltinType::Short => tree::Type::Short,
                syntax::BuiltinType::Int => tree::Type::Int,
                syntax::BuiltinType::Long => tree::Type::Long,
                syntax::BuiltinType::Float => tree::Type::Float,
                syntax::BuiltinType::Double => tree::Type::Double,
                syntax::BuiltinType::String => tree::Type::String,
                syntax::BuiltinType::Boolean => tree::Type::Boolean,
                syntax::BuiltinType::Char => tree::Type::Char,
                syntax::BuiltinType::Allocator => tree::Type::Allocator,
                syntax::BuiltinType::Arena => tree::Type::Arena,
            },
            syntax::TypeSyntaxKind::Named { path, arguments } => {
                if arguments.is_empty() {
                    tree::Type::Custom(path.qualified_name())
                } else {
                    tree::Type::GenericInstance {
                        base: path.qualified_name(),
                        type_arguments: arguments.iter().map(|ty| self.ty(ty)).collect(),
                    }
                }
            }
            syntax::TypeSyntaxKind::Const(inner) => tree::Type::Const {
                inner: Box::new(self.ty(inner)),
            },
            syntax::TypeSyntaxKind::Array {
                element,
                dimensions,
            } => tree::Type::Array {
                elem_type: Box::new(self.ty(element)),
                dimensions: dimensions
                    .iter()
                    .map(|dimension| self.expr(dimension))
                    .collect(),
            },
            syntax::TypeSyntaxKind::Reference { target, mutable } => tree::Type::Reference {
                target_type: Box::new(self.ty(target)),
                mutable: *mutable,
            },
            syntax::TypeSyntaxKind::Pointer(target) => tree::Type::Pointer {
                target_type: Box::new(self.ty(target)),
            },
            syntax::TypeSyntaxKind::Slice(element) => tree::Type::Slice {
                elem_type: Box::new(self.ty(element)),
            },
            syntax::TypeSyntaxKind::Union(types) => {
                tree::Type::Union(types.iter().map(|ty| self.ty(ty)).collect())
            }
            syntax::TypeSyntaxKind::Intersection(types) => {
                tree::Type::Intersection(types.iter().map(|ty| self.ty(ty)).collect())
            }
            syntax::TypeSyntaxKind::Function { parameters, result } => tree::Type::Function {
                parameters: parameters.iter().map(|ty| self.ty(ty)).collect(),
                return_type: Box::new(self.ty(result)),
            },
            syntax::TypeSyntaxKind::SelfType { mutable: true } => tree::Type::MutSelf,
            syntax::TypeSyntaxKind::SelfType { mutable: false } => tree::Type::SkSelf,
        }
    }
}

fn split_binding_const(ty: &tree::Type) -> (bool, &tree::Type) {
    match ty {
        tree::Type::BindingConst { inner } => (true, inner),
        other => (false, other),
    }
}

fn binary_operator(operator: tree::Operator) -> syntax::BinaryOperator {
    match operator {
        tree::Operator::Add => syntax::BinaryOperator::Add,
        tree::Operator::Subtract => syntax::BinaryOperator::Subtract,
        tree::Operator::Multiply => syntax::BinaryOperator::Multiply,
        tree::Operator::Divide => syntax::BinaryOperator::Divide,
        tree::Operator::Mod => syntax::BinaryOperator::Modulo,
        tree::Operator::Power => syntax::BinaryOperator::Power,
        tree::Operator::Equals => syntax::BinaryOperator::Equals,
        tree::Operator::NotEquals => syntax::BinaryOperator::NotEquals,
        tree::Operator::LessThan => syntax::BinaryOperator::LessThan,
        tree::Operator::GreaterThan => syntax::BinaryOperator::GreaterThan,
        tree::Operator::LessThanOrEqual => syntax::BinaryOperator::LessThanOrEqual,
        tree::Operator::GreaterThanOrEqual => syntax::BinaryOperator::GreaterThanOrEqual,
        tree::Operator::And => syntax::BinaryOperator::And,
        tree::Operator::Or => syntax::BinaryOperator::Or,
    }
}

fn tree_binary_operator(operator: syntax::BinaryOperator) -> tree::Operator {
    match operator {
        syntax::BinaryOperator::Add => tree::Operator::Add,
        syntax::BinaryOperator::Subtract => tree::Operator::Subtract,
        syntax::BinaryOperator::Multiply => tree::Operator::Multiply,
        syntax::BinaryOperator::Divide => tree::Operator::Divide,
        syntax::BinaryOperator::Modulo => tree::Operator::Mod,
        syntax::BinaryOperator::Power => tree::Operator::Power,
        syntax::BinaryOperator::Equals => tree::Operator::Equals,
        syntax::BinaryOperator::NotEquals => tree::Operator::NotEquals,
        syntax::BinaryOperator::LessThan => tree::Operator::LessThan,
        syntax::BinaryOperator::GreaterThan => tree::Operator::GreaterThan,
        syntax::BinaryOperator::LessThanOrEqual => tree::Operator::LessThanOrEqual,
        syntax::BinaryOperator::GreaterThanOrEqual => tree::Operator::GreaterThanOrEqual,
        syntax::BinaryOperator::And => tree::Operator::And,
        syntax::BinaryOperator::Or => tree::Operator::Or,
    }
}

fn unary_operator(operator: tree::UnaryOperator) -> syntax::UnaryOperator {
    match operator {
        tree::UnaryOperator::Plus => syntax::UnaryOperator::Plus,
        tree::UnaryOperator::Minus => syntax::UnaryOperator::Minus,
        tree::UnaryOperator::Negate => syntax::UnaryOperator::Not,
        tree::UnaryOperator::AddressOf => syntax::UnaryOperator::AddressOf,
        tree::UnaryOperator::AddressOfMut => syntax::UnaryOperator::AddressOfMut,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_representative_program_through_typed_syntax() {
        let source = r#"
            struct Point { x: int; y: int; }
            function sum(point: Point): int { return point.x + point.y; }
            function main(): void { print(sum(Point { x: 2, y: 3 })); }
        "#;
        let tree = crate::specialization::tree::try_parse(source).unwrap();
        let syntax = from_tree(&tree, FileId::new(0), source.len()).unwrap();
        let rebuilt = to_tree(&syntax);

        crate::specialization::expand::prepare_program(&rebuilt).unwrap();
    }
}
